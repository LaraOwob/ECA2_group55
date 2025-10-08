"""Assignment 3 template code."""

# Standard library
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
import pandas as pd

# Third-party libraries
import matplotlib.pyplot as plt
import mujoco as mj
import numpy as np
import numpy.typing as npt
from mujoco import viewer
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from functools import partial
from deap import base, creator, tools

# Local libraries
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
    save_graph_as_json,
)
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.environments import OlympicArena
from ariel.simulation.environments import SimpleFlatWorld
from ariel.simulation.environments import RuggedTerrainWorld
from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.utils.runners import simple_runner
from ariel.utils.tracker import Tracker
from ariel.utils.video_recorder import VideoRecorder
import random as pyrand


# brain_only_gecko_experiment2.py
import os
import time
import csv
import numpy as np
import torch
import torch.nn as nn
import mujoco as mj
from mujoco import viewer
import cma

from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.simulation.environments import OlympicArena
from ariel.simulation.controllers.controller import Controller
from ariel.utils.runners import simple_runner
from ariel.utils.tracker import Tracker

# ---------- Config ----------
SPAWN_POSITIONS = [np.array([2.5, 0.0, 0.1]), np.array([0.5, 0.0, 0.1])]
TARGET_POS = np.array([4.5, 0.0, 0.55], dtype=np.float32)
DURATION = 10.0

# ---------- NN Controller ----------
class NNController(nn.Module):
    def __init__(self, nu: int, hidden_size: int = 32, freq_hz: float = 3.0):
        super().__init__()
        self.nu = nu
        self.freq_hz = freq_hz
        self.fc1 = nn.Linear(nu*2+4, hidden_size)  # simplified obs size
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, nu)
        self.act = nn.Tanh()

    def forward(self, qpos, qvel, t, target_pos, current_pos):
        w = 2 * np.pi * self.freq_hz
        time_feats = torch.tensor([np.sin(w*t), np.cos(w*t)], dtype=torch.float32)
        delta = target_pos[:2] - current_pos[:2]
        obs = torch.cat([qpos, qvel, time_feats, delta])
        x = self.act(self.fc1(obs))
        x = self.act(self.fc2(x))
        y = self.act(self.fc3(x))
        return y

# ---------- Helpers ----------
def flatten_params(net: nn.Module) -> np.ndarray:
    return np.concatenate([p.detach().cpu().numpy().ravel() for p in net.parameters()])

def unflatten_params(net: nn.Module, flat: np.ndarray) -> None:
    i = 0
    with torch.no_grad():
        for p in net.parameters():
            n = p.numel()
            p.copy_(torch.tensor(flat[i:i+n].reshape(p.shape), dtype=p.dtype))
            i += n

def nn_obj_control(nn_obj, model, data, target_pos):
    current_pos = torch.tensor(data.xpos[0], dtype=torch.float32)
    qpos = torch.tensor(data.qpos[:nn_obj.nu], dtype=torch.float32)
    qvel = torch.tensor(data.qvel[:nn_obj.nu], dtype=torch.float32)
    raw = nn_obj(qpos, qvel, data.time, torch.tensor(target_pos), current_pos).detach().numpy()
    global AMP, MID, CTRL_RANGE_ARR
    a = MID + AMP * np.clip(raw, -1, 1)
    return np.clip(a, CTRL_RANGE_ARR[:,0], CTRL_RANGE_ARR[:,1])

def evaluate_fitness_multiple_spawns(nn_obj, model, world, spawn_positions, duration):
    fitnesses = []
    for sp in spawn_positions:
        data = mj.MjData(model)
        mj.mj_resetData(model, data)
        data.qpos[:3] = sp  # assume first 3 entries correspond to root
        data.qvel[:] = 0.0
        mj.mj_step(model, data, nstep=10)
        # Setup tracker
        tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_BODY, name_to_bind="core")
        tracker.setup(world.spec, data)
        ctrl = Controller(controller_callback_function=lambda m,d: nn_obj_control(nn_obj,m,d,TARGET_POS),
                          tracker=tracker)
        mj.set_mjcb_control(ctrl.set_control)
        simple_runner(model, data, duration=duration)
        traj = tracker.history.get("xpos", {})
        if not traj:
            fitnesses.append(-1e6)
            continue
        key = next(iter(traj))
        traj_arr = np.asarray(traj[key])
        if traj_arr.shape[0]<2:
            fitnesses.append(-1e6)
            continue
        progress = traj_arr[-1,0] - traj_arr[0,0]  # simple forward progress
        dist_to_target = np.linalg.norm(traj_arr[-1,:2]-TARGET_POS[:2])
        proximity_bonus = max(0.0, 3.0 - dist_to_target)
        fitnesses.append(progress + proximity_bonus)
    return float(np.mean(fitnesses))

# ---------- Training ----------
def train_nn_cma_multiple_spawns_to_csv(nn_obj, model, world, spawn_positions,
                                        generations=50, pop_size=16, sigma=0.5, duration=10.0,
                                        output_csv="./results.csv", seed=42):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    x0 = flatten_params(nn_obj)
    es = cma.CMAEvolutionStrategy(x0, sigma, {'popsize': pop_size, 'seed': seed})

    best_theta = None
    best_fitness = -np.inf

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["generation", "best_fitness", "mean_fitness", "median_fitness", "elapsed_time"]
        header += [f"theta_{i}" for i in range(len(x0))]
        writer.writerow(header)

        start_time = time.time()
        for gen in range(generations):
            candidates = es.ask()
            fitnesses = []
            for cand in candidates:
                unflatten_params(nn_obj, cand)
                fit = evaluate_fitness_multiple_spawns(nn_obj, model, world, spawn_positions, duration)
                fitnesses.append(-fit)

            es.tell(candidates, fitnesses)
            es.disp()

            best_idx = int(np.argmin(fitnesses))
            gen_best_fit = -fitnesses[best_idx]
            gen_best_candidate = candidates[best_idx]

            if gen_best_fit > best_fitness:
                best_fitness = gen_best_fit
                best_theta = gen_best_candidate.copy()

            mean_fit = -np.mean(fitnesses)
            median_fit = -np.median(fitnesses)
            elapsed_time = time.time() - start_time
            writer.writerow([gen, gen_best_fit, mean_fit, median_fit, elapsed_time] + list(gen_best_candidate))

        elapsed_time = time.time() - start_time
        writer.writerow(["final_best", best_fitness, "", "", elapsed_time] + list(best_theta))
        np.save(output_csv.replace(".csv", "_final_best_theta.npy"), best_theta)

    unflatten_params(nn_obj, best_theta)
    return best_theta, best_fitness

# ---------- Experiment ----------
def experiment2():
    core = gecko()
    world = OlympicArena()
    world.spawn(core.spec, spawn_position=[0,0,0.1])
    model = world.spec.compile()

    global AMP, MID, CTRL_RANGE_ARR
    CTRL_RANGE_ARR = model.actuator_ctrlrange.copy()
    AMP = (CTRL_RANGE_ARR[:,1]-CTRL_RANGE_ARR[:,0])*0.5
    MID = (CTRL_RANGE_ARR[:,1]+CTRL_RANGE_ARR[:,0])*0.5

    nn_obj = NNController(nu=model.nu, hidden_size=32)
    best_theta, best_fit = train_nn_cma_multiple_spawns_to_csv(
        nn_obj, model, world, SPAWN_POSITIONS,
        generations=200, pop_size=40, sigma=0.5, duration=10.0,
        output_csv="./dataA3/CMA_results_nn.csv"
    )
    print(f"Best fitness: {best_fit}")

# ---------- Visualization ----------
# ---------- Visualization ----------
def run_results():
    core = gecko()
    world = OlympicArena()
    world.spawn(core.spec, spawn_position=[1, 0, 0.1])
    world.spec.worldbody.add_geom(
        type=mj.mjtGeom.mjGEOM_SPHERE,
        name="target_marker",
        size=[0.05, 0, 0],
        pos=TARGET_POS,
        rgba=[1, 0, 0, 1],
        contype=0,
        conaffinity=0
    )
    model = world.spec.compile()

    global AMP, MID, CTRL_RANGE_ARR
    CTRL_RANGE_ARR = model.actuator_ctrlrange.copy()
    AMP = (CTRL_RANGE_ARR[:, 1] - CTRL_RANGE_ARR[:, 0]) * 0.5
    MID = (CTRL_RANGE_ARR[:, 1] + CTRL_RANGE_ARR[:, 0]) * 0.5

    nn_obj = NNController(nu=model.nu, hidden_size=32)

    # 🔹 Load the final best theta from .npy or CSV
    theta = None
    try:
        theta = np.load("./dataA3/CMA_results_nn_final_best_theta.npy")
        print("Loaded final best parameters from .npy file.")
    except FileNotFoundError:
        print("No .npy file found, checking CSV...")

        # import pandas as pd
        results_path = './dataA3/CMA_results_nn.csv'
        if not os.path.exists(results_path) or os.path.getsize(results_path) == 0:
            print("⚠️ No CMA results file found or it’s empty — using random init.")
            theta = None
        else:
            results = pd.read_csv(results_path)
            if results.empty:
                print("⚠️ CMA results CSV is empty — using random init.")
                theta = None
            else:
                row = results.iloc[-1]
                theta = row.filter(like="theta_").to_numpy(dtype=np.float64)
                print("✅ Loaded final best parameters from CSV.")


    # 🔹 Apply parameters to NN
    if theta is not None:
        unflatten_params(nn_obj, theta)
        print("✅ Applied trained parameters to neural network.")
    else:
        print("⚠️ Using random-initialized network (no trained parameters).")

    # Tracker setup
    tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_BODY, name_to_bind="core")
    data = mj.MjData(model)
    tracker.setup(world.spec, data)

    # Control callback
    ctrl = Controller(
        controller_callback_function=lambda m, d: nn_obj_control(nn_obj, m, d, TARGET_POS),
        tracker=tracker
    )
    mj.set_mjcb_control(ctrl.set_control)

    # Run simulation and visualize
    simple_runner(model, data, duration=50)
    viewer.launch(model=model, data=data)


# ---------- Main ----------
def main():
    #experiment2()  # uncomment to train
    run_results()

if __name__ == "__main__":
    main()
