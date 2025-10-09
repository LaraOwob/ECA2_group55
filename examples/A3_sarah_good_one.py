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

def _bind_geom_name(model: mj.MjModel) -> str:
    try:
        _ = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "core")
        return "core"
    except Exception:
        return mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, 0)


def nn_control_step(nn_obj, m, d, target_pos):
    # use body (root) pos for current_pos (as you set up)
    current_pos = torch.tensor(d.xpos[ROBOT_BODY_ID], dtype=torch.float32)

    qpos = torch.tensor(d.qpos[:nn_obj.nq], dtype=torch.float32)  # nq
    qvel = torch.tensor(d.qvel[:nn_obj.nv], dtype=torch.float32)  # nv

    raw = nn_obj(qpos, qvel, d.time, torch.tensor(target_pos), current_pos).detach().numpy()
    # raw has length nu; scale to actuator range:
    a = MID + (AMP * ACT_GAIN) * np.clip(raw, -1.0, 1.0)
    return np.clip(a, CTRL_RANGE_ARR[:, 0], CTRL_RANGE_ARR[:, 1])

# ---------- Fitness ----------
'''def evaluate(model: mj.MjModel,
             world: OlympicArena,
             nn_obj: NNController,
             target_pos: np.ndarray = TARGET_POS,
             duration: float = DURATION) -> float:
    mj.set_mjcb_control(None)  # important to clear old callbacks
    data = mj.MjData(model)
    mj.mj_resetData(model, data)

    bind = _bind_geom_name(model)
    tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind=bind)
    tracker.setup(world.spec, data)

    def _cb(m, d):  # type: ignore # controller callback
        return nn_control_step(nn_obj, m, d, target_pos)
    ctrl = Controller(controller_callback_function=_cb, tracker=tracker)
    mj.set_mjcb_control(ctrl.set_control)

    try:
        simple_runner(model, data, duration=duration)
    except Exception as e:
        print(f"[WARN] Simulation failed: {e}")
        return -1e6

    path = tracker.history.get("xpos", {})
    if not path:
        return -1e6
    key0 = next(iter(path))
    traj = np.asarray(path[key0])
    if traj.size == 0:
        return -1e6

    start, end = traj[0], traj[-1]
    progress_x = float(end[0] - start[0])            # forward progress
    dist_to_target = float(np.linalg.norm(end[:2] - target_pos[:2]))
    proximity_bonus = max(0.0, 3.0 - dist_to_target) # simple shaping
    return progress_x + proximity_bonus'''

def rot_to_yaw(xmat9: np.ndarray) -> float:
    # xmat9 is a 3x3 rotation matrix in row-major (flattened length 9)
    R = np.array([[xmat9[0], xmat9[1], xmat9[2]],
                  [xmat9[3], xmat9[4], xmat9[5]],
                  [xmat9[6], xmat9[7], xmat9[8]]])
    # yaw from ZYX euler: yaw = atan2(R[1,0], R[0,0])
    return float(np.arctan2(R[1,0], R[0,0]))

def evaluateFitness(model: mj.MjModel,
                    world: OlympicArena,
                    nn_obj: NNController,
                    target_pos: np.ndarray = np.array(TARGET_POS, dtype=np.float32),
                    duration: float = 8.0,
                    # ----- arena section boundaries -----
                    x_flat_end: float = 0.5,     # flat:   [-2.5, 0.5]
                    x_rugged_end: float = 2.5,   # rugged: [0.5, 2.5]
                    x_finish: float = 5.42,      # finish stripe
                    # ----- weights & tolerances -----
                    w_flat: float = 1.00,
                    w_rugged: float = 1.60,      # a tad higher to value rough progress
                    w_uphill: float = 1.80,
                    w_finish: float = 10.0,
                    w_speed: float = 4.0,
                    w_climb: float = 0.5,
                    y_tol: float = 0.6,
                    w_lat: float = 0.5,
                    fall_pitch_roll_deg: float = 60.0) -> float:
    """
    Reward grounded forward progress (harder sections worth more), finishing fast,
    slight preference for facing +x; penalize sideways drift, sliding, moving backward,
    tipping, going below floor, and leaving arena bounds.
    """
    # --- reset & install control callback ---
    mj.set_mjcb_control(None)
    data = mj.MjData(model)
    mj.mj_resetData(model, data)

    tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_BODY,
                      name_to_bind=ROBOT_BODY_NAME)
    tracker.setup(world.spec, data)

    def _cb(m, d):  # controller callback
        return nn_control_step(nn_obj, m, d, target_pos)
    ctrl = Controller(controller_callback_function=_cb, tracker=tracker)
    mj.set_mjcb_control(ctrl.set_control)

    # --- simulate ---
    try:
        simple_runner(model, data, duration=duration)
    except Exception as e:
        print(f"[WARN] Simulation failed: {e}")
        return -1e6

    # --- trajectory from tracker ---
    hist = tracker.history.get("xpos", {})
    if not hist:
        return -1e6
    key = next(iter(hist))
    traj = np.asarray(hist[key])  # [T,3]
    if traj.shape[0] < 2:
        return -1e6

    xs, ys, zs = traj[:, 0], traj[:, 1], traj[:, 2]
    x0, y0, z0 = traj[0]
    xT, yT, zT = traj[-1]

    # --- floor breach penalty (fell underground) ---
    fell_below_floor = bool(np.any(zs < 0.0))
    floor_pen = 20.0 if fell_below_floor else 0.0

    # --- section-weighted grounded forward progress ---
    dx = np.diff(xs)
    x_mid = 0.5 * (xs[1:] + xs[:-1])

    # only credit progress when near ground (reduce "flying" exploits)
    near_ground = (zs[:-1] <= z0 + 0.15)   # slightly tighter gate
    credited_dx = np.clip(dx, 0.0, None) * near_ground.astype(float)

    section_w = np.where(
        x_mid < x_flat_end, w_flat,
        np.where(x_mid < x_rugged_end, w_rugged, w_uphill)
    )
    progress = float(np.sum(credited_dx * section_w))

    # --- finish + speed bonus (must be grounded at end) ---
    finished = (xT >= x_finish) and (zT <= z0 + 0.20)
    finish_bonus = w_finish if finished else 0.0
    speed_bonus = 0.0
    if finished:
        idx_finish = int(np.argmax(xs >= x_finish))
        t_finish = (idx_finish / max(1, (len(xs) - 1))) * duration
        speed_bonus = w_speed * max(0.0, (duration - t_finish) / duration)

    # --- climb bonus (small) ---
    climb_bonus = w_climb * max(0.0, (zT - z0))

    # --- lateral deviation (soft tolerance) ---
    y_mean_abs = float(np.mean(np.abs(ys)))
    lateral_pen = w_lat * max(0.0, y_mean_abs - y_tol)

    # --- lateral velocity penalty (discourage persistent sliding) ---
    dt = duration / max(1, len(xs) - 1)
    vy = np.diff(ys) / dt if len(ys) > 1 else np.array([0.0])
    w_latvel = 0.3
    latvel_pen = w_latvel * float(np.mean(np.clip(np.abs(vy) - 0.05, 0.0, None)))

    # --- backward motion penalty ---
    neg_dx = np.clip(-np.diff(xs), 0.0, None)
    w_back = 0.5
    back_pen = w_back * float(np.sum(neg_dx))

    # --- out-of-bounds penalty (falling off sides) ---
    arena_half_width = getattr(world, "arena_width", 2.0) / 2.0
    margin = 0.05
    oob = bool(np.any(np.abs(ys) > (arena_half_width - margin)))
    oob_pen = 10.0 if oob else 0.0

    # --- fall penalty via BODY orientation (roll/pitch) ---
    xmat9 = data.xmat[ROBOT_BODY_ID].copy()  # 3x3 rot matrix (flat 9)
    R = np.array([[xmat9[0], xmat9[1], xmat9[2]],
                  [xmat9[3], xmat9[4], xmat9[5]],
                  [xmat9[6], xmat9[7], xmat9[8]]])
    pitch = np.arcsin(-R[2, 0])
    roll  = np.arctan2(R[2, 1], R[2, 2])
    roll_deg, pitch_deg = np.degrees(roll), np.degrees(pitch)
    fall = (abs(roll_deg) > fall_pitch_roll_deg) or (abs(pitch_deg) > fall_pitch_roll_deg)
    fall_pen = 5.0 if fall else 0.0

    # --- heading alignment (tiny nudge to face +x) ---
    w_yaw_align = 0.05
    yaw = rot_to_yaw(xmat9)
    yaw_align = w_yaw_align * np.cos(yaw)

    # --- final score ---
    fitness = (
        progress + finish_bonus + speed_bonus + climb_bonus + yaw_align
        - lateral_pen - latvel_pen - back_pen - fall_pen - floor_pen - oob_pen
    )
    return float(fitness)


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
