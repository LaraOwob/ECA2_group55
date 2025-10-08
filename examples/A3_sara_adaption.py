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


from pathlib import Path
from typing import Literal
import os
import csv

# Third-party libraries
#import matplotlib.pyplot as plt
import mujoco as mj
import numpy as np
import torch
import torch.nn as nn
import cma # type: ignore
from mujoco import viewer


# Type Checking
if TYPE_CHECKING:
    from networkx import DiGraph

# Type Aliases
type ViewerTypes = Literal["launcher", "video", "simple", "no_control", "frame"]





# --- CONFIGURATION --- #
SEED = 42
RNG = np.random.default_rng(SEED)
CTRL_RANGE = 1.0

# --- DATA SETUP ---
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True, parents=True)

print(f"[INFO] Data directory: {DATA}")

import os
import csv
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import mujoco as mj
import matplotlib.pyplot as plt
from functools import partial
from typing import Any

# -------------------------
# SYSTEM SETUP
# -------------------------
SEED = 42
RNG = np.random.default_rng(SEED)
TARGET_POS = np.array([4.5, 0.0, 0.55])
SPAWN_POSITIONS = [np.array([2.5, 0.0, 0.1]), np.array([0.5, 0.0, 0.1])]

SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True)


# -------------------------
# NN CONTROLLER
# -------------------------
class NNController(nn.Module):
    def __init__(self, nu: int, hidden_size: int = 32):
        super().__init__()
        self.nu = nu
        self.in_dim = nu + nu + 2 + 2  # qpos + qvel + sin/cos + delta_x/y
        self.hidden = hidden_size
        self.fc1 = nn.Linear(self.in_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, nu)
        self.tanh = nn.Tanh()

    def forward(self, qpos: torch.Tensor, qvel: torch.Tensor, t: float,
                target_pos: torch.Tensor, current_pos: torch.Tensor):
        t_tensor = torch.tensor(t, dtype=torch.float32)
        delta = target_pos[:2] - current_pos[:2]
        time_features = torch.stack([torch.sin(2 * np.pi * t_tensor), torch.cos(2 * np.pi * t_tensor)])
        obs = torch.cat([qpos, qvel, time_features, delta], dim=0)
        h = self.tanh(self.fc1(obs))
        h = self.tanh(self.fc2(h))
        return self.tanh(self.fc3(h))


# -------------------------
# UTILITY FUNCTIONS
# -------------------------
def flatten_params(nn: nn.Module) -> np.ndarray:
    return np.concatenate([p.detach().cpu().numpy().ravel() for p in nn.parameters()])


def unflatten_params(nn: nn.Module, flat: np.ndarray) -> None:
    offset = 0
    with torch.no_grad():
        for p in nn.parameters():
            numel = p.numel()
            p.copy_(torch.tensor(flat[offset:offset+numel].reshape(p.shape), dtype=p.dtype))
            offset += numel


def show_xpos_history(history: list[float]) -> None:
    pos_data = np.array(history)
    plt.figure(figsize=(10, 6))
    plt.plot(pos_data[:, 0], pos_data[:, 1], "b-", label="Path")
    plt.plot(pos_data[0, 0], pos_data[0, 1], "go", label="Start")
    plt.plot(pos_data[-1, 0], pos_data[-1, 1], "ro", label="End")
    plt.plot(0, 0, "kx", label="Origin")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Robot Path in XY Plane")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()


def nn_obj_control(nn_obj: NNController, model: mj.MjModel, data: mj.MjData, target_pos: np.ndarray):
    current_pos = torch.tensor(data.geom_xpos[0], dtype=torch.float32)
    qpos = torch.tensor(data.qpos[:nn_obj.nu], dtype=torch.float32)
    qvel = torch.tensor(data.qvel[:nn_obj.nu], dtype=torch.float32)
    action = nn_obj(qpos=qpos, qvel=qvel, t=data.time,
                    target_pos=torch.tensor(target_pos, dtype=torch.float32),
                    current_pos=current_pos)
    return action.detach().numpy()


def evaluateFitness_SARA(model: mj.MjModel, world: Any, nn_obj: NNController,
                         target_pos: np.ndarray = TARGET_POS, duration: float = 8.0) -> float:
    mj.set_mjcb_control(None)
    data = mj.MjData(model)
    mj.mj_resetData(model, data)

    bind_name = "core"
    tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind=bind_name)
    tracker.setup(world.spec, data)

    def _control_cb(m, d):
        return nn_obj_control(nn_obj, m, d, target_pos)

    ctrl = Controller(controller_callback_function=_control_cb, tracker=tracker)
    mj.set_mjcb_control(ctrl.set_control)

    try:
        simple_runner(model, data, duration=duration)
    except Exception as e:
        print(f"[WARNING] Simulation failed: {e}")
        return -1e6

    xpos_dict = tracker.history.get("xpos", {})
    if not xpos_dict:
        return -1e6

    first_key = next(iter(xpos_dict))
    traj = np.asarray(xpos_dict[first_key])
    if traj.size == 0 or len(traj) < 2:
        return -1e6

    start, end = traj[0], traj[-1]
    progress_x = float(end[0] - start[0])
    dist_to_target = float(np.linalg.norm(end[:2] - target_pos[:2]))
    proximity_bonus = max(0.0, 3.0 - dist_to_target)
    fitness = progress_x + proximity_bonus
    return fitness


def evaluate_fitness_multiple_spawns(nn_obj: NNController, model: mj.MjModel, world: Any,
                                     spawn_positions: list[np.ndarray], duration: float = 10.0) -> float:
    fitnesses = []
    spawn_positions = [np.array([2.5, 0.0, 0.1]), np.array([0.5, 0.0, 0.1])]
    for pos in spawn_positions:
        data = mj.MjData(model)
        mj.mj_resetData(model, data)
        data.qpos[:len(pos)] = pos  # safe broadcast
        data.qvel[:] = 0
        mj.mj_step(model, data, nstep=10)
        fitness = evaluateFitness_SARA(model, world, nn_obj, target_pos=TARGET_POS, duration=duration)
        fitnesses.append(fitness)
    return float(np.mean(fitnesses))


# -------------------------
# CMA TRAINING
# -------------------------
def train_nn_cma_multiple_spawns_to_csv(nn_obj, model, world, spawn_positions: list[np.ndarray],
                                        generations=50, pop_size=16, sigma=0.5, duration=10.0,
                                        output_csv="./results.csv", seed=42):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    x0 = flatten_params(nn_obj)
    import cma
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


# -------------------------
# MAIN EXPERIMENT
# -------------------------
def experiment2():
    core = gecko()
    world = OlympicArena()
    world.spawn(core.spec, spawn_position=[0, 0, 0.1])

    model = world.spec.compile()
    nu = model.nu
    nn_obj = NNController(nu=nu, hidden_size=32)

    best_theta, best_fit = train_nn_cma_multiple_spawns_to_csv(
        nn_obj,
        model,
        world,
        SPAWN_POSITIONS,
        generations=200,
        pop_size=40,
        sigma=0.5,
        duration=10.0,
        output_csv="./dataA3/CMA_results_nn.csv"
    )
    print(f"Best fitness: {best_fit}")


def run_results():
    # Create the world and robot
    core = gecko()
    world = OlympicArena()
    
    # Spawn the robot
    world.spawn(core.spec, spawn_position=[0.5, 0, 0.1])

    # Add target marker
    world.spec.worldbody.add_geom(
        type=mj.mjtGeom.mjGEOM_SPHERE,
        name="target_marker",
        size=[0.05, 0, 0],
        pos=TARGET_POS,
        rgba=[1, 0, 0, 1],
        contype=0,
        conaffinity=0
    )

    # 🔑 Compile after all modifications
    model = world.spec.compile()
    data = mj.MjData(model)

    # Initialize NNController
    nn_obj = NNController(nu=model.nu, hidden_size=32)

    # Load previously optimized parameters if available
    try:
        theta = np.load("./dataA3/CMA_results_nn_final_best_theta.npy")
        unflatten_params(nn_obj, theta)
    except FileNotFoundError:
        print("No saved parameters found, using random init.")

    # Setup tracker
    tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")
    tracker.setup(world.spec, data)  # must use fresh data matching compiled model

    # Set controller callback
    def _control_cb(m, d):
        return nn_obj_control(nn_obj, m, d, TARGET_POS)

    ctrl = Controller(controller_callback_function=_control_cb, tracker=tracker)
    mj.set_mjcb_control(ctrl.set_control)

    # Run simulation
    simple_runner(model, data, duration=50)
    viewer.launch(model=model, data=data)

    # Show XY trajectory
    if "xpos" in tracker.history:
        show_xpos_history(tracker.history["xpos"][0])



def main():
    #experiment2()
    run_results()


if __name__ == "__main__":
    main()
