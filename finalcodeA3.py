#from pyswarm import pso
from scipy.optimize import differential_evolution
import numpy as np
import time


# Standard library
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

# Third-party libraries
import matplotlib.pyplot as plt
import mujoco as mj
import numpy as np
import numpy.typing as npt
from mujoco import viewer

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
# from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.utils.runners import simple_runner
from ariel.utils.tracker import Tracker
from ariel.utils.video_recorder import VideoRecorder

from ariel.simulation.tasks.targeted_locomotion import distance_to_target

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import mujoco as mj
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import HighProbabilityDecoder
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.environments import OlympicArena
from ariel.utils.runners import simple_runner
from ariel.utils.tracker import Tracker
#from stable_baselines3 import SAC
import os
import cma
from scipy.optimize import differential_evolution
import csv
import numpy as np
import csv
import os
import time
import cma
#from pyswarm import pso
import glob

import time
import matplotlib.pyplot as plt
import pandas as pd
import os

######################################

from pathlib import Path
from typing import Literal, Optional, Any
import os
import csv

import numpy as np
import torch
import torch.nn as nn
import mujoco as mj
from mujoco import viewer
import cma  # type: ignore

# ARIEL imports
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.simulation.environments import OlympicArena
from ariel.simulation.controllers.controller import Controller
from ariel.utils.runners import simple_runner
from ariel.utils.tracker import Tracker
from ariel.utils.video_recorder import VideoRecorder
from ariel.utils.renderers import video_renderer, single_frame_renderer
from examples.z_ec_course.A3_plot_function import show_xpos_history

# ---------- Config ----------


# Type Checking
if TYPE_CHECKING:
    from networkx import DiGraph
#type ViewerTypes = Literal["launcher", "video", "simple", "no_control", "frame","timestep"]


WORKING_BODIES =set()


SEED = 42
RNG = np.random.default_rng(SEED)
CTRL_RANGE = 1.0

NUM_OF_MODULES = 30
TARGET_POS = [5, 0, 0.5]
SPAWN_POS = [-0.8, 0, 0]
DURATION = 10.0
################
#################
#############
#IMPORTANT to change!!!!!!!!!!!!!!!!!!!!

GLOBAL_SEED = 55














# --- scenarios you want to average over ---
SCENARIOS = [
    ("rugged", np.array([0.5, 0.0, 0.04], dtype=np.float32), 15.0),  # start at rugged (lower spawn)
    ("uphill", np.array([2.5, 0.0, 0.04], dtype=np.float32), 20.0),  # start at steep (lower spawn)
]

ACTION_LOG = []


# --- DATA SETUP ---
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True)


# ---------- NN controller ----------
class NNController(nn.Module):
    def __init__(self, nq: int, nv: int, nu: int, hidden: int = 24, freq_hz: float = 1.5):
        super().__init__()
        self.nq, self.nv, self.nu = nq, nv, nu
        self.in_dim = nq + nv + 2 + 2  # correct
        self.fc1 = nn.Linear(self.in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, nu)  # output matches number of actuators
        self.act = nn.Tanh()
        self.freq_hz = freq_hz
        self.u_prev = None

    def forward(self, qpos, qvel, t, target_pos, current_pos):
        w = 2 * torch.pi * self.freq_hz
        tt = torch.tensor(t, dtype=torch.float32)
        time_feats = torch.stack([torch.sin(w * tt), torch.cos(w * tt)])
        delta = target_pos[:2] - current_pos[:2]
        obs = torch.cat([qpos, qvel, time_feats, delta], dim=0)
        x = self.act(self.fc1(obs))
        x = self.act(self.fc2(x))
        y = self.act(self.fc3(x))
        return y  # in [-1,1]


# ---------- Helper utils ----------
def flatten_params(net: nn.Module) -> np.ndarray:
    return np.concatenate([p.detach().cpu().numpy().ravel() for p in net.parameters()])

def flatten_size(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())

def unflatten_params(net: nn.Module, flat: np.ndarray) -> None:
    need = flatten_size(net)
    if flat.size != need:
        raise ValueError(f"Param size mismatch: file has {flat.size}, network expects {need}.")
    i = 0
    with torch.no_grad():
        for p in net.parameters():
            n = p.numel()
            p.copy_(torch.tensor(flat[i:i+n].reshape(p.shape), dtype=p.dtype))
            i += n

def _bind_geom_name(model: mj.MjModel) -> str:
    """Return a valid geometry name to bind the Tracker to.
    Prefer a geom named 'core' if present; otherwise fall back to the first geom name.
    """
    try:
        _ = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "core")
        return "core"
    except Exception:
        return mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, 0)


def prepare_scenarios(robot_spec):
    worlds, models, durations = [], [], []
    for name, spos, dur in SCENARIOS:
        w, m = build_world_and_model_at(robot_spec, spos)
        worlds.append(w); models.append(m); durations.append(dur)

    global CTRL_RANGE_ARR, AMP, MID, ACT_GAIN, ROBOT_BODY_ID, ROBOT_BODY_NAME
    CTRL_RANGE_ARR = models[0].actuator_ctrlrange.copy()
    AMP = (CTRL_RANGE_ARR[:,1] - CTRL_RANGE_ARR[:,0]) * 0.5
    MID = (CTRL_RANGE_ARR[:,1] + CTRL_RANGE_ARR[:,0]) * 0.5
    ACT_GAIN = 1.0
    ROBOT_BODY_ID, ROBOT_BODY_NAME = find_robot_root_body(models[0])
    return worlds, models, durations

def mean_fitness_over_scenarios(nn_obj, worlds, models, durations):
    vals = []
    for w, m, dur in zip(worlds, models, durations):
        f = evaluateFitness(m, w, nn_obj, target_pos=TARGET_POS, duration=dur)
        vals.append(f)
    return float(np.mean(vals))

def nn_control_step(nn_obj, m, d, target_pos):
    global ACTION_LOG
    # use body (root) pos for current_pos
    current_pos = torch.tensor(d.xpos[ROBOT_BODY_ID], dtype=torch.float32)

    qpos = torch.tensor(d.qpos[:nn_obj.nq], dtype=torch.float32)  # nq
    qvel = torch.tensor(d.qvel[:nn_obj.nv], dtype=torch.float32)  # nv

    raw = nn_obj(qpos, qvel, d.time, torch.tensor(target_pos), current_pos).detach().numpy()

    if nn_obj.u_prev is None: nn_obj.u_prev = raw

    raw = 0.8*nn_obj.u_prev + 0.2*raw   # EMA smoothing
    nn_obj.u_prev = raw

    # return normalized action ONLY in [-1, 1]; Controller will scale to actuator ranges
    u = np.clip(raw, -1.0, 1.0)

    if len(ACTION_LOG) == 0:
        print(f"[CTRLDBG] first cmd norm≈{np.linalg.norm(u):.3f}")

    ACTION_LOG.append(u.copy())

    if len(ACTION_LOG) > 2048:
        ACTION_LOG[:] = ACTION_LOG[-1024:]
    return u

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

def morph_sig(model: mj.MjModel) -> str:
    return f"nq{model.nq}_nv{model.nv}_nu{model.nu}"

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
                    duration: float = 10.0,
                    x_flat_end: float = 0.5,
                    x_rugged_end: float = 2.5,
                    x_finish: float = 5.42,
                    w_flat: float = 1.00,
                    w_rugged: float = 1.60,
                    w_uphill: float = 2.50,
                    w_finish: float = 10.0,
                    w_speed: float = 4.0,
                    w_climb: float = 0.5,
                    y_tol: float = 0.6,
                    w_lat: float = 0.5,
                    fall_pitch_roll_deg: float = 60.0) -> float:

    # --- reset & install control callback ---
    mj.set_mjcb_control(None)
    data = mj.MjData(model)
    mj.mj_resetData(model, data)

    tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")
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

    # --- trajectory ---
    hist = tracker.history.get("xpos", {})
    if not hist:
        return -1e6
    key = next(iter(hist))
    traj = np.asarray(hist[key])  # [T, 3]
    if traj.shape[0] < 2:
        return -1e6

    xs, ys, zs = traj[:, 0], traj[:, 1], traj[:, 2]
    x0, y0, z0 = traj[0]
    xT, yT, zT = traj[-1]

    # time diffs
    T = len(xs)
    dt = duration / max(1, (T - 1))

    # per-step mids & masks
    dx = np.diff(xs)
    dy = np.diff(ys)
    dz = np.diff(zs)
    x_mid = 0.5 * (xs[1:] + xs[:-1])
    on_slope = (x_mid >= x_rugged_end)

    # --- floor breach penalty ---
    #floor_pen = 20.0 if np.any(zs < 0.0) else 0.0
    # be tolerant to small numerical dips, only penalize if torso sinks clearly below terrain
    z_floor_tol = -0.12   # allow slight negative due to contact/mesh noise
    floor_pen = 15.0 if np.any(zs < z_floor_tol) else 0.0

    # --- ground gating (looser on slope) ---
    gate = np.where(on_slope, z0 + 0.40, z0 + 0.15)  # allow torso rise on slope
    near_ground = (zs[:-1] <= gate)

    # --- section weights for progress ---
    section_w = np.where(
        x_mid < x_flat_end, w_flat,
        np.where(x_mid < x_rugged_end, w_rugged, w_uphill)
    )

    # progress only when forward + near ground
    credited_dx = np.clip(dx, 0.0, None) * near_ground.astype(float)
    progress = float(np.sum(credited_dx * section_w))

    # --- extra rewards to help climbing ---
    vx = dx / dt
    vz = dz / dt

    # forward velocity reward (higher on slope)
    w_vx_flat, w_vx_slope = 0.3, 0.8
    vx_rew = (
        w_vx_flat  * float(np.sum(np.clip(vx[~on_slope], 0.0, None))) +
        w_vx_slope * float(np.sum(np.clip(vx[ on_slope], 0.0, None)))
    )

    # upward velocity on slope
    w_dz_slope = 1.0
    dz_rew = w_dz_slope * float(np.sum(np.clip(vz[on_slope], 0.0, None)))

    # --- finish + speed bonus (must be grounded at end) ---
    finished = (xT >= x_finish) and (zT <= z0 + 0.20)
    if finished:
        idx_finish = int(np.argmax(xs >= x_finish))
        t_finish = (idx_finish / max(1, (T - 1))) * duration
        finish_bonus = w_finish
        speed_bonus = w_speed * max(0.0, (duration - t_finish) / duration)
    else:
        finish_bonus = 0.0
        speed_bonus = 0.0

    # --- climb bonus (terminal height) ---
    climb_bonus = w_climb * max(0.0, (zT - z0))

    # --- lateral penalties (relaxed on slope) ---
    lat_w_step = np.where(on_slope, 0.3 * w_lat, w_lat)
    latvel_w_step = np.where(on_slope, 0.2, 0.3)

    lat_step = np.abs(ys[1:]) - y_tol
    lateral_pen = float(np.sum(np.clip(lat_step, 0.0, None) * lat_w_step))

    vy = dy / dt
    latvel_pen = float(np.sum(np.clip(np.abs(vy) - 0.05, 0.0, None) * latvel_w_step))

    # --- backward motion penalty (smaller on slope) ---
    w_back_flat, w_back_slope = 0.3, 0.10
    neg_dx = np.clip(-dx, 0.0, None)
    back_pen = (
        w_back_flat  * float(np.sum(neg_dx[~on_slope])) +
        w_back_slope * float(np.sum(neg_dx[ on_slope]))
    )

    # --- out-of-bounds penalty ---
    arena_half_width = getattr(world, "arena_width", 2.0) / 2.0
    oob_pen = 10.0 if np.any(np.abs(ys) > (arena_half_width - 0.05)) else 0.0

    # --- fall penalty via BODY orientation ---
    xmat9 = data.xmat[ROBOT_BODY_ID].copy()
    R = np.array([[xmat9[0], xmat9[1], xmat9[2]],
                  [xmat9[3], xmat9[4], xmat9[5]],
                  [xmat9[6], xmat9[7], xmat9[8]]])
    pitch = np.arcsin(-R[2, 0])
    roll  = np.arctan2(R[2, 1], R[2, 2])
    fall = (abs(np.degrees(roll))  > fall_pitch_roll_deg or
            abs(np.degrees(pitch)) > fall_pitch_roll_deg)
    fall_pen = 5.0 if fall else 0.0

    # --- tiny heading nudge ---
    try:
        yaw_align = 0.01 * np.cos(rot_to_yaw(xmat9))*np.clip(dx.sum(),0,None)
    except Exception:
        yaw_align = 0.0

    # --- symmetry penalty (encourage L/R parity) ---
    sym_pairs = [(i, i+1) for i in range(0, model.nu - 1, 2)]  # (0,1), (2,3), ...
    if ACTION_LOG and sym_pairs:
        A = np.asarray(ACTION_LOG)  # [T, nu]
        mu = np.mean(np.abs(A), axis=0)
        sym_pen = 0.1 * sum((mu[i] - mu[j])**2 for (i, j) in sym_pairs)
    else:
        sym_pen = 0.0
    ACTION_LOG.clear()

    # --- final score ---
    fitness = (
        progress + vx_rew + dz_rew + finish_bonus + speed_bonus + climb_bonus + yaw_align
        - lateral_pen - latvel_pen - back_pen - fall_pen - floor_pen - oob_pen - sym_pen
    )

    # DEBUG: print term breakdown (first few steps only)
    print(f"[FITDBG] prog={progress:.3f} vx={vx_rew:.3f} dz={dz_rew:.3f} fin={finish_bonus:.3f} spd={speed_bonus:.3f} "
          f"clmb={climb_bonus:.3f} yaw={yaw_align:.3f} | "
          f"lat={lateral_pen:.3f} lvel={latvel_pen:.3f} back={back_pen:.3f} fall={fall_pen:.3f} "
          f"floor={floor_pen:.3f} oob={oob_pen:.3f} sym={sym_pen:.3f} => fit={fitness:.3f}")

    return float(fitness)


# ---------- Build world+model once for gecko ----------
def build_world_and_model_at(robot_spec, spawn_pos):
    mj.set_mjcb_control(None)
    world = OlympicArena( load_precompiled=False,)
    print("print this is spwan",robot_spec)
    world.spawn(robot_spec.spec, position=spawn_pos.tolist(),correct_collision_with_floor=True)
    model = world.spec.compile() #<mujoco._specs.MjSpec object at 0x000001C5019AE370>

    # --- diagnostic: print actuator names and ranges ---
    print("\n[Actuator ranges]")
    for i in range(model.nu):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, i)
        lo, hi = model.actuator_ctrlrange[i]
        print(f"{i:02d} {name:20s} [{lo:.2f}, {hi:.2f}]")
    print("-------------------------------------------------\n")

    return world, model


def find_robot_root_body(model: mj.MjModel) -> tuple[int, str]:
    # pick the first body whose name starts with "robot-"
    for bid in range(model.nbody):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, bid)
        if name and name.startswith("robot-"):
            return bid, name
    # fallback to the world child (usually 1)
    return 1, mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, 1)


# ---------- Train (CMA-ES over NN weights) ----------
'''def train_controller(out_csv: Path, # type: ignore
                     duration: float = DURATION,
                     iterations: int = 12,
                     popsize: int = 16,
                     hidden: int = 32):
    world, model = build_world_and_model()
    nn_obj = NNController(nq=model.nq, nv=model.nv, nu=model.nu, hidden=32, freq_hz=3.0)

    x0 = flatten_params(nn_obj)
    def obj(x):
        unflatten_params(nn_obj, x)
        f = evaluateFitness(model, world, nn_obj,
                        target_pos=TARGET_POS,
                        duration=duration)
        return -f  # CMA-ES minimizes

    es = cma.CMAEvolutionStrategy(x0, 0.8,
                                  {'popsize': popsize, 'maxiter': iterations,
                                   'verb_disp': 1, 'verb_log': 0})
    #es.optimize(obj)
    history = []

    while not es.stop():
        xs = es.ask()
        fs = []
        for x in xs:
            unflatten_params(nn_obj, x)
            f = evaluateFitness(model, world, nn_obj, target_pos=TARGET_POS, duration=duration)
            fs.append(-f)  # minimize
        es.tell(xs, fs)
        best_gen = -float(np.min(fs))
        history.append(best_gen)

    es.result_pretty()
    best_x = es.result.xbest
    best_f = -es.result.fbest
    unflatten_params(nn_obj, best_x)

    # save curve
    np.savetxt(DATA / "fitness_curve.csv", np.array(history), delimiter=",")

    best_x = es.result.xbest
    best_f = -es.result.fbest
    unflatten_params(nn_obj, best_x)

    # save artifacts
    DATA.mkdir(parents=True, exist_ok=True)
    np.save(DATA / "best_controller_params.npy", best_x)
    with open(out_csv, "w", newline="") as f:
        csv.writer(f).writerow(["best_fitness"]); csv.writer(f).writerow([best_f])

    # optional: a single frame
    data = mj.MjData(model)
    single_frame_renderer(model, data, save=True, save_path=str(DATA / "gecko_frame.png"))
    print(f"[DONE] Best fitness: {best_f:.3f}. Saved params & frame into {DATA}")
    return best_f, nn_obj'''

'''def train_controller(out_csv: Path, duration: float = DURATION,
                     iterations: int = 12, popsize: int = 16, hidden: int = 32):
    # build scenarios once
    worlds, models, durations = prepare_scenarios()

    # controller sized to the (identical) gecko model
    nn_obj = NNController(nq=models[0].nq, nv=models[0].nv, nu=models[0].nu,
                          hidden=hidden, freq_hz=3.0)

    x0 = flatten_params(nn_obj)
    es = cma.CMAEvolutionStrategy(x0, 0.8, {
        'popsize': popsize, 'maxiter': iterations, 'verb_disp': 1, 'verb_log': 0
    })

    history = []
    while not es.stop():
        xs = es.ask()
        fs = []
        for x in xs:
            unflatten_params(nn_obj, x)
            f_mean = mean_fitness_over_scenarios(nn_obj, worlds, models, durations)
            fs.append(-f_mean)  # CMA-ES minimizes
        es.tell(xs, fs)
        history.append(-float(np.min(fs)))

    es.result_pretty()
    best_x = es.result.xbest
    best_f = -es.result.fbest
    unflatten_params(nn_obj, best_x)

    # save
    DATA.mkdir(parents=True, exist_ok=True)
    np.savetxt(DATA / "fitness_curve.csv", np.array(history), delimiter=",")
    np.save(DATA / "best_controller_params.npy", best_x)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["best_fitness"]); w.writerow([best_f])
    print(f"[DONE] Best avg fitness over {len(SCENARIOS)} scenarios: {best_f:.3f}")
    return best_f, nn_obj'''

def train_controller(out_csv: Path,
                     robot_spec,  # <- pass the candidate body spec
                     duration: float = DURATION,
                     iterations: int = 12,
                     popsize: int = 16,
                     hidden: int = 24,
                     resume: bool = False,
                     init_sigma: float = 0.6):

    # Build scenarios/worlds/models for THIS body
    worlds, models, durations = prepare_scenarios(robot_spec)
    nn_obj = NNController(nq=models[0].nq, nv=models[0].nv, nu=models[0].nu,
                          hidden=hidden, freq_hz=3.0)

    DATA.mkdir(parents=True, exist_ok=True)
    sig = morph_sig(models[0])
    params_path = DATA / f"{GLOBAL_SEED}_best_controller_params_{sig}.npy"
    curve_path  = DATA / f"{GLOBAL_SEED}_fitness_curve_{sig}.csv"

    # Warm start for THIS morphology only
    x0 = flatten_params(nn_obj)
    if resume and params_path.exists():
        try:
            x0_loaded = np.load(params_path)
            unflatten_params(nn_obj, x0_loaded)
            x0 = x0_loaded
            print(f"[INFO] Warm-starting from {params_path.name}")
        except Exception as e:
            print(f"[WARN] Could not warm-start ({e}). Starting fresh.")

    # CMA-ES
    es = cma.CMAEvolutionStrategy(x0, init_sigma, {
        'popsize': popsize, 'maxiter': iterations, 'verb_disp': 1, 'verb_log': 0
    })

    history = []
    try:
        while not es.stop():
            xs = es.ask()
            fs = []
            for x in xs:
                unflatten_params(nn_obj, x)
                f_mean = mean_fitness_over_scenarios(nn_obj, worlds, models, durations)
                fs.append(-f_mean)  # minimize
            es.tell(xs, fs)
            best_iter = -float(np.min(fs))
            print(f"[INFO] Iter {es.countiter}/{iterations} best fitness: {best_iter:.3f}")
            history.append(best_iter)
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Stopping early—saving best-so-far.")
    finally:
        if es.countevals and es.countevals > 0:
            es.result_pretty()
            best_x = es.result.xbest
            best_f = -es.result.fbest
            unflatten_params(nn_obj, best_x)
        else:
            print("[WARN] CMA-ES ended without successful evaluations.")
            best_x = flatten_params(nn_obj)  # fallback to current
            best_f = -1e6  # or another sentinel

        # Save ONLY morphology-keyed artifacts (avoid generic file confusion)
        np.savetxt(curve_path, np.array(history), delimiter=",")
        np.save(params_path, best_x)

        # Optional: CSV summary you’re already writing
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["best_fitness"])
            w.writerow([best_f])

        print(f"[DONE] Best avg fitness over {len(SCENARIOS)} scenarios: {best_f:.3f} "
              f"({'resumed' if resume and params_path.exists() else 'fresh'})")

    return nn_obj, best_f


# ---------- Visualize ----------
'''def visualize(mode: ViewerTypes = "launcher", duration: float = DURATION):
    world, model = build_world_and_model()
    nn_obj = NNController(nq=model.nq, nv=model.nv, nu=model.nu, hidden=32, freq_hz=3.0)
    params_path = DATA / "best_controller_params.npy"
    if not params_path.exists():
        print(f"[ERR] Missing {params_path}. Run with mode=train first.")
        return
    unflatten_params(nn_obj, np.load(params_path))

    def _cb(m, d): return nn_control_step(nn_obj, m, d, TARGET_POS)
    bind = _bind_geom_name(model)
    tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_BODY,
                  name_to_bind=ROBOT_BODY_NAME)
    data = mj.MjData(model); tracker.setup(world.spec, data)
    ctrl = Controller(controller_callback_function=_cb, tracker=tracker)
    mj.set_mjcb_control(ctrl.set_control)

    if mode == "launcher":
        viewer.launch(model=model, data=data)
    elif mode == "video":
        rec = VideoRecorder(output_folder=str(DATA / "videos"))
        video_renderer(model, data, duration=duration, video_recorder=rec)
        print(f"[OK] Video saved under {DATA/'videos'}")
    elif mode == "simple":
        simple_runner(model, data, duration=duration)
    elif mode == "frame":
        single_frame_renderer(model, data, save=True, save_path=str(DATA / "robot.png"))
        print(f"[OK] Frame saved to {DATA / 'robot.png'}")'''

def visualize(robot_spec,
              mode: Literal["launcher","video","simple","no_control","frame","timestep"]="launcher",
              duration: float = DURATION,
              scene: str = "rugged"):
    match = [s for s in SCENARIOS if s[0] == scene]
    if not match:
        print(f"[ERR] Unknown scene '{scene}', choose from {[s[0] for s in SCENARIOS]}")
        return
    _, spos, dur = match[0]

    world, model = build_world_and_model_at(robot_spec, spos)
    nn_obj = NNController(nq=model.nq, nv=model.nv, nu=model.nu, hidden=24, freq_hz=3.0)

    sig = morph_sig(model)
    params_path = DATA / f"best_controller_params_{sig}.npy"
    if not params_path.exists():
        print(f"[ERR] Missing {params_path}. Train for this body first.")
        return
    unflatten_params(nn_obj, np.load(params_path))

    global CTRL_RANGE_ARR, AMP, MID, ACT_GAIN, ROBOT_BODY_ID, ROBOT_BODY_NAME
    CTRL_RANGE_ARR = model.actuator_ctrlrange.copy()
    AMP = (CTRL_RANGE_ARR[:,1] - CTRL_RANGE_ARR[:,0]) * 0.5
    MID = (CTRL_RANGE_ARR[:,1] + CTRL_RANGE_ARR[:,0]) * 0.5
    ACT_GAIN = 1.0
    ROBOT_BODY_ID, ROBOT_BODY_NAME = find_robot_root_body(model)

    def _cb(m, d): return nn_control_step(nn_obj, m, d, TARGET_POS)
    #tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_BODY, name_to_bind=ROBOT_BODY_NAME)
    
    mujoco_type_to_find = mj.mjtObj.mjOBJ_GEOM
    name_to_bind = "core"
    tracker = Tracker(
        mujoco_obj_to_find=mujoco_type_to_find,
        name_to_bind=name_to_bind,
    )

    
    
    data = mj.MjData(model); tracker.setup(world.spec, data)
    ctrl = Controller(controller_callback_function=_cb, tracker=tracker)
    mj.set_mjcb_control(ctrl.set_control)
    

    if mode == "launcher":
        viewer.launch(model=model, data=data)
    elif mode == "video":
        rec = VideoRecorder(output_folder=str(DATA / "videos"))
        video_renderer(model, data, duration=dur, video_recorder=rec)
        print(f"[OK] Video saved under {DATA/'videos'}")
    elif mode == "simple":
        simple_runner(model, data, duration=dur)
    elif mode == "frame":
        single_frame_renderer(model, data, save=True, save_path=str(DATA / "robot.png"))
        print(f"[OK] Frame saved to {DATA / 'robot.png'}")
        
    print(SPAWN_POS)
    print(tracker.history["xpos"][0])
    show_xpos_history(
            tracker.history["xpos"][0],
            spawn_position=SPAWN_POS,
            target_position=TARGET_POS,
            save=True,
            show=True,
        )



def plot_xy(traj_xy, title="Robot path"):
    if traj_xy is None or len(traj_xy) < 2:
        print("[plot] no trajectory")
        return
    x, y = traj_xy[:, 0], traj_xy[:, 1]
    plt.figure(figsize=(6, 4))
    plt.plot(x, y, linewidth=2)
    plt.scatter([x[0]], [y[0]], marker='o', label='start')
    plt.scatter([x[-1]], [y[-1]], marker='x', label='end')
    plt.axis('equal'); plt.grid(True, alpha=0.3); plt.legend()
    plt.title(title); plt.xlabel("x [m]"); plt.ylabel("y [m]")
    plt.tight_layout(); plt.show()

# ---------- CLI ----------
# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--mode", choices=["train", "visualize"], default="train")
#     parser.add_argument("--iters", type=int, default=100)
#     parser.add_argument("--pop", type=int, default=128)
#     parser.add_argument("--dur", type=float, default=60.0)
#     parser.add_argument("--viz", choices=["launcher","video","simple","frame"], default="launcher")
#     parser.add_argument("--scene", type=str, default="rugged",
#                         help="which scenario to visualize (rugged or uphill)")
#     parser.add_argument("--resume", action="store_true",
#                     help="continue training from previous best_controller_params.npy", default=False)
#     args = parser.parse_args()

#     if args.mode == "train":
#         train_controller(DATA / "best.csv", duration=args.dur,
#                      iterations=args.iters, popsize=args.pop,
#                      resume=args.resume)
#     else:
#         visualize(mode=args.viz, duration=args.dur)

########################################



# ----------------------------
# Neural Network Controller
# ----------------------------
# class NNController(nn.Module):
#     def __init__(self, nu: int, hidden_size: int = 8):
#         super().__init__()
#         self.nu = nu
#         self.in_dim = nu + 4   # prev_ctrl + sin/cos + delta_x, delta_y
#         self.hidden = hidden_size
#         self.out_dim = nu

#         self.fc1 = nn.Linear(self.in_dim, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, hidden_size)
#         self.fc3 = nn.Linear(hidden_size, self.out_dim)
#         self.tanh = nn.Tanh()

#     def forward(self, prev_ctrl: torch.Tensor, t: float, target_pos: torch.Tensor, current_pos: torch.Tensor):
#         t_tensor = torch.tensor(t, dtype=torch.float32)
#         w = 2 * torch.pi
#         delta = target_pos[:2] - current_pos[:2]
        
#         # Use t_tensor here
#         time_features = torch.stack([torch.sin(w * t_tensor), torch.cos(w * t_tensor)])
        
#         obs = torch.cat([prev_ctrl, time_features, delta], dim=0)
#         h = self.tanh(self.fc1(obs))
#         h = self.tanh(self.fc2(h))
#         y = self.tanh(self.fc3(h))
#         return CTRL_RANGE * y
    

def nn_controller(
    model: mj.MjModel,
    data: mj.MjData,
) -> npt.NDArray[np.float64]:
    # Simple 3-layer neural network
    input_size = len(data.qpos)
    hidden_size = 8
    output_size = model.nu

    # Initialize the networks weights randomly
    # Normally, you would use the genes of an individual as the weights,
    # Here we set them randomly for simplicity.
    w1 = RNG.normal(loc=0.0138, scale=0.5, size=(input_size, hidden_size))
    w2 = RNG.normal(loc=0.0138, scale=0.5, size=(hidden_size, hidden_size))
    w3 = RNG.normal(loc=0.0138, scale=0.5, size=(hidden_size, output_size))

    # Get inputs, in this case the positions of the actuator motors (hinges)
    inputs = data.qpos

    # Run the inputs through the lays of the network.
    layer1 = np.tanh(np.dot(inputs, w1))
    layer2 = np.tanh(np.dot(layer1, w2))
    outputs = np.tanh(np.dot(layer2, w3))

    # Scale the outputs
    return outputs * np.pi



def experiment(
    robot: Any,
    controller: Controller,
    duration: int = 5,
    mode: Literal["launcher", "video", "simple", "no_control", "frame", "timestep"] = "simple",
) -> tuple[Optional[float], Optional[float], Optional[npt.NDArray[np.float64]]]: # type: ignore
    
    """Run the simulation with random movements."""
    # ==================================================================== #
    # Initialise controller to controller to None, always in the beginning.
    mj.set_mjcb_control(None)  # DO NOT REMOVE
    
    # Initialise world
    # Import environments from ariel.simulation.environments
    world = OlympicArena(load_precompiled=False,)

    # Spawn robot in the world
    # Check docstring for spawn conditions
    world.spawn(robot.spec, position=[0, 0, 0.04])
    #gecko_core = gecko()  # DO NOT CHANGE
    #world.spawn(gecko_core.spec, spawn_position=[0, 0, 0])
    # Generate the model and data
    # These are standard parts of the simulation USE THEM AS IS, DO NOT CHANGE
    model = world.spec.compile()
    data = mj.MjData(model)
    
    # Reset state and time of simulation
    mj.mj_resetData(model, data)
    
    # Count hinge joints
    hinge_indices = np.where(model.jnt_type == mj.mjtJoint.mjJNT_HINGE)[0]
    num_hinges = len(hinge_indices)
    qacc_fitness  = None
    # Count blocks (bodies)
    # Exclude the world body (index 0) if you only want the robot parts
    num_blocks = model.nbody - 1  

    print("Number of hinge joints:", num_hinges)
    print("Number of blocks (bodies) used:", num_blocks)
    if num_hinges < 7 or num_blocks < 30:
        print("insufficient body")
        qacc_fitness = 0
        return None, qacc_fitness, None
    
    '''# Pass the model and data to the tracker
    if controller.tracker is not None:
        controller.tracker.setup(world.spec, data)
    
    # Set the control callback function
    # This is called every time step to get the next action.
    args: list[Any] = []  # IF YOU NEED MORE ARGUMENTS ADD THEM HERE!
    kwargs: dict[Any, Any] = {}  # IF YOU NEED MORE ARGUMENTS ADD THEM HERE!

    mj.set_mjcb_control(
        lambda m, d: controller.set_control(m, d, *args, **kwargs),
    )'''

    # Pass the model and data to the tracker (if any)
    if controller.tracker is not None:
        controller.tracker.setup(world.spec, data)

    args: list[Any] = []
    kwargs: dict[Any, Any] = {}

    # ------------------------------------------------------------------ #
    
    # Record starting position
    start_position = np.copy(data.qpos[:2])  # store x,y
    
    match mode:
        case "simple":
            mj.set_mjcb_control(lambda m, d: controller.set_control(m, d, *args, **kwargs))
            simple_runner(model, data, duration=duration)

        case "frame":
            mj.set_mjcb_control(lambda m, d: controller.set_control(m, d, *args, **kwargs))
            save_path = str(DATA / "robot.png")
            single_frame_renderer(model, data, save=True, save_path=save_path)

        case "video":
            mj.set_mjcb_control(lambda m, d: controller.set_control(m, d, *args, **kwargs))
            path_to_video_folder = str(DATA / "videos")
            video_recorder = VideoRecorder(output_folder=path_to_video_folder)
            video_renderer(model, data, duration=duration, video_recorder=video_recorder)

        case "launcher":
            mj.set_mjcb_control(lambda m, d: controller.set_control(m, d, *args, **kwargs))
            viewer.launch(model=model, data=data)

        case "no_control":
            mj.set_mjcb_control(None)
            viewer.launch(model=model, data=data)

        case "timestep":
            # IMPORTANT: no controller during feasibility
            mj.set_mjcb_control(None)
            print("in timestep")
            num_steps = int(duration / model.opt.timestep)
            maxQacc = 0.0
            #bad_streak = 0
            warmup_steps = int(0.2 / model.opt.timestep)  # ignore first 0.2s

            for step in range(num_steps):
                mj.mj_step(model, data)

                if step < warmup_steps:
                    continue

                qa = np.abs(data.qacc)
                if np.any(np.isnan(qa)) or np.any(np.isinf(qa)):
                    return None, 1.0, None  # NaN/Inf detected

                maxQacc = max(maxQacc, float(np.max(qa)))
                if maxQacc > 12000:
                    #bad_streak += 1. # opcion remover esto
                    #if bad_streak >= 3:   # require persistence.  #. opcion remover esto
                    return None, maxQacc, None
                #else:
                 #   bad_streak = 0

    '''match mode:
        
        case "simple":
            # This disables visualisation (fastest option)
            simple_runner(
                model,
                data,
                duration=duration,
            )
        case "frame":
            # Render a single frame (for debugging)
            save_path = str(DATA / "robot.png")
            single_frame_renderer(model, data, save=True, save_path=save_path)
        case "video":
            # This records a video of the simulation
            path_to_video_folder = str(DATA / "videos")
            video_recorder = VideoRecorder(output_folder=path_to_video_folder)

            # Render with video recorder
            video_renderer(
                model,
                data,
                duration=duration,
                video_recorder=video_recorder,
            )
        case "launcher":
            # This opens a liver viewer of the simulation
            viewer.launch(
                model=model,
                data=data,
            )
        case "no_control":
            # If mj.set_mjcb_control(None), you can control the limbs manually.
            mj.set_mjcb_control(None)
            viewer.launch(
                model=model,
                data=data,
            )
        case "timestep":
            print("in timestep")
            # Manually step simulation and check for instability
            unstable = False
            num_steps = int(duration / model.opt.timestep)
            maxQacc = 0
            for _ in range(num_steps):
                mj.mj_step(model, data)
                
                # Check for NaN, Inf, or huge qacc
                if (np.any(np.isnan(data.qacc)) or
                    np.any(np.isinf(data.qacc)) or
                    np.any(np.abs(data.qacc) > 12000)):
                    print("Unstable simulation detected! Aborting timestep mode.", max(np.abs(data.qacc)))
                    return None, max(np.abs(data.qacc))
                #if np.any(np.abs(data.qacc) > maxQacc):
                 #   maxQacc =max(np.abs(data.qacc))
                  #  print(maxQacc)'''

    # After simulation — record end position
    end_position = np.copy(data.qpos[:2])  # final x,y
    distance = np.linalg.norm(end_position - start_position)

    # Optional: collect XY trajectory if a tracker exists and recorded 'xpos'
    traj_xy = None
    if controller.tracker is not None:
        hist = controller.tracker.history.get("xpos", {})
        if hist:
            key = next(iter(hist))
            arr = np.asarray(hist[key])
            if arr.ndim == 2 and arr.shape[1] >= 2:
                traj_xy = arr[:, :2].copy()

    # Return (distance, feasibility penalty or None, trajectory)
    return distance, qacc_fitness, traj_xy     
  
    '''# After simulation — record end position
    end_position = np.copy(data.qpos[:2])  # final x,y

    # Compute Euclidean distance travelled
    distance = np.linalg.norm(end_position - start_position)
    
    # Return the scalar distance
    return distance, qacc_fitness'''



def makeBody(num_modules: int = 20, genotype=None, simulation = "timestep", duration = 5):
    """
    Add the random moves method to check if the body is feasible
    or not so give it a quick simulation
    1. If feasible, then return the body
    2. If not feasible, create a new body
    
    """
   
    nde = NeuralDevelopmentalEncoding(number_of_modules=num_modules)
    p_matrices = nde.forward(genotype)
    hpd = HighProbabilityDecoder(num_modules)
    robot_graph = hpd.probability_matrices_to_graph(
        p_matrices[0], p_matrices[1], p_matrices[2]
    )
    core = construct_mjspec_from_graph(robot_graph)
        
   
    #simulate core quickly to see if it is feasible
    mujoco_type_to_find = mj.mjtObj.mjOBJ_GEOM
    name_to_bind = "core"
    tracker = Tracker(
        mujoco_obj_to_find=mujoco_type_to_find,
        name_to_bind=name_to_bind,
    )
    # ? ------------------------------------------------------------------ #
    # Simulate the robot
    ctrl = Controller(
        controller_callback_function=nn_controller,
        # controller_callback_function=random_move,
        tracker=tracker,
    )
    distance, weirdfitness, traj = experiment(robot=core, controller=ctrl, mode=simulation, duration=duration)

    if simulation != "timestep":
        plot_xy(traj, "Feasibility run path")

    if weirdfitness is not None and weirdfitness != 0:
        # unstable
        print("got weird fitness (unstable feasibility)")
        return core, weirdfitness

    if weirdfitness == 0:
        # insufficient body
        print("insufficient body feasibility")
        return core, 0

    # feasible
    WORKING_BODIES.add(core)
    return core, None


'''def makeIndividual(body,genotype,QACC_fitness):
    #Create a body
    #Spawn core to get model, for nu
    if QACC_fitness:
        if QACC_fitness == 0:
            individual ={
                "genotype": genotype,
                "robot_spec": body,
                "nn": None,
                "fitness": 10
            }
        else:
            individual ={
                "genotype": genotype,
                "robot_spec": body,
                "nn": None,
                "fitness": QACC_fitness / 10000
            }
        return individual
    else:
        #Pass numm hinges as inputsize NN 
        nn_obj = 0
        #Set target_position
        target_pos = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        # Train and evaluate
        
        trained_nn, fitness = train_controller(DATA / "best.csv", duration= 10,
                     iterations=5, popsize=5,
                     resume= False)
        #train_Net(model, world, nn_obj, core, target_pos, duration = 5.0, iterations=50, lr=0.01)
        individual ={
                "genotype": genotype,
                "robot_spec": body,
                "nn": trained_nn,
                "fitness": fitness
            }
       
        return individual'''
BRAINTIME = []
def makeIndividual(body, genotype, QACC_fitness):
    global BRAINTIME
   
    # QACC_fitness meanings from experiment():
    #   None  -> feasible (train brain)
    #   0     -> insufficient body (penalize, skip training)
    #  > 0    -> unstable/NaN/high qacc (penalize, skip training)

    if QACC_fitness is None:
        # feasible -> train controller
        startbrain = time.time()
        trained_nn, fitness = train_controller(
        DATA / f"{GLOBAL_SEED}_best.csv",
        robot_spec=body,          # ← pass the candidate body
        duration=5, iterations=100, popsize=5, resume=True
    )
        endbrain = time.time()
        BRAINTIME.append(endbrain-startbrain)
        print(f"THE TIME FOR A BRAIN IS {BRAINTIME}\n\n\n\n\n\n\n\n")
        return {
            "genotype": genotype,
            "robot_spec": body,
            "nn": trained_nn,
            "fitness": float(fitness),
        }

    # infeasible or unstable -> penalize
    if QACC_fitness == 0:
        # insufficient body
        fitness = -1e6
    else:
        # unstable -> larger qacc is worse
        fitness = -float(QACC_fitness)

    return {
        "genotype": genotype,
        "robot_spec": body,
        "nn": None,
        "fitness": fitness,
    }
 
def flatten_genotype(genotype):
    """Flatten the genotype (3 vectors of size 64) into a single 192-element vector."""
    return np.concatenate(genotype)

def unflatten_genotype(flat_vector):
    """Convert a 192-element vector back into a genotype (3 vectors of size 64)."""
    return [flat_vector[0:64], flat_vector[64:128], flat_vector[128:192]]

def train_mlp_cmaes(out_csv, generations=100, pop_size=30, seed=0, sigma=0.5):
    rng = np.random.default_rng(seed)
    dim = 192
    x0 = rng.random(dim)
    es = cma.CMAEvolutionStrategy(x0, sigma, {'popsize': pop_size, 'seed': seed})

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    best_fit = -np.inf          # maximize
    best_candidate = None

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["generation", "best_fitness", "mean_fitness", "median_fitness"])

        for g in range(generations):
            solutions = es.ask()
            fit_pos: list[float] = []     # keep POSITIVE fitnesses here

            for sol in solutions:
                genotype = unflatten_genotype(np.clip(sol, 0.1, 0.9).astype(np.float32))
                body, qacc_fitness = makeBody(num_modules=20, genotype=genotype)
                indiv = makeIndividual(body, genotype, qacc_fitness)

                f_val = float(indiv["fitness"])     # ensure numeric
                fit_pos.append(f_val)

                if f_val > best_fit:                # maximize
                    best_fit = f_val
                    best_candidate = indiv

            # CMA-ES MINIMIZES -> pass NEGATIVES
            fitnesses_neg = [-f for f in fit_pos]
            es.tell(solutions, fitnesses_neg)

            # Log stats using positive values
            gen_best = float(np.max(f_pos := np.array(fit_pos, dtype=float)))
            gen_mean = float(np.mean(f_pos))
            gen_median = float(np.median(f_pos))
            writer.writerow([g, gen_best, gen_mean, gen_median])

    return best_candidate, best_fit




def makeGenotype(num_modules: int = 20):
    genotype_size = 64
    type_p = RNG.random(genotype_size).astype(np.float32)
    conn_p = RNG.random(genotype_size).astype(np.float32)
    rot_p = RNG.random(genotype_size).astype(np.float32)
    genotype = [type_p, conn_p, rot_p]
    return genotype


def initializePopulation(pop_size: int = 10, num_modules: int = 20):
    # add unique bodies to the population
    population = []
    all_fitness = []
    for _ in range(pop_size):
        genotype = makeGenotype(num_modules)
        body,Qacc_fitness = makeBody(num_modules=20, genotype=genotype)
        individual = makeIndividual(body, genotype,Qacc_fitness)
        all_fitness.append(individual["fitness"])
        population.append(individual)
        
    return population, all_fitness



def mutation(a, b, c, F):
    # a, b, c are 3-vector genotypes (each vector length 64)
    mutant_genotype = [np.zeros(64, dtype=np.float32) for _ in range(3)]
    for vector in range(3):
        # ensure we operate on numpy arrays
        A = np.asarray(a[vector], dtype=np.float32)
        B = np.asarray(b[vector], dtype=np.float32)
        C = np.asarray(c[vector], dtype=np.float32)
        mutant_genotype[vector] = A + F * (B - C)
    return mutant_genotype

def crossover(parent, mutant, CR, vector_size=3, gene_size=64):
    """
    Make a crossover between the parent and the mutant per vector
    so that each vector has at least one gene from the mutant
    
    """
    rng = np.random.default_rng(SEED)
    crossover_genotype =[]
    for i in range(vector_size):
        cross = rng.random(gene_size) < CR
        cross[rng.integers(0, gene_size)] = True 
        trial_vector = np.where(cross, mutant[i], parent[i])
        crossover_genotype.append(trial_vector)
    
    return crossover_genotype


#  Differential Evolution on MLP (1b)
def train_mlp_de(out_csv, generations=100, pop_size=30, T=10.0, seed=0, model=None, data=None, core_bind=None,
                 hidden=32, freq_hz=1.0, F=0.7, CR=0.9, init_scale=0.5, alpha=0.2):
    """
    Simple, Differential Evolution for MLP parameters.
    """
    rng = np.random.default_rng(seed)

    # Initialize population by sampling normal distribution with mean 0 and stdv of init_scale
    pop,fits = initializePopulation(pop_size=pop_size, num_modules=20)
    print("made population")
    #Finds best fitnesses
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    best_idx = int(np.argmax(fits))
    best_candidate = pop[best_idx].copy()
    best_fit = float(fits[best_idx])

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["generation", "best_fitness", "mean_fitness", "median_fitness"])
        for g in range(generations):
            print(f"NOW DOING GENERATION {g+1} OUT OF {generations}")
            for i in range(pop_size):
                print(f"NOW DOING BODY NUMBER {i+1} out of {pop_size} in generation {g+1}")
                #Select all indices expcept for i, the candidate you want to mutate
                choices = []
                for index in range(pop_size):
                    if index != i:
                        choices.append(index)
                #Out of valid choices, choose 3 random candidates 
                a, b, c = rng.choice(choices, size=3, replace=False)
                #Mutant candidate:
                mutant = mutation(pop[a]["genotype"],pop[b]["genotype"],pop[c]["genotype"],F)
                trial = crossover(pop[i]["genotype"], mutant, CR, vector_size=3, gene_size=64)
                # Evaluate trial
                trialbody,Qacc_fitness = makeBody(num_modules=20, genotype=trial)
                trialindividual = makeIndividual(trialbody, trial,Qacc_fitness)
                f_trial = trialindividual["fitness"] #fitness is negative distance to target

                # If trial is better than parent than trial enters population
                if f_trial >= fits[i]:
                    pop[i] = trialindividual
                    fits[i] = f_trial

            # Retrieve stats: best fitness, mean fitness and median fitness and write to csv
            gen_best = float(np.max(fits))
            gen_mean = float(np.mean(fits))
            gen_median = float(np.median(fits))
            writer.writerow([g, gen_best, gen_mean, gen_median])

            # Update and check best candidate with best fitness
            if gen_best > best_fit:
                best_fit = gen_best
                best_candidate = pop[int(np.argmax(fits))].copy()

            #Print every after every 10th generation
            #if (g + 1) % max(1, generations // 10) == 0:
                #print(f"[mlp_de seed {seed}] Gen {g+1}/{generations}: best={gen_best:.3f} mean={gen_mean:.3f}")

    return best_candidate, best_fit




def TrainDummyNet(nn_obj): #(nn_obj: NNController):   
    RNG = np.random.default_rng(SEED)  # No fixed seed
    fitness = RNG.random()
    return None, fitness



def plot(file_path,output_path):
        # --- Load your results file ---
    # Works for both .csv and .xlsx
    file_path = str(file_path)
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    # --- Inspect columns ---
    print("Columns:", df.columns.tolist())

    # Assuming your file has columns: generation, best_fitness, mean_fitness, median_fitness
    plt.figure(figsize=(8, 5))

    # --- Plot the fitness trends ---
    plt.plot(df["generation"], df["best_fitness"], label="Best fitness", linewidth=2)
    plt.plot(df["generation"], df["mean_fitness"], label="Mean fitness", linestyle="--")
    plt.plot(df["generation"], df["median_fitness"], label="Median fitness", linestyle=":")

    # --- Add labels and styling ---
    plt.xlabel("Generation", fontsize=12)
    plt.ylabel("Fitness", fontsize=12)
    plt.title("CMA-ES Fitness over Generations", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    # === Save the plot ===
    plt.savefig(output_path, dpi=300)
    print(f"✅ Plot saved to {output_path}")
    
    
    
    
    
    
    
def load_series(csv_path):
    print(csv_path)
    df = pd.read_csv(csv_path)
    return df["best_fitness"].to_numpy()


def plot_experiment(exp_name, exp_csvs, final_exp ,out_png, window=10, title=None, xlabel="Generation", ylabel="Fitness"):
    # Load runs
    print(exp_csvs)
    exp_runs = [load_series(p) for p in exp_csvs]
    if len(exp_runs) == 0:
        print(f"[plot] No runs found for {exp_name}")
        return
    L = min(len(r) for r in exp_runs)
    exp_runs = [r[:L] for r in exp_runs]
    exp_mean = np.mean(exp_runs, axis=0)
    exp_std  = np.std(exp_runs, axis=0)


    x = np.arange(len(exp_mean))
    plt.figure(figsize=(8,4.5))
    # individual runs (light)
    k = 1 
    if not  final_exp:
        for r in exp_runs:
            plt.plot(np.arange(len(r[:L])), r[:L], linewidth=2, alpha=0.35, label = f"Run {exp_name} {k}")
            k+=1 
    # mean + std
    plt.plot(x, exp_mean, linewidth=2.5, label=f"{exp_name} Mean")
    if not final_exp:
        plt.fill_between(x, exp_mean-exp_std, exp_mean+exp_std, alpha=0.25, label=f"{exp_name} ± std", color = "pink")

    
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    if title is None:
        title = f"{exp_name}"
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.show()
    print(f"[plot] Saved {out_png}")

def show_xpos_history1(history: list[float]) -> None:
    # Create a tracking camera
    camera = mj.MjvCamera()
    camera.type = mj.mjtCamera.mjCAMERA_FREE
    camera.lookat = [2.5, 0, 0]
    camera.distance = 10
    camera.azimuth = 0
    camera.elevation = -90

    # Initialize world to get the background
    mj.set_mjcb_control(None)
    world = OlympicArena()
    model = world.spec.compile()
    data = mj.MjData(model)
    save_path = str(DATA / "background.png")
    single_frame_renderer(
        model,
        data,
        save_path=save_path,
        save=True,
    )

    # Setup background image
    img = plt.imread(save_path)
    _, ax = plt.subplots()
    ax.imshow(img)
    w, h, _ = img.shape

    # Convert list of [x,y,z] positions to numpy array
    pos_data = np.array(history)

    # Calculate initial position
    x0, y0 = int(h * 0.483), int(w * 0.815)
    xc, yc = int(h * 0.483), int(w * 0.9205)
    ym0, ymc = 0, SPAWN_POS[0]

    # Convert position data to pixel coordinates
    pixel_to_dist = -((ymc - ym0) / (yc - y0))
    pos_data_pixel = [[xc, yc]]
    for i in range(len(pos_data) - 1):
        xi, yi, _ = pos_data[i]
        xj, yj, _ = pos_data[i + 1]
        xd, yd = (xj - xi) / pixel_to_dist, (yj - yi) / pixel_to_dist
        xn, yn = pos_data_pixel[i]
        pos_data_pixel.append([xn + int(xd), yn + int(yd)])
    pos_data_pixel = np.array(pos_data_pixel)

    # Plot x,y trajectory
    ax.plot(x0, y0, "kx", label="[0, 0, 0]")
    ax.plot(xc, yc, "go", label="Start")
    ax.plot(pos_data_pixel[:, 0], pos_data_pixel[:, 1], "b-", label="Path")
    ax.plot(pos_data_pixel[-1, 0], pos_data_pixel[-1, 1], "ro", label="End")

    # Add labels and title
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.legend()

    # Title
    plt.title("Robot Path in XY Plane")

    # Show results
    plt.show()

def save_body_json_from_genotype(genotype, num_modules: int, out_path: str | Path):
    nde = NeuralDevelopmentalEncoding(number_of_modules=num_modules)
    type_p, conn_p, rot_p = nde.forward(genotype)
    hpd = HighProbabilityDecoder(num_modules)
    robot_graph = hpd.probability_matrices_to_graph(type_p, conn_p, rot_p)
    save_graph_as_json(robot_graph, str(out_path))



    
def main(action, generations = 10, pop_size = 5,seed = [GLOBAL_SEED]):
    global WORKING_BODIES
    
    if action == "CMAES":
        WORKING_BODIES = set()
        best_candidate_CMAES = None
        for s in seed:
            file_path =  f"./results/CMAES/CMAES_{generations}_{pop_size}_seed{s}.csv"
            print(f"Starting CMAES algorithm with {generations} generations and population {pop_size} and seed {s}...")
            start_cmaes = time.time()
            best_candidate_CMAES, best_fit_CMAES = train_mlp_cmaes(out_csv=file_path, generations=generations, pop_size=pop_size, seed=s)
            end_cmaes = time.time()
            print(f"Seed {s}")
            print("Best CMAES candidate fitness:", best_fit_CMAES)
            print("CMAES Time:", end_cmaes - start_cmaes)
            #simulate core quickly to see if it is feasible
            print(f"CMAES found {len(WORKING_BODIES)} bodies in total \n\n\n")
        if best_candidate_CMAES["robot_spec"]:
            print("find a body")
            makeBody(genotype = best_candidate_CMAES["genotype"], simulation = "launcher", duration = 15)
    
    if action == "DE":
        WORKING_BODIES = set()
        best_candidate_CMAES = None
        for s in seed:
            file_path =  f"./results_{GLOBAL_SEED}/DE/DE{generations}_{pop_size}_seed{s}.csv"
            print(f"Starting DE algorithm with {generations} generations and population {pop_size} and seed {s}...")
            start_de = time.time()
            best_candidate_DE, best_fit_DE = train_mlp_de(out_csv=file_path, generations=generations, pop_size=pop_size, seed=s)
            end_de = time.time()
            print(f"Seed {s}")
            print("Best DE candidate fitness:", best_fit_DE)
            print("DE Time:", (end_de - start_de)/60)
            #simulate core quickly to see if it is feasible
            print(f"DE found {len(WORKING_BODIES)} bodies in total \n\n\n")
        if best_candidate_DE and best_candidate_DE.get("robot_spec") is not None:
            visualize(robot_spec=best_candidate_DE["robot_spec"], mode="video", scene="rugged")
                #show_xpos_history(tracker.history["xpos"][0])
        if best_candidate_DE and best_candidate_DE.get("genotype") is not None:
            out_dir = DATA / "bodies"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_json = out_dir / f"{GLOBAL_SEED}_best_DE_body.json"
            save_body_json_from_genotype(best_candidate_DE["genotype"], num_modules=20, out_path=out_json)
            print(f"[OK] Saved best DE body JSON to {out_json}")
        
        

            
    if action == "plot CMAES":
        exp_csvs = sorted(glob.glob(f"./results/CMAES/CMAES_{generations}_{pop_size}_seed*.csv"))
        print(exp_csvs)
        plot_experiment(exp_name = "CMAES",
                        exp_csvs=exp_csvs,
                        final_exp=False,
                        out_png=f"./results/plotsA3/CMAES_{generations}_{pop_size}.png",
                        window=10,
                        title="fitness of CMAES over generations")
    if action == "plot DE":
        exp_csvs = sorted(glob.glob(f"./results/DE/DE{generations}_{pop_size}_seed*.csv"))
        print(exp_csvs)
        plot_experiment(exp_name = "DE",
                        exp_csvs=exp_csvs,
                        final_exp=False,
                        out_png=f"./results/plotsA3/DE{generations}_{pop_size}.png",
                        window=10,
                        title="fitness of DE over generations")
    print(f"all the braintimes {BRAINTIME}")
  
    
if __name__ == "__main__":
    action = "DE" # 'plot CMAES', 'plot DE', 'DE', 'CMAES'
    main(action)
    
