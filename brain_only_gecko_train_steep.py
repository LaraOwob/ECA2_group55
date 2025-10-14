# brain_only_gecko.py
# Minimal “brain-only” run on a single fixed body (gecko)
# Modes: train (CMA-ES controller optimization), visualize (load & watch)

from pathlib import Path
from typing import Literal
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

# ---------- Config ----------
SEED = 42
RNG = np.random.default_rng(SEED)
CTRL_RANGE = 1.0

SPAWN_POS = [0, 0.0, 0]
TARGET_POS = np.array([5.0, 0.0, 0.5], dtype=np.float32)
DURATION = 10.0

# --- scenarios you want to average over ---
SCENARIOS = [
    ("rugged", np.array([0.5, 0.0, 0.1], dtype=np.float32), 15.0),  # start at rugged
    ("uphill", np.array([2.5, 0.0, 0.1], dtype=np.float32), 20.0),  # start at steep
]

ACTION_LOG = []

ViewerTypes = Literal["launcher", "video", "simple", "frame"]

# Data dir
SCRIPT_NAME = Path(__file__).stem
DATA = Path("__data__") / SCRIPT_NAME
(DATA / "videos").mkdir(parents=True, exist_ok=True)


# ---------- Tiny NN controller ----------
class NNController(nn.Module):
    def __init__(self, nq: int, nv: int, nu: int, hidden: int = 32, freq_hz: float = 3.0):
        super().__init__()
        self.nq, self.nv, self.nu = nq, nv, nu
        self.in_dim = nq + nv + 2 + 2  # correct
        self.fc1 = nn.Linear(self.in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, nu)  # output matches number of actuators
        self.act = nn.Tanh()
        self.freq_hz = freq_hz

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

def build_world_and_model_at(spawn_pos):
    mj.set_mjcb_control(None)
    world = OlympicArena()
    core = gecko()
    world.spawn(core.spec, spawn_position=spawn_pos.tolist(), correct_for_bounding_box=True)
    model = world.spec.compile()

    # --- diagnostic: print actuator names and ranges ---
    print("\n[Actuator ranges]")
    for i in range(model.nu):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, i)
        lo, hi = model.actuator_ctrlrange[i]
        print(f"{i:02d} {name:20s} [{lo:.2f}, {hi:.2f}]")
    print("-------------------------------------------------\n")

    return world, model

def prepare_scenarios(): # type: ignore
    worlds, models, durations = [], [], []
    for name, spos, dur in SCENARIOS: # type: ignore
        w, m = build_world_and_model_at(spos)
        worlds.append(w); models.append(m); durations.append(dur)
    # actuator scaling globals (use the first; same morphology)
    global CTRL_RANGE_ARR, AMP, MID, ACT_GAIN, ROBOT_BODY_ID, ROBOT_BODY_NAME
    CTRL_RANGE_ARR = models[0].actuator_ctrlrange.copy()
    AMP = (CTRL_RANGE_ARR[:,1] - CTRL_RANGE_ARR[:,0]) * 0.5
    MID = (CTRL_RANGE_ARR[:,1] + CTRL_RANGE_ARR[:,0]) * 0.5
    ACT_GAIN = 1.2
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
    # use body (root) pos for current_pos (as you set up)
    current_pos = torch.tensor(d.xpos[ROBOT_BODY_ID], dtype=torch.float32)

    qpos = torch.tensor(d.qpos[:nn_obj.nq], dtype=torch.float32)  # nq
    qvel = torch.tensor(d.qvel[:nn_obj.nv], dtype=torch.float32)  # nv

    raw = nn_obj(qpos, qvel, d.time, torch.tensor(target_pos), current_pos).detach().numpy()

    # indices: (2,3)=FL, (4,5)=FR, (6)=BL, (7)=BR
    SIGN = np.ones(m.nu)
    # try flipping right-side joints just to test
    SIGN[[4,5,7]] = -1.0
    raw *= SIGN

    # raw has length nu; scale to actuator range:
    a = MID + (AMP * ACT_GAIN) * np.clip(raw, -1.0, 1.0)

    a = np.clip(a, CTRL_RANGE_ARR[:, 0], CTRL_RANGE_ARR[:, 1])

    # log normalized command per joint: roughly in [-ACT_GAIN, ACT_GAIN]
    norm_u = (a - MID) / np.maximum(AMP, 1e-6)
    ACTION_LOG.append(norm_u.copy())

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

    tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_BODY, name_to_bind=ROBOT_BODY_NAME)
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
    floor_pen = 20.0 if np.any(zs < 0.0) else 0.0

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
    latvel_w_step = np.where(on_slope, 0.3 * 0.3, 0.3)

    lat_step = np.abs(ys[1:]) - y_tol
    lateral_pen = float(np.sum(np.clip(lat_step, 0.0, None) * lat_w_step))

    vy = dy / dt
    latvel_pen = float(np.sum(np.clip(np.abs(vy) - 0.05, 0.0, None) * latvel_w_step))

    # --- backward motion penalty (smaller on slope) ---
    w_back_flat, w_back_slope = 0.5, 0.15
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
    yaw_align = 0.05 * np.cos(rot_to_yaw(xmat9))

    # --- symmetry penalty (encourage L/R parity) ---
    sym_pairs = [(2, 4), (3, 5), (6, 7)]
    if ACTION_LOG:
        A = np.asarray(ACTION_LOG)  # [T, nu]
        mu = np.mean(np.abs(A), axis=0)
        sym_pen = 0.1 * sum((mu[i] - mu[j])**2 for i, j in sym_pairs)
    else:
        sym_pen = 0.0
    ACTION_LOG.clear()

    # --- final score ---
    fitness = (
        progress + vx_rew + dz_rew + finish_bonus + speed_bonus + climb_bonus + yaw_align
        - lateral_pen - latvel_pen - back_pen - fall_pen - floor_pen - oob_pen - sym_pen
    )
    return float(fitness)


# ---------- Build world+model once for gecko ----------
def build_world_and_model():
    mj.set_mjcb_control(None)
    world = OlympicArena()
    core = gecko()
    world.spawn(core.spec, spawn_position=SPAWN_POS, correct_for_bounding_box=True)
    model = world.spec.compile()

    # --- diagnostic: print actuator names and ranges ---
    print("\n[Actuator ranges]")
    for i in range(model.nu):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, i)
        lo, hi = model.actuator_ctrlrange[i]
        print(f"{i:02d} {name:20s} [{lo:.2f}, {hi:.2f}]")
    print("-------------------------------------------------\n")

    # --- actuator range scaling ---
    global CTRL_RANGE_ARR, AMP, MID, ACT_GAIN
    CTRL_RANGE_ARR = model.actuator_ctrlrange.copy()
    AMP = (CTRL_RANGE_ARR[:, 1] - CTRL_RANGE_ARR[:, 0]) * 0.5
    MID = (CTRL_RANGE_ARR[:, 1] + CTRL_RANGE_ARR[:, 0]) * 0.5
    ACT_GAIN = 1.2
    # --------------------------------

    # --- find and cache robot body ID ---
    global ROBOT_BODY_ID, ROBOT_BODY_NAME
    ROBOT_BODY_ID, ROBOT_BODY_NAME = find_robot_root_body(model)
    # -------------------------------------

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
                     duration: float = DURATION,
                     iterations: int = 12,
                     popsize: int = 16,
                     hidden: int = 32,
                     resume: bool = False,
                     init_sigma: float = 0.8):
    """
    CMA-ES training over NN weights, averaged across prepared scenarios.
    - resume=True  -> warm-start from __data__/.../best_controller_params.npy if present
    - resume=False -> start from random initialization
    """
    # build scenarios once
    worlds, models, durations = prepare_scenarios()

    # controller sized to the (identical) gecko model
    nn_obj = NNController(nq=models[0].nq, nv=models[0].nv, nu=models[0].nu,
                          hidden=hidden, freq_hz=3.0)

    # ----- warm start (optional) -----
    DATA.mkdir(parents=True, exist_ok=True)
    warm_path = DATA / "best_controller_params.npy"
    x0 = flatten_params(nn_obj)
    if resume and warm_path.exists():
        try:
            x0 = np.load(warm_path)
            unflatten_params(nn_obj, x0)
            print("[INFO] Warm-starting from previous best_controller_params.npy")
        except Exception as e:
            print(f"[WARN] Could not warm-start ({e}). Starting fresh.")

    # ----- CMA-ES -----
    es = cma.CMAEvolutionStrategy(x0, init_sigma, {
        'popsize': popsize,
        'maxiter': iterations,
        'verb_disp': 1,
        'verb_log': 0
    })

    history = []
    try:
        while not es.stop():
            xs = es.ask()
            fs = []
            for x in xs:
                unflatten_params(nn_obj, x)
                f_mean = mean_fitness_over_scenarios(nn_obj, worlds, models, durations)
                fs.append(-f_mean)  # CMA-ES minimizes
            es.tell(xs, fs)
            history.append(-float(np.min(fs)))
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Stopping early—saving best-so-far.")
    finally:
        es.result_pretty()
        best_x = es.result.xbest
        best_f = -es.result.fbest
        unflatten_params(nn_obj, best_x)

        # save artifacts
        np.savetxt(DATA / "fitness_curve.csv", np.array(history), delimiter=",")
        np.save(warm_path, best_x)
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["best_fitness"])
            w.writerow([best_f])

        print(f"[DONE] Best avg fitness over {len(SCENARIOS)} scenarios: {best_f:.3f} "
              f"({'resumed' if resume else 'fresh'})")

    return best_f, nn_obj


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

def visualize(mode: ViewerTypes = "launcher", duration: float = DURATION, scene: str = "rugged"):
    # pick scenario by name
    match = [s for s in SCENARIOS if s[0] == scene]
    if not match:
        print(f"[ERR] Unknown scene '{scene}', choose from {[s[0] for s in SCENARIOS]}")
        return
    _, spos, dur = match[0]

    spawn_override = np.array([2.5, 0.0, 0.12], dtype=np.float32)

    world, model = build_world_and_model_at(spawn_override)
    nn_obj = NNController(nq=model.nq, nv=model.nv, nu=model.nu, hidden=32, freq_hz=3.0)

    params_path = DATA / "best_controller_params.npy"
    if not params_path.exists():
        print(f"[ERR] Missing {params_path}. Run with mode=train first.")
        return
    unflatten_params(nn_obj, np.load(params_path))

    global CTRL_RANGE_ARR, AMP, MID, ACT_GAIN, ROBOT_BODY_ID, ROBOT_BODY_NAME
    CTRL_RANGE_ARR = model.actuator_ctrlrange.copy()
    AMP = (CTRL_RANGE_ARR[:,1] - CTRL_RANGE_ARR[:,0]) * 0.5
    MID = (CTRL_RANGE_ARR[:,1] + CTRL_RANGE_ARR[:,0]) * 0.5
    ACT_GAIN = 1.2
    ROBOT_BODY_ID, ROBOT_BODY_NAME = find_robot_root_body(model)

    def _cb(m, d): return nn_control_step(nn_obj, m, d, TARGET_POS)
    tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_BODY, name_to_bind=ROBOT_BODY_NAME)
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


# ---------- CLI ----------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "visualize"], default="train")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--pop", type=int, default=128)
    parser.add_argument("--dur", type=float, default=60.0)
    parser.add_argument("--viz", choices=["launcher","video","simple","frame"], default="launcher")
    parser.add_argument("--scene", type=str, default="rugged",
                        help="which scenario to visualize (rugged or uphill)")
    parser.add_argument("--resume", action="store_true",
                    help="continue training from previous best_controller_params.npy", default=False)
    args = parser.parse_args()

    if args.mode == "train":
        train_controller(DATA / "best.csv", duration=args.dur,
                     iterations=args.iters, popsize=args.pop,
                     resume=args.resume)
    else:
        visualize(mode=args.viz, duration=args.dur)