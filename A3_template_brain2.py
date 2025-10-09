"""Assignment 3 template code."""
import cma


# Standard library
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import mujoco as mj
import numpy as np
import numpy.typing as npt
from mujoco import viewer

# Local libraries
from ariel import console
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
from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.utils.runners import simple_runner
from ariel.utils.tracker import Tracker
from ariel.utils.video_recorder import VideoRecorder

# Type Checking
if TYPE_CHECKING:
    from networkx import DiGraph

# Type Aliases
type ViewerTypes = Literal["launcher", "video", "simple", "no_control", "frame"]

# --- RANDOM GENERATOR SETUP --- #
SEED = 42
RNG = np.random.default_rng(SEED)

# --- DATA SETUP ---
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True)

# Global variables
SPAWN_POS = [-0.8, 0, 0.1]
NUM_OF_MODULES = 30
TARGET_POSITION = [5, 0, 0.5]

# Multi-spawn anchors (tune if needed)
SPAWN_FLAT   = [-0.8, 0.0, 0.10]  # start
SPAWN_RUGGED = [ 0.5, 0.0, 0.1]  # mid arena
SPAWN_UPHILL = [ 2.5, 0.0, 0.1]  # near start of uphill
SPAWN_LIST   = [SPAWN_FLAT, SPAWN_RUGGED, SPAWN_UPHILL]


def fitness_function(history: list[float]) -> float:
    xt, yt, zt = TARGET_POSITION
    xc, yc, zc = history[-1]

    # Minimize the distance --> maximize the negative distance
    cartesian_distance = np.sqrt(
        (xt - xc) ** 2 + (yt - yc) ** 2 + (zt - zc) ** 2,
    )
    return -cartesian_distance


def show_xpos_history(history: list[float]) -> None:
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
        camera=camera,
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

# ===== CPG Brain (Phase A) ============================================= #
def _sample_cpg_params(nu: int, rng: np.random.Generator = RNG) -> npt.NDArray[np.float64]:
    """Sample reasonable CPG params for a model with `nu` actuators.
    Layout: [a_0..a_{nu-1}, phi_0..phi_{nu-1}, b_0..b_{nu-1}, omega]"""
    # Amplitudes in radians (kept below hinge limits), phases 0..2π, small offsets, global freq
    amp_max = np.pi / 2 * 0.8
    a = rng.uniform(0.2 * amp_max, amp_max, size=nu)
    phi = rng.uniform(0.0, 2 * np.pi, size=nu)
    b = rng.uniform(-0.1, 0.1, size=nu)
    omega = rng.uniform(1.5, 4.0, size=1)  # Hz-ish (scaled by Mujoco time)
    return np.concatenate([a, phi, b, omega]).astype(np.float64)


class CPGBrain:
    """Central Pattern Generator controller used via ARIEL's Controller class.

    Params layout (size = 3*nu + 1):
        a[nu]   : amplitudes
        phi[nu] : phases
        b[nu]   : offsets
        omega   : global frequency
    Output is clipped to actuator range [-pi/2, +pi/2].
    """
    def __init__(self, params: npt.NDArray[np.float64]):
        self.params = params

    def forward(self, model: mj.MjModel, data: mj.MjData) -> npt.NDArray[np.float64]:
        nu = model.nu
        a   = self.params[0:nu]
        phi = self.params[nu:2*nu]
        b   = self.params[2*nu:3*nu]
        omega = float(self.params[3*nu])
        t = float(data.time)
        out = a * np.sin(omega * t + phi) + b
        return np.clip(out, -np.pi / 2, np.pi / 2).astype(np.float64)
# ======================================================================= #

def fitness_from_history_time_or_distance(
    xpos_history: list[list[float]],
    dt: float,
    target_x: float,
) -> tuple[float, bool, float | None]:
    """Return (fitness, reached, t_hit).
    If the finish line is crossed, fitness = 1000 - time_to_finish (larger is better).
    Otherwise fitness = -distance_to_goal (negative)."""
    xs = np.array([p[0] for p in xpos_history], dtype=np.float64)
    reach_mask = xs >= target_x
    if np.any(reach_mask):
        hit_idx = int(np.argmax(reach_mask))
        t_hit = hit_idx * dt
        return 1000.0 - t_hit, True, t_hit
    else:
        dist = float(target_x - xs[-1])
        return -dist, False, None
    
def training_fitness_from_history(
    xpos_history: list[list[float]],
    dt: float,
    target_x: float,
    lane_half_width: float = 0.30,   # tune if your robot tends to drift
    z_floor: float = 0.02,           # treat as “fell” if COM/core goes below this
) -> float:
    """
    Training-only shaping:
      + base term: time-to-goal (or -distance)
      + penalties: lateral drift (|y|), off-lane, and falls (z too low)
    """
    arr = np.asarray(xpos_history, dtype=np.float64)  # shape [T, 3]
    _, ys, zs = arr[:, 0], arr[:, 1], arr[:, 2]

    base_fit, _, _ = fitness_from_history_time_or_distance(xpos_history, dt, target_x)

    # keep center: average |y|
    center_pen = -0.5 * float(np.mean(np.abs(ys)))     # small, continuous

    # off-lane hard penalty (if we drift outside lane bounds at any time)
    off_lane = float(np.max(np.abs(ys)) > lane_half_width)
    off_lane_pen = -2.0 * off_lane

    # fall penalty (dips too low)
    fell = float(np.min(zs) < z_floor)
    fall_pen = -3.0 * fell

    # very small smoothness term (optional)
    # smooth_pen = -0.01 * float(np.mean(np.abs(np.diff(xs))))

    shaped = base_fit + center_pen + off_lane_pen + fall_pen  # + smooth_pen
    return shaped

    
def rollout_with_cpg(
    core: Any,
    cpg_params: npt.NDArray[np.float64] | None = None,
    duration: int = 15,
    spawn_pos: list[float] = SPAWN_POS,
) -> tuple[float, bool, float | None, list[list[float]]]:
    """Headless rollout in OlympicArena with a CPG brain; returns (fitness, reached, t_hit, xpos_history)."""
    # Reset callback, build world and spawn
    mj.set_mjcb_control(None)
    world = OlympicArena()
    world.spawn(core.spec, spawn_position=spawn_pos)

    # Compile and reset
    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)

    # Tracker on the "core" geom (matches your template)
    tracker = Tracker(mj.mjtObj.mjOBJ_GEOM, "core")
    tracker.setup(world.spec, data)

    # CPG params (sample if none given)
    if cpg_params is None:
        cpg_params = _sample_cpg_params(model.nu, RNG)

    # Controller wrapper
    brain = CPGBrain(cpg_params)
    ctrl = Controller(controller_callback_function=brain.forward, tracker=tracker)

    # Install controller callback and run headless
    mj.set_mjcb_control(lambda m, d: ctrl.set_control(m, d))
    simple_runner(model, data, duration=duration)

    # Compute fitness from tracker history
    history = tracker.history["xpos"][0]
    dt = float(model.opt.timestep)
    fitness, reached, t_hit = fitness_from_history_time_or_distance(
        history, dt, TARGET_POSITION[0]
    )
    return fitness, reached, t_hit, history

# Put this above main(), near rollout helpers
from typing import NamedTuple

class SimSession(NamedTuple):
    world: OlympicArena
    model: mj.MjModel
    data: mj.MjData

def make_session(core: Any, spawn_pos: list[float] = SPAWN_POS) -> SimSession:
    """Build a world once, spawn the body once, compile once; reuse thereafter."""
    mj.set_mjcb_control(None)
    world = OlympicArena()
    world.spawn(core.spec, spawn_position=spawn_pos)
    model = world.spec.compile()
    data = mj.MjData(model)
    return SimSession(world=world, model=model, data=data)

def rollout_cpg_on_session(
    session: SimSession,
    cpg_params: npt.NDArray[np.float64],
    duration: int,
    spawn_pos: list[float] | None = None,   # NEW
) -> tuple[float, bool, float | None, list[list[float]]]:
    """Run a headless episode using an existing compiled model/data."""
    mj.set_mjcb_control(None)
    model, data, world = session.model, session.data, session.world

    # Reset state & time
    mj.mj_resetData(model, data)

    # --- NEW: teleport to requested spawn (no respawn/rebuild) ---
    if spawn_pos is not None:
        # Assume root free joint at qpos[0:7] => [x,y,z, qw,qx,qy,qz]
        data.qpos[0:3] = np.array(spawn_pos, dtype=np.float64)
        data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)  # identity quat
        data.qvel[:] = 0.0
        mj.mj_forward(model, data)

    # New tracker for a fresh history
    tracker = Tracker(mj.mjtObj.mjOBJ_GEOM, "core")
    tracker.setup(world.spec, data)

    # Controller
    brain = CPGBrain(cpg_params)
    ctrl = Controller(controller_callback_function=brain.forward, tracker=tracker)
    mj.set_mjcb_control(lambda m, d: ctrl.set_control(m, d))

    # Headless run
    simple_runner(model, data, duration=duration)

    # Fitness
    history = tracker.history["xpos"][0]
    dt = float(model.opt.timestep)
    fitness, reached, t_hit = fitness_from_history_time_or_distance(
        history, dt, TARGET_POSITION[0]
    )
    return fitness, reached, t_hit, history

def evaluate_cpg_multi_spawn(
    session: SimSession,
    params: npt.NDArray[np.float64],
    duration: int,
    spawns: list[list[float]] = SPAWN_LIST,
) -> tuple[float, list[list[float]], float]:
    fits, histories, xends = [], [], []
    best_idx, best_fit = 0, -np.inf
    for i, s in enumerate(spawns):
        # Run one episode from a given spawn
        _, _, _, h = rollout_cpg_on_session(session, params, duration=duration, spawn_pos=s)

        # Replace raw fitness with shaped fitness during training
        dt = float(session.model.opt.timestep)
        f = training_fitness_from_history(h, dt, TARGET_POSITION[0])

        fits.append(f)
        histories.append(h)
        xends.append(h[-1][0])
        if f > best_fit:
            best_fit, best_idx = f, i

    agg_fit = float(np.mean(fits))
    best_hist = histories[best_idx]
    best_x_end = float(np.max(xends))
    return agg_fit, best_hist, best_x_end


def play_viewer_on_session(session: SimSession, cpg_params: npt.NDArray[np.float64]) -> None:
    """Open MuJoCo viewer for the current body using given params (no respawn)."""
    mj.set_mjcb_control(None)
    model, data, world = session.model, session.data, session.world
    mj.mj_resetData(model, data)

    tracker = Tracker(mj.mjtObj.mjOBJ_GEOM, "core")
    tracker.setup(world.spec, data)

    brain = CPGBrain(cpg_params)
    ctrl = Controller(controller_callback_function=brain.forward, tracker=tracker)
    mj.set_mjcb_control(lambda m, d: ctrl.set_control(m, d))

    viewer.launch(model=model, data=data)  # closes when you close the window


def record_video_on_session(
    session: SimSession,
    cpg_params: npt.NDArray[np.float64],
    duration: int = 20,
    out_dir: Path = DATA / "videos",
) -> Path:
    """Record a video for the current body/params (headless, no respawn)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    mj.set_mjcb_control(None)
    model, data, world = session.model, session.data, session.world
    mj.mj_resetData(model, data)

    tracker = Tracker(mj.mjtObj.mjOBJ_GEOM, "core")
    tracker.setup(world.spec, data)

    brain = CPGBrain(cpg_params)
    ctrl = Controller(controller_callback_function=brain.forward, tracker=tracker)
    mj.set_mjcb_control(lambda m, d: ctrl.set_control(m, d))

    video_recorder = VideoRecorder(output_folder=str(out_dir))
    video_renderer(model, data, duration=duration, video_recorder=video_recorder)

    # VideoRecorder names files automatically; return folder path
    return out_dir




# ---------- Duration schedule (tune thresholds if needed) ----------
# We grow episode length only when we see farther progress along +x.
CHECKPOINT1_X = -0.2    # ~got off the spawn area (spawn is -0.8)
CHECKPOINT2_X =  2.0    # roughly beyond rugged region (tune)
FINISH_X      =  TARGET_POSITION[0]  # 5.0 in your template

def duration_schedule(best_x: float) -> int:
    if best_x < CHECKPOINT1_X:
        return 20
    if best_x < CHECKPOINT2_X:
        return 40
    return 120
# ------------------------------------------------------------------ #

# ---------- CPG param bounds & clipping ----------
def _cpg_bounds(nu: int):
    amp_max = np.pi / 2 * 0.9
    return {
        "a":   (np.zeros(nu),            np.full(nu, amp_max)),
        "phi": (np.zeros(nu),            np.full(nu, 2 * np.pi)),
        "b":   (np.full(nu, -np.pi/4),   np.full(nu,  np.pi/4)),
        "omega": (np.array([0.5]),       np.array([6.0])),
    }

def _clip_cpg_params(theta: npt.NDArray[np.float64], nu: int) -> npt.NDArray[np.float64]:
    a_lo, a_hi   = _cpg_bounds(nu)["a"]
    # p_lo, p_hi   = _cpg_bounds(nu)["phi"]
    b_lo, b_hi   = _cpg_bounds(nu)["b"]
    w_lo, w_hi   = _cpg_bounds(nu)["omega"]
    out = theta.copy()
    out[0:nu]          = np.clip(out[0:nu],          a_lo, a_hi)
    out[nu:2*nu]       = np.mod(out[nu:2*nu],        2*np.pi)  # wrap phases
    out[2*nu:3*nu]     = np.clip(out[2*nu:3*nu],     b_lo, b_hi)
    out[3*nu:3*nu+1]   = np.clip(out[3*nu:3*nu+1],   w_lo, w_hi)
    return out
# -------------------------------------------------

# ---------- (1+λ)-ES trainer for CPG ----------
def es_train_cpg_on_body(
    core: Any,              # a single, already-decoded body (e.g., Gecko now; NDE later)
    iters: int = 25,
    lam: int = 12,
    sigma: float = 0.25,
    seed: int = SEED,
) -> tuple[npt.NDArray[np.float64], float, list[list[float]], SimSession]:
    rng = np.random.default_rng(seed)

    # Build one session and reuse it for all evaluations
    session = make_session(core)
    nu = session.model.nu

    parent = _clip_cpg_params(_sample_cpg_params(nu, rng), nu)

    best_fit, best_hist, best_x = evaluate_cpg_multi_spawn(session, parent, duration=15)
    best_x = float(best_hist[-1][0])
    success = 0

    for g in range(1, iters + 1):
        dur = duration_schedule(best_x)

        fit_p, hist_p, best_x_p = evaluate_cpg_multi_spawn(session, parent, duration=dur)

        cand_best_fit, cand_best_theta, cand_best_hist = fit_p, parent, hist_p

        for _ in range(lam):
            child = _clip_cpg_params(parent + sigma * rng.normal(size=parent.shape), nu)
            fit_c, hist_c, _ = evaluate_cpg_multi_spawn(session, child, duration=dur)
            if fit_c > cand_best_fit:
                cand_best_fit, cand_best_theta, cand_best_hist = fit_c, child, hist_c

        if cand_best_fit > fit_p:
            parent, fit_p, hist_p = cand_best_theta, cand_best_fit, cand_best_hist
            success += 1

        if fit_p > best_fit:
            best_fit, best_hist = fit_p, hist_p

        best_x = max(best_x, best_x_p)

        if g % 5 == 0:
            rate = success / 5.0
            sigma = sigma * 1.2 if rate > 0.2 else sigma / 1.2 if rate < 0.2 else sigma
            success = 0

        console.log(
            f"[ES g{g}/{iters}] dur={dur}s sigma={sigma:.3f} "
            f"fit_parent={fit_p:.3f} best_global={best_fit:.3f} best_x={best_x:.3f}"
        )

    return parent, best_fit, best_hist, session

# -------------------------------------------------


# ---------- CMA-ES trainer for CPG ----------
def cma_train_cpg_on_body(
    core: Any,
    iters: int = 30,           # CMA generations (tell/ask steps)
    sigma0: float = 0.30,      # initial step-size (like your ES sigma)
    seed: int = SEED,
    popsize: int | None = None # None = CMA default; or set (e.g., 16, 24)
) -> tuple[npt.NDArray[np.float64], float, list[list[float]], SimSession]:
    """
    Train CPG params with CMA-ES on a *fixed* body (single compiled session).
    Returns: (best_params, best_fit, best_hist, session)
    """
    # Build one reusable session
    session = make_session(core)
    nu = session.model.nu
    dim = 3 * nu + 1  # [a(ν), phi(ν), b(ν), omega(1)] – matches your current brain

    # Bounds (match your _cpg_bounds)
    amp_max = np.pi / 2 * 0.9
    lo = np.concatenate([
        np.zeros(nu),                      # a
        np.zeros(nu),                      # phi
        np.full(nu, -np.pi/4),            # b
        np.array([0.5]),                  # omega
    ])
    hi = np.concatenate([
        np.full(nu, amp_max),             # a
        np.full(nu, 2*np.pi),             # phi
        np.full(nu,  np.pi/4),            # b
        np.array([6.0]),                  # omega
    ])

    # Initial point
    x0 = _clip_cpg_params(_sample_cpg_params(nu, RNG), nu)

    # CMA options
    opts: dict[str, Any] = {
        "seed": seed,
        "bounds": [lo, hi],           # per-dimension bounds
        "popsize": popsize,           # can be None      -> CMA default
        "CMA_stds": (hi - lo) * 0.2,  # exploration scale per dim (tweak)
        "verb_disp": 0,               # silence CMA’s own prints (we log ourselves)
    }

    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    best_fit = -np.inf
    best_theta = x0.copy()
    best_hist: list[list[float]] = []
    best_x_seen = -np.inf  # for curriculum duration

    for g in range(1, iters + 1):
        # Curriculum duration based on progress so far
        dur = duration_schedule(best_x_seen)

        # Ask for a population
        X = es.ask()  # list of candidate vectors
        fits = []
        gen_best_fit = -np.inf
        gen_best_hist: list[list[float]] = []
        gen_best_x_end = -np.inf

        for x in X:
            theta = _clip_cpg_params(np.asarray(x, dtype=np.float64), nu)
            f, hist, x_end = evaluate_cpg_multi_spawn(session, theta, duration=dur)
            # CMA minimizes -> use negative of fitness
            fits.append(-f)

            if f > gen_best_fit:
                gen_best_fit = f
                gen_best_hist = hist
                gen_best_x_end = x_end

        # Tell CMA the evaluated losses
        es.tell(X, fits)
        es.disp()  # optional: let CMA print a one-liner

        # Track global best
        if gen_best_fit > best_fit:
            best_fit = gen_best_fit
            best_theta = _clip_cpg_params(np.asarray(es.result.xbest, dtype=np.float64), nu)
            best_hist = gen_best_hist

        # Update curriculum progress
        best_x_seen = max(best_x_seen, gen_best_x_end)

        console.log(
            f"[CMA g{g}/{iters}] dur={dur}s pop={len(X)} "
            f"gen_best={gen_best_fit:.3f} best_global={best_fit:.3f} best_x={best_x_seen:.3f}"
        )

        if es.stop():
            console.log(f"[CMA] stop condition: {es.stop()}")
            break

    # After loop, es.result.xbest is CMA’s best guess; keep our clipped variant
    return best_theta, best_fit, best_hist, session
# ------------------------------------------------- #


def experiment(
    robot: Any,
    controller: Controller,
    duration: int = 15,
    mode: ViewerTypes = "viewer",
) -> None:
    """Run the simulation with random movements."""
    # ==================================================================== #
    # Initialise controller to controller to None, always in the beginning.
    mj.set_mjcb_control(None)  # DO NOT REMOVE

    # Initialise world
    # Import environments from ariel.simulation.environments
    world = OlympicArena()

    # Spawn robot in the world
    # Check docstring for spawn conditions
    world.spawn(robot.spec, spawn_position=SPAWN_POS)

    # Generate the model and data
    # These are standard parts of the simulation USE THEM AS IS, DO NOT CHANGE
    model = world.spec.compile()
    data = mj.MjData(model)

    # Reset state and time of simulation
    mj.mj_resetData(model, data)

    # Pass the model and data to the tracker
    if controller.tracker is not None:
        controller.tracker.setup(world.spec, data)

    # Set the control callback function
    # This is called every time step to get the next action.
    args: list[Any] = []  # IF YOU NEED MORE ARGUMENTS ADD THEM HERE!
    kwargs: dict[Any, Any] = {}  # IF YOU NEED MORE ARGUMENTS ADD THEM HERE!

    mj.set_mjcb_control(
        lambda m, d: controller.set_control(m, d, *args, **kwargs),
    )

    # ------------------------------------------------------------------ #
    match mode:
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
    # ==================================================================== #


def main() -> None:
    from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
    """Entry point."""
    # # ? ------------------------------------------------------------------ #
    # genotype_size = 64
    # type_p_genes = RNG.random(genotype_size).astype(np.float32)
    # conn_p_genes = RNG.random(genotype_size).astype(np.float32)
    # rot_p_genes = RNG.random(genotype_size).astype(np.float32)

    # genotype = [
    #     type_p_genes,
    #     conn_p_genes,
    #     rot_p_genes,
    # ]

    # nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES)
    # p_matrices = nde.forward(genotype)

    # # Decode the high-probability graph
    # hpd = HighProbabilityDecoder(NUM_OF_MODULES)
    # robot_graph: DiGraph[Any] = hpd.probability_matrices_to_graph(
    #     p_matrices[0],
    #     p_matrices[1],
    #     p_matrices[2],
    # )

    # # ? ------------------------------------------------------------------ #
    # # Save the graph to a file
    # save_graph_as_json(
    #     robot_graph,
    #     DATA / "robot_graph.json",
    # )

    # # ? ------------------------------------------------------------------ #
    # Print all nodes
    # core = construct_mjspec_from_graph(robot_graph)
    core = gecko()

    # # ? ------------------------------------------------------------------ #
    # mujoco_type_to_find = mj.mjtObj.mjOBJ_GEOM
    # name_to_bind = "core"
    # tracker = Tracker(
    #     mujoco_obj_to_find=mujoco_type_to_find,
    #     name_to_bind=name_to_bind,
    # )
    # # ? ------------------------------------------------------------------ #
    # # Simulate the robot
    # ctrl = Controller(
    #     controller_callback_function=nn_controller,
    #     # controller_callback_function=random_move,
    #     tracker=tracker,
    # )

    # experiment(robot=core, controller=ctrl, mode="launcher")

    # show_xpos_history(tracker.history["xpos"][0])

    # fitness = fitness_function(tracker.history["xpos"][0])
    # msg = f"Fitness of generated robot: {fitness}"
    # console.log(msg)

    # # Train the CPG params on Gecko
    # best_params, _best_fit, _best_hist, session = es_train_cpg_on_body(
    #     core,
    #     iters=30,
    #     lam=24,
    #     sigma=0.30,
    #     seed=SEED,
    # )

    # ? ------------------------------------------------------------------ #
    best_params, _best_fit, _best_hist, session = cma_train_cpg_on_body(
    core,
    iters=40,       # try 30–80; more if budget allows
    sigma0=0.30,    # like your ES sigma
    seed=SEED,
    popsize=16      # try 12–32; CMA default also fine
)
    # ? ------------------------------------------------------------------ #


    # Final eval on the SAME session/body (no rebuild)
    final_fit, reached, t_hit, history = rollout_cpg_on_session(session, best_params, duration=120)

    np.save(DATA / "gecko_cpg_best.npy", best_params)
    show_xpos_history(history)

    # See it live (close window to continue)
    play_viewer_on_session(session, best_params)

    # Save a short video (headless)
    video_folder = record_video_on_session(session, best_params, duration=60)
    console.log(f"Saved video to: {video_folder}")

    # e.g., record short clips at each spawn
    # for s in SPAWN_LIST:
    #     record_video_on_session(session, best_params, duration=30)
    # #  or
    # # play_viewer_on_session(session, best_params)  # then hit ESC to quit


    console.log(
        f"[CPG final] {'Reached' if reached else 'Not reached'} "
        f"{'(t_hit=%.2fs)' % t_hit if reached else ''} fitness={final_fit:.2f} "
        f"last_x={history[-1][0]:.3f}"
    )
if __name__ == "__main__":
    main()
