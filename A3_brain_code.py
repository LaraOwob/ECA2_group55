"""Assignment 3 template code."""
from email import policy

from fastapi import params
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko

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

# --- Multi-spawn presets (approximate waypoints inside OlympicArena) ---
SPAWN_FLAT   = [-0.8, 0, 0.1]  # start
SPAWN_RUGGED = [ 1.6, 0, 0.1]  # before/inside rugged
SPAWN_UPHILL = [ 3.8, 0, 0.1]  # before slope
LEARN_SPAWNS = [SPAWN_FLAT, SPAWN_RUGGED, SPAWN_UPHILL]

def inner_learn_controller(
    base_params: npt.NDArray[np.float64],
    policy: Literal["mlp","cpg"],
    learn_steps: int = 2,
    learn_sigma: float = 0.02,
    learn_duration: float = 8.0,
    spawns: list[list[float]] = LEARN_SPAWNS,
    seed: int = 0,
) -> npt.NDArray[np.float64]:
    """
    Simple accept-if-better hill climber on controller params.
    Trains over short rollouts while cycling spawn positions.
    Returns the improved params (used only for evaluation unless you toggle Lamarckian).
    """
    best = base_params.copy()
    best_fit, _ = evaluate_brain_on_gecko_with_params(
        best, policy=policy, duration=learn_duration, spawn_pos=spawns[0], seeds=(seed,)
    )
    if learn_steps <= 0:
        return best

    for i in range(learn_steps):
        spawn = spawns[(i+1) % len(spawns)]  # cycle through terrains
        cand = best + RNG.normal(0, learn_sigma, size=best.size)
        # keep CPG params valid
        if policy == "cpg":
            # need model.nu to clamp slices; infer from length
            n = len(best) // 4
            A_min, A_max = 0.1*np.pi, 0.7*np.pi
            w_min, w_max = 0.2, 4.0
            b_min, b_max = -0.5*np.pi, 0.5*np.pi
            cand[0*n:1*n] = np.clip(cand[0*n:1*n], A_min, A_max)
            cand[1*n:2*n] = np.clip(cand[1*n:2*n], w_min, w_max)
            cand[2*n:3*n] = np.mod(cand[2*n:3*n], 2*np.pi)
            cand[3*n:4*n] = np.clip(cand[3*n:4*n], b_min, b_max)
        else:
            cand = np.clip(cand, -3.0, 3.0)

        fit, _ = evaluate_brain_on_gecko_with_params(
            cand, policy=policy, duration=learn_duration, spawn_pos=spawn, seeds=(seed,)
        )
        if fit > best_fit:
            best, best_fit = cand, fit
    return best


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

# === Controllers with flat-parameter APIs step2.1 =====================================
class MLPPolicy:
    """
    Stateless MLP controller with tanh activations.
    Flat params layout: [W1 (nq*h), W2 (h*h), W3 (h*nu)]
    """
    def __init__(self, model: mj.MjModel, rng: np.random.Generator, hidden: int = 32):
        self.in_dim = model.nq
        self.h = hidden
        self.out_dim = model.nu
        # random init (will be overwritten by EA via set_flat_params)
        self.W1 = rng.normal(0, 0.2, size=(self.in_dim, self.h))
        self.W2 = rng.normal(0, 0.2, size=(self.h, self.h))
        self.W3 = rng.normal(0, 0.2, size=(self.h, self.out_dim))

    @property
    def flat_size(self) -> int:
        return self.in_dim*self.h + self.h*self.h + self.h*self.out_dim

    def set_flat_params(self, w: npt.NDArray[np.float64]) -> None:
        assert w.size == self.flat_size
        i = 0
        n = self.in_dim*self.h
        self.W1 = w[i:i+n].reshape(self.in_dim, self.h); i += n
        n = self.h*self.h
        self.W2 = w[i:i+n].reshape(self.h, self.h); i += n
        n = self.h*self.out_dim
        self.W3 = w[i:i+n].reshape(self.h, self.out_dim)

    def __call__(self, model: mj.MjModel, data: mj.MjData) -> npt.NDArray[np.float64]:
        x = np.asarray(data.qpos)  # shape [nq]
        h1 = np.tanh(x @ self.W1)
        h2 = np.tanh(h1 @ self.W2)
        y  = np.tanh(h2 @ self.W3)  # [-1,1]
        return y * np.pi            # [-π, π]

class CPGPolicyPaired:
    """
    Symmetric paired CPG with small proprioceptive feedback.
    We assume 4 leg pairs (ν must be even; Gecko has 8).
      For each pair k:
        u_left  = A_k * sin(ω * t + φ_k) + b_k + g * qvel_left
        u_right = A_k * sin(ω * t + φ_k + π) + b_k + g * qvel_right
    Output clamped to MuJoCo hinge range [-π/2, π/2].
    Flat params: [A(4), φ(4), b(4), ω(1), g(1)]  -> size = 14
    """
    def __init__(self, model: mj.MjModel, rng: np.random.Generator):
        assert model.nu % 2 == 0, "paired CPG expects even number of actuators"
        self.nu = model.nu
        self.n_pairs = self.nu // 2

        # conservative ranges (radians & rad/s)
        self.A_min, self.A_max = 0.1, 0.6      # much smaller than π
        self.b_min, self.b_max = -0.25, 0.25
        self.w_min, self.w_max = 0.5, 3.0
        self.g_min, self.g_max = 0.0, 0.4      # feedback gain

        # init (overwritten by set_flat_params)
        self.A  = rng.uniform(self.A_min, self.A_max, size=self.n_pairs)
        self.ph = rng.uniform(0.0, 2*np.pi, size=self.n_pairs)
        self.b  = rng.uniform(self.b_min, self.b_max, size=self.n_pairs)
        self.w  = rng.uniform(self.w_min, self.w_max)   # shared frequency
        self.g  = rng.uniform(self.g_min, self.g_max)   # shared feedback

    @property
    def flat_size(self) -> int:
        return 3*self.n_pairs + 2  # A(4)+φ(4)+b(4)+ω+g = 14 for ν=8

    def set_flat_params(self, w: npt.NDArray[np.float64]) -> None:
        assert w.size == self.flat_size
        n = self.n_pairs
        self.A  = np.clip(w[0*n:1*n], self.A_min, self.A_max)
        self.ph = np.mod(w[1*n:2*n], 2*np.pi)
        self.b  = np.clip(w[2*n:3*n], self.b_min, self.b_max)
        self.w  = np.clip(w[3*n], self.w_min, self.w_max)
        self.g  = np.clip(w[3*n+1], self.g_min, self.g_max)

    def __call__(self, model: mj.MjModel, data: mj.MjData) -> npt.NDArray[np.float64]:
        t = data.time
        out = np.zeros(self.nu, dtype=np.float64)

        # Use joint velocities as a simple feedback signal
        # Bind qvel for actuated DOFs only (approximate: take first ν entries)
        qv = np.asarray(data.qvel)[: self.nu]

        # Build paired outputs
        for k in range(self.n_pairs):
            base = self.A[k] * np.sin(self.w * t + self.ph[k]) + self.b[k]
            left  =  base + self.g * qv[2*k]
            right = -base + self.g * qv[2*k + 1]   # π phase shift == sign flip
            out[2*k]     = left
            out[2*k + 1] = right

        # Clamp to hinge limits (safer than ±π)
        return np.clip(out, -np.pi/2, np.pi/2)

class CPGPolicy:
    """
    Uncoupled sinusoidal CPG per actuator:
      u_j(t) = A_j * sin(ω_j * t + φ_j) + b_j
    Flat params layout: [A(ν), ω(ν), φ(ν), b(ν)]
    """
    def __init__(self, model: mj.MjModel, rng: np.random.Generator):
        self.nu = model.nu
        # parameter ranges
        self.A_min, self.A_max   = 0.1*np.pi, 0.7*np.pi
        self.w_min, self.w_max   = 0.2, 4.0      # rad/s
        self.b_min, self.b_max   = -0.5*np.pi, 0.5*np.pi
        # init (overwritten by EA)
        self.A   = rng.uniform(self.A_min, self.A_max, size=self.nu)
        self.omega = rng.uniform(self.w_min, self.w_max, size=self.nu)
        self.phi = rng.uniform(0.0, 2*np.pi, size=self.nu)
        self.bias = rng.uniform(self.b_min, self.b_max, size=self.nu)

    @property
    def flat_size(self) -> int:
        return 4*self.nu

    def set_flat_params(self, w: npt.NDArray[np.float64]) -> None:
        assert w.size == self.flat_size
        w = w.astype(float, copy=False)
        n = self.nu
        self.A     = np.clip(w[0*n:1*n], self.A_min, self.A_max)
        self.omega = np.clip(w[1*n:2*n], self.w_min, self.w_max)
        self.phi   = np.mod(w[2*n:3*n], 2*np.pi)
        self.bias  = np.clip(w[3*n:4*n], self.b_min, self.b_max)

    def __call__(self, model: mj.MjModel, data: mj.MjData) -> npt.NDArray[np.float64]:
        t = data.time
        u = self.A * np.sin(self.omega * t + self.phi) + self.bias
        return np.clip(u, -np.pi, np.pi)
    
#step2.2
def evaluate_brain_on_gecko_with_params(
    params: npt.NDArray[np.float64],
    policy: Literal["mlp", "cpg"] = "mlp",
    duration: float = 15.0,
    spawn_pos: list[float] = SPAWN_POS,
    seeds: tuple[int, ...] = (0, 1),
    record_video: bool = False,
) -> tuple[float, list[list[float]]]:
    """
    Evaluate a *given* controller parameter vector on Gecko in OlympicArena.
    Returns (mean_fitness_over_seeds, last_history_from_last_seed).
    """
    if policy == "mlp":
        brain = MLPPolicy(model, rng_local, hidden=32)
    elif policy == "cpg":
        brain = CPGPolicy(model, rng_local)
    else:
        brain = CPGPolicyPaired(model, rng_local)
    brain.set_flat_params(params)

    last_history: list[list[float]] = []

    fitnesses: list[float] = []
    for seed in seeds:
        rng_local = np.random.default_rng(SEED + seed)
        mj.set_mjcb_control(None)

        # World + fixed body (Gecko)
        world = OlympicArena()
        core = gecko()
        world.spawn(core.spec, spawn_position=spawn_pos)

        # Model/Data/Tracker
        model = world.spec.compile()
        data = mj.MjData(model)
        tracker = Tracker(mj.mjtObj.mjOBJ_GEOM, "core")

        # Build policy and load params
        if policy == "mlp":
            brain = MLPPolicy(model, rng_local, hidden=32)
        else:
            brain = CPGPolicy(model, rng_local)
        brain.set_flat_params(params)

        ctrl = Controller(controller_callback_function=brain, tracker=tracker)

        # Reset, setup tracker, set callback
        mj.mj_resetData(model, data)
        tracker.setup(world.spec, data)
        mj.set_mjcb_control(lambda m, d: ctrl.set_control(m, d))

        # Headless run (or video)
        if record_video and seed == seeds[-1]:
            path_to_video_folder = str(DATA / "videos")
            video_recorder = VideoRecorder(output_folder=path_to_video_folder)
            video_renderer(model, data, duration=duration, video_recorder=video_recorder)
        else:
            simple_runner(model, data, duration=duration)

        history = tracker.history["xpos"][0]
        fit = fitness_function(history)
        fitnesses.append(float(fit))
        last_history = history

    return float(np.mean(fitnesses)), 

def probe_dims_for_policy(policy: Literal["mlp","cpg","cpg_paired"] = "mlp", hidden: int = 32) -> int:
    world = OlympicArena(); core = gecko(); world.spawn(core.spec, spawn_position=SPAWN_POS)
    model = world.spec.compile()
    rng_local = np.random.default_rng(SEED+999)
    if policy == "mlp":
        return MLPPolicy(model, rng_local, hidden=hidden).flat_size
    if policy == "cpg":
        return CPGPolicy(model, rng_local).flat_size
    return CPGPolicyPaired(model, rng_local).flat_size


# === EA helpers step 2.3 ================================================================

def probe_dims_for_policy(policy: Literal["mlp", "cpg"] = "mlp", hidden: int = 32) -> int:
    """Build a probe model to determine flat parameter size for the chosen policy."""
    world = OlympicArena()
    core = gecko()
    world.spawn(core.spec, spawn_position=SPAWN_POS)
    model = world.spec.compile()
    rng_local = np.random.default_rng(SEED+999)
    if policy == "mlp":
        brain = MLPPolicy(model, rng_local, hidden=hidden)
    else:
        brain = CPGPolicy(model, rng_local)
    return brain.flat_size

def init_param_vector(
    size: int,
    policy: Literal["mlp","cpg"],
    model_nu: int | None = None,
) -> npt.NDArray[np.float64]:
    if policy == "mlp":
        # Small Gaussian around 0
        return RNG.normal(0, 0.2, size=size).astype(np.float64)

    # CPG: sample directly using ranges, no MuJoCo calls
    assert model_nu is not None, "model_nu required for CPG init"
    n = model_nu
    A_min, A_max = 0.1*np.pi, 0.7*np.pi
    w_min, w_max = 0.2, 4.0
    b_min, b_max = -0.5*np.pi, 0.5*np.pi

    A   = RNG.uniform(A_min, A_max, size=n)
    w   = RNG.uniform(w_min, w_max, size=n)
    phi = RNG.uniform(0.0, 2*np.pi, size=n)
    b   = RNG.uniform(b_min, b_max, size=n)
    return np.concatenate([A, w, phi, b]).astype(np.float64)


def uniform_crossover(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64], p: float = 0.5):
    mask = RNG.random(a.size) < p
    child = np.where(mask, a, b).astype(np.float64, copy=False)
    return child

def mutate(vec: npt.NDArray[np.float64], sigma: float, policy: Literal["mlp","cpg"], model_nu: int | None = None) -> npt.NDArray[np.float64]:
    out = vec + RNG.normal(0, sigma, size=vec.size)
    if policy == "cpg":
        assert model_nu is not None
        n = model_nu
        # ranges aligned with CPGPolicy
        A_min, A_max = 0.1*np.pi, 0.7*np.pi
        w_min, w_max = 0.2, 4.0
        b_min, b_max = -0.5*np.pi, 0.5*np.pi
        out[0*n:1*n] = np.clip(out[0*n:1*n], A_min, A_max)
        out[1*n:2*n] = np.clip(out[1*n:2*n], w_min, w_max)
        out[2*n:3*n] = np.mod(out[2*n:3*n], 2*np.pi)
        out[3*n:4*n] = np.clip(out[3*n:4*n], b_min, b_max)
    else:
        out = np.clip(out, -3.0, 3.0)  # keep MLP weights sane
    return out

def tournament_select(pop, k: int = 3):
    idxs = RNG.integers(0, len(pop), size=k)
    best = max((pop[i] for i in idxs), key=lambda ind: ind["fitness"])
    return best

def duration_from_best(best_fitness: float) -> float:
    """
    fitness = -distance_to_goal, so distance = -fitness
    Use rough checkpoints to scale episode length.
    """
    dist = -best_fitness
    if dist > 3.0:   return 15.0   # hasn’t reached rugged yet
    if dist > 1.0:   return 45.0   # around/after rugged part
    return 100.0                   # near uphill/finish

#step 3.3
def evolve_brain_only(
    policy: Literal["mlp","cpg"] = "cpg",
    pop_size: int = 24,
    gens: int = 20,
    sigma_mut: float = 0.05,
    crossover_p: float = 0.5,
    seeds: tuple[int, ...] = (0,1),
    # --- lifetime learning knobs ---
    learn_steps: int = 2,
    learn_sigma: float = 0.02,
    learn_duration: float = 8.0,
    lamarckian: bool = False,
) -> dict[str, Any]:
    """
    Evolves controller parameters on the fixed Gecko body in OlympicArena.
    Returns the best individual dict: {"params", "fitness"}.
    """
    # Probe once
    world = OlympicArena()
    core = gecko()
    world.spawn(core.spec, spawn_position=SPAWN_POS)
    probe_model = world.spec.compile()
    model_nu = probe_model.nu
    flat_size = probe_dims_for_policy(policy)

    # --- Initialize population (with inner learning) ---
    population: list[dict[str, Any]] = []
    for _ in range(pop_size):
        p0 = init_param_vector(flat_size, policy, model_nu=model_nu)
        # lifetime learning on copied params
        p_eval = inner_learn_controller(
            p0, policy=policy, learn_steps=learn_steps, learn_sigma=learn_sigma, learn_duration=learn_duration
        )
        # robust score from true start (averaged seeds)
        f, _ = evaluate_brain_on_gecko_with_params(
            p_eval, policy=policy, duration=15.0, seeds=seeds, record_video=False
        )
        # offspring inheritance (Darwinian vs Lamarckian)
        p_inherit = p_eval if lamarckian else p0
        population.append({"params": p_inherit, "fitness": f})

    best = max(population, key=lambda ind: ind["fitness"])
    console.log(f"Gen 0: best={best['fitness']:.4f} mean={np.mean([i['fitness'] for i in population]):.4f}")

    # --- Evolution loop ---
    for g in range(1, gens+1):
        duration = duration_from_best(best["fitness"])
        mu = max(1, pop_size // 5)
        elites = sorted(population, key=lambda ind: ind["fitness"], reverse=True)[:mu]

        offspring: list[dict[str, Any]] = []
        while len(offspring) < pop_size - mu:
            p1 = tournament_select(population, k=3)
            p2 = tournament_select(population, k=3)
            child = uniform_crossover(p1["params"], p2["params"], p=crossover_p)
            child = mutate(child, sigma_mut, policy, model_nu=model_nu)

            # lifetime learning before scoring
            child_eval = inner_learn_controller(
                child, policy=policy, learn_steps=learn_steps, learn_sigma=learn_sigma, learn_duration=learn_duration
            )
            f, _ = evaluate_brain_on_gecko_with_params(
                child_eval, policy=policy, duration=duration, seeds=seeds, record_video=False
            )
            child_inherit = child_eval if lamarckian else child
            offspring.append({"params": child_inherit, "fitness": f})

        population = elites + offspring
        gen_best = max(population, key=lambda ind: ind["fitness"])
        if gen_best["fitness"] > best["fitness"]:
            best = {"params": gen_best["params"].copy(), "fitness": gen_best["fitness"]}

        console.log(
            f"Gen {g}: best={best['fitness']:.4f} mean={np.mean([i['fitness'] for i in population]):.4f} "
            f"dur={duration:.0f}s learn={learn_steps}x{learn_duration:.0f}s"
        )

        # Checkpoint
        ckpt_path = DATA / f"brain_only_gen{g}.npz"
        np.savez_compressed(
            ckpt_path,
            policy=policy,
            best_fitness=np.array([best["fitness"]]),
            best_params=best["params"],
            pop_fitness=np.array([ind['fitness'] for ind in population]),
            learn_steps=np.array([learn_steps]),
            learn_sigma=np.array([learn_sigma]),
            learn_duration=np.array([learn_duration]),
            lamarckian=np.array([lamarckian]),
        )

    return best

def policy_bounds(policy: Literal["mlp","cpg","cpg_paired"], flat_size: int, model_nu: int):
    if policy == "mlp":
        low  = np.full(flat_size, -3.0); high = np.full(flat_size, 3.0); return low, high
    if policy == "cpg":
        # (your existing bounds for uncoupled cpg)
        n = model_nu
        A_min, A_max = 0.1*np.pi, 0.7*np.pi
        w_min, w_max = 0.2, 4.0
        phi_min, phi_max = 0.0, 2*np.pi
        b_min, b_max = -0.5*np.pi, 0.5*np.pi
        low  = np.concatenate([np.full(n,A_min), np.full(n,w_min), np.full(n,phi_min), np.full(n,b_min)]).astype(np.float64)
        high = np.concatenate([np.full(n,A_max), np.full(n,w_max), np.full(n,phi_max), np.full(n,b_max)]).astype(np.float64)
        return low, high
    # paired
    n = model_nu // 2
    A_min, A_max = 0.1, 0.6
    b_min, b_max = -0.25, 0.25
    w_min, w_max = 0.5, 3.0
    g_min, g_max = 0.0, 0.4
    low  = np.concatenate([np.full(n,A_min), np.full(n,0.0), np.full(n,b_min), np.array([w_min, g_min])]).astype(np.float64)
    high = np.concatenate([np.full(n,A_max), np.full(n,2*np.pi), np.full(n,b_max), np.array([w_max, g_max])]).astype(np.float64)
    return low, high


def evolve_brain_cmaes(
    policy: Literal["mlp","cpg"] = "cpg",
    iters: int = 40,
    seeds: tuple[int, ...] = (0, 1),
    learn_steps: int = 2,
    learn_sigma: float = 0.02,
    learn_duration: float = 8.0,
    lamarckian: bool = False,
    sigma0_mlp: float = 0.3,
    sigma0_cpg: float = 0.1,
    popsize: int | None = None,
):
    import cma  # ensure installed

    # --- probe dims once ---
    world = OlympicArena()
    core = gecko()
    world.spawn(core.spec, spawn_position=SPAWN_POS)
    probe_model = world.spec.compile()
    model_nu = probe_model.nu
    flat_size = probe_dims_for_policy(policy)

    # --- init mean, bounds, sigma ---
    m0 = init_param_vector(flat_size, policy, model_nu=model_nu)
    lb, ub = policy_bounds(policy, flat_size, model_nu)
    sigma0 = sigma0_mlp if policy == "mlp" else sigma0_cpg

    if popsize is None:
        popsize = 4 + int(3 * np.log(flat_size + 1))
    popsize = max(popsize, 4)  # ensure ≥2; 4 is a safe default

    opts = {
        "bounds": [lb.tolist(), ub.tolist()],
        "popsize": popsize,
        "seed": SEED,
        "verbose": -9,
        # Optional: disable mirrors if you ever need to do single-candidate tells
        # "CMA_mirrors": 0,
    }
    es = cma.CMAEvolutionStrategy(m0, sigma0, opts)

    best = {"params": m0.copy(), "fitness": -1e9}
    duration = 15.0

    for g in range(1, iters + 1):
        xs = es.ask()  # list of candidates, len == popsize

        # ---- evaluate the whole batch, once ----
        fitnesses = []
        xs_eval = []
        for x in xs:
            x = np.asarray(x, dtype=np.float64)
            # lifetime learning (evaluate-polish only)
            x_learned = inner_learn_controller(
                x, policy=policy, learn_steps=learn_steps, learn_sigma=learn_sigma, learn_duration=learn_duration
            )
            f, _ = evaluate_brain_on_gecko_with_params(
                x_learned, policy=policy, duration=duration, seeds=seeds, record_video=False
            )
            fitnesses.append(float(f))
            xs_eval.append(x_learned if lamarckian else x)

        # CMA-ES MINIMIZES → pass negative fitness.
        # Note: we pass costs associated with xs. If lamarckian=True, xs_eval contains learned params,
        # but CMA still updates from xs; this mild mismatch is common in practice. For strict Lamarckian,
        # you can set x := x_learned before passing to tell (requires bounds handling).
        costs = [-f for f in fitnesses]
        es.tell(xs, costs)

        # Track best
        idx_best = int(np.argmax(fitnesses))
        cand_best = xs_eval[idx_best]
        fit_best = fitnesses[idx_best]
        if fit_best > best["fitness"]:
            best = {"params": cand_best.copy(), "fitness": fit_best}

        # Log & adapt duration
        mean_fit = float(np.mean(fitnesses))
        duration = duration_from_best(best["fitness"])
        console.log(f"[CMA] iter {g}/{iters}: best={best['fitness']:.4f} mean={mean_fit:.4f} dur={duration:.0f}s")

        # Checkpoint
        ckpt_path = DATA / f"brain_cma_iter{g}.npz"
        np.savez_compressed(
            ckpt_path,
            policy=policy,
            best_fitness=np.array([best["fitness"]]),
            best_params=best["params"],
            pop_fitness=np.array(fitnesses),
            learn_steps=np.array([learn_steps]),
            learn_sigma=np.array([learn_sigma]),
            learn_duration=np.array([learn_duration]),
            lamarckian=np.array([lamarckian]),
        )

        if es.stop():
            break

    return best

def init_param_vector(size: int, policy: Literal["mlp","cpg","cpg_paired"], model_nu: int | None = None):
    if policy == "mlp":
        return RNG.normal(0, 0.2, size=size).astype(np.float64)
    if policy == "cpg":
        assert model_nu is not None
        n = model_nu
        A   = RNG.uniform(0.1*np.pi, 0.7*np.pi, size=n)
        w   = RNG.uniform(0.2, 4.0, size=n)
        phi = RNG.uniform(0.0, 2*np.pi, size=n)
        b   = RNG.uniform(-0.5*np.pi, 0.5*np.pi, size=n)
        return np.concatenate([A, w, phi, b]).astype(np.float64)
    # paired
    assert model_nu is not None and model_nu % 2 == 0
    n = model_nu // 2
    A   = RNG.uniform(0.15, 0.45, size=n)
    ph  = RNG.uniform(0.0, 2*np.pi, size=n)
    b   = RNG.uniform(-0.1, 0.1, size=n)
    w   = np.array([RNG.uniform(0.8, 2.0)])
    g   = np.array([RNG.uniform(0.05, 0.2)])
    return np.concatenate([A, ph, b, w, g]).astype(np.float64)

def shaped_progress_reward(history: list[list[float]]) -> float:
    arr = np.asarray(history)
    dx = arr[-1, 0] - arr[0, 0]
    # encourage early progress with a small weight; keep scale modest
    return 0.2 * dx  # tweak 0.1–0.5

# Inside inner_learn_controller, replace:
#   fit, _ = evaluate_brain_on_gecko_with_params(...)
# with:
fit, hist = evaluate_brain_on_gecko_with_params(
    cand, policy=policy, duration=learn_duration, spawn_pos=spawn, seeds=(seed,)
)
fit += shaped_progress_reward(hist)

# #step 2.4
# def evolve_brain_only(
#     policy: Literal["mlp","cpg"] = "cpg",
#     pop_size: int = 24,
#     gens: int = 20,
#     sigma_mut: float = 0.05,
#     crossover_p: float = 0.5,
#     seeds: tuple[int, ...] = (0,1),
# ) -> dict[str, Any]:
#     """
#     Evolves controller parameters on the fixed Gecko body in OlympicArena.
#     Returns the best individual dict: {"params", "fitness"}.
#     """
#     # Probe dimensions and model.nu for mutation constraints (CPG)
#     world = OlympicArena()
#     core = gecko()
#     world.spawn(core.spec, spawn_position=SPAWN_POS)
#     probe_model = world.spec.compile()
#     model_nu = probe_model.nu
#     flat_size = probe_dims_for_policy(policy)

#     # Initialize population
#     population: list[dict[str, Any]] = []
#     for _ in range(pop_size):
#         p = init_param_vector(flat_size, policy, model_nu=model_nu)
#         f, _ = evaluate_brain_on_gecko_with_params(
#             p, policy=policy, duration=15.0, seeds=seeds, record_video=False
#         )
#         population.append({"params": p, "fitness": f})

#     best = max(population, key=lambda ind: ind["fitness"])
#     console.log(f"Gen 0: best={best['fitness']:.4f} mean={np.mean([i['fitness'] for i in population]):.4f}")

#     # Evolution loop
#     for g in range(1, gens+1):
#         # Dynamic episode duration based on current best
#         duration = duration_from_best(best["fitness"])

#         # Elitism: keep top μ (e.g., 20% of pop)
#         mu = max(1, pop_size // 5)
#         elites = sorted(population, key=lambda ind: ind["fitness"], reverse=True)[:mu]

#         # Generate offspring to refill population
#         offspring: list[dict[str, Any]] = []
#         while len(offspring) < pop_size - mu:
#             p1 = tournament_select(population, k=3)
#             p2 = tournament_select(population, k=3)
#             child_params = uniform_crossover(p1["params"], p2["params"], p=crossover_p)
#             child_params = mutate(child_params, sigma_mut, policy, model_nu=model_nu)
#             f, _ = evaluate_brain_on_gecko_with_params(
#                 child_params, policy=policy, duration=duration, seeds=seeds, record_video=False
#             )
#             offspring.append({"params": child_params, "fitness": f})

#         population = elites + offspring
#         gen_best = max(population, key=lambda ind: ind["fitness"])
#         if gen_best["fitness"] > best["fitness"]:
#             best = {"params": gen_best["params"].copy(), "fitness": gen_best["fitness"]}

#         console.log(f"Gen {g}: best={best['fitness']:.4f} mean={np.mean([i['fitness'] for i in population]):.4f} dur={duration:.0f}s")

#         # Simple checkpoint
#         ckpt_path = DATA / f"brain_only_gen{g}.npz"
#         np.savez_compressed(
#             ckpt_path,
#             policy=policy,
#             best_fitness=np.array([best["fitness"]]),
#             best_params=best["params"],
#             pop_fitness=np.array([ind['fitness'] for ind in population]),
#         )

#     return best



#-----------------------------------------------------------------#
# Old versions without flat-parameter APIs
# class MLPPolicy:
#     """Stateless MLP; weights set at init. Later you’ll pass EA-evolved weights."""
#     def __init__(self, model: mj.MjModel, rng: np.random.Generator, hidden: int = 32):
#         # Inputs: qpos (you can add qvel later)
#         self.in_dim = model.nq
#         self.h = hidden
#         self.out_dim = model.nu

#         # Random init for Step 1; EA will replace these
#         self.W1 = rng.normal(0, 0.2, size=(self.in_dim, self.h))
#         self.W2 = rng.normal(0, 0.2, size=(self.h, self.h))
#         self.W3 = rng.normal(0, 0.2, size=(self.h, self.out_dim))

#     def __call__(self, model: mj.MjModel, data: mj.MjData) -> npt.NDArray[np.float64]:
#         x = np.asarray(data.qpos)  # shape [nq]
#         h1 = np.tanh(x @ self.W1)
#         h2 = np.tanh(h1 @ self.W2)
#         y  = np.tanh(h2 @ self.W3)  # [-1,1]
#         return y * np.pi            # scale to [-π, π]

# class CPGPolicy:
#     """
#     Uncoupled sinusoidal CPG per actuator:
#         u_j(t) = A_j * sin(ω_j * t + φ_j) + b_j
#     Returns values in [-π, π].
#     """
#     def __init__(self, model: mj.MjModel, rng: np.random.Generator):
#         self.nu = model.nu

#         # Reasonable random init ranges (tune later)
#         # amplitudes in [0.2π, 0.6π] to avoid saturating at start
#         self.A = rng.uniform(0.2*np.pi, 0.6*np.pi, size=self.nu)
#         # frequencies in [0.5, 3.0] rad/s
#         self.omega = rng.uniform(0.5, 3.0, size=self.nu)
#         # phases in [0, 2π]
#         self.phi = rng.uniform(0.0, 2*np.pi, size=self.nu)
#         # biases in [-0.25π, 0.25π]
#         self.bias = rng.uniform(-0.25*np.pi, 0.25*np.pi, size=self.nu)

#     def __call__(self, model: mj.MjModel, data: mj.MjData) -> npt.NDArray[np.float64]:
#         # data.time is simulation time (s)
#         t = data.time
#         u = self.A * np.sin(self.omega * t + self.phi) + self.bias
#         # hard clamp to actuator expected range
#         return np.clip(u, -np.pi, np.pi)

#-----------------------------------------------------------------#
# def evaluate_brain_on_gecko(
#     duration: float = 15.0,
#     spawn_pos: list[float] = SPAWN_POS,
#     record_video: bool = False,
#     policy: Literal["mlp", "cpg"] = "mlp",
# ) -> tuple[float, list[list[float]]]:
#     """Run one headless rollout of Gecko in OlympicArena using the chosen policy."""
#     mj.set_mjcb_control(None)

#     world = OlympicArena()
#     core = gecko()  # fixed body for Phase 1
#     world.spawn(core.spec, spawn_position=spawn_pos)

#     model = world.spec.compile()
#     data = mj.MjData(model)
#     tracker = Tracker(mj.mjtObj.mjOBJ_GEOM, "core")

#     # ---- choose brain ----
#     if policy == "mlp":
#         brain = MLPPolicy(model, RNG, hidden=32)
#     else:
#         brain = CPGPolicy(model, RNG)

#     ctrl = Controller(controller_callback_function=brain, tracker=tracker)

#     mj.mj_resetData(model, data)
#     tracker.setup(world.spec, data)
#     mj.set_mjcb_control(lambda m, d: ctrl.set_control(m, d))

#     if record_video:
#         path_to_video_folder = str(DATA / "videos")
#         video_recorder = VideoRecorder(output_folder=path_to_video_folder)
#         video_renderer(model, data, duration=duration, video_recorder=video_recorder)
#     else:
#         simple_runner(model, data, duration=duration)

#     history = tracker.history["xpos"][0]
#     fit = fitness_function(history)
#     return float(fit), history
#-----------------------------------------------------------------#


def main() -> None:
    # """Entry point."""
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
    # # Print all nodes
    # core = construct_mjspec_from_graph(robot_graph)

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


    # Try the CPG first; change to "mlp" to compare step 1
    # fitness, history = evaluate_brain_on_gecko(duration=15.0, policy="cpg")
    # console.log(f"[Step 1 - CPG] rollout fitness: {fitness:.4f}")
    # show_xpos_history(history)

    # """
    # Step 2: Brain-only evolution on Gecko in OlympicArena.
    # Toggle policy between "cpg" and "mlp" below.
    # """
    # policy: Literal["mlp","cpg"] = "mlp"  # try "mlp" as well

    # best = evolve_brain_only(
    #     policy=policy,
    #     pop_size=24,
    #     gens=20,
    #     sigma_mut=0.05 if policy == "mlp" else 0.03,
    #     crossover_p=0.5,
    #     seeds=(0,1),
    # )
    # console.log(f"[EA done] Best fitness: {best['fitness']:.4f}")

    # # Render the best as a video and show path once
    # fit, history = evaluate_brain_on_gecko_with_params(
    #     best["params"], policy=policy, duration=duration_from_best(best["fitness"]), record_video=True
    # )
    # console.log(f"[Best video] Fitness (re-eval): {fit:.4f}")
    # show_xpos_history(history)
    


    # # Pick controller type
    # policy = "mlp"  # or "mlp"

    # # Mutation scale mapping (from your earlier warning fix)
    # SIGMA_BY_POLICY = {"mlp": 0.05, "cpg": 0.03}
    # sigma_mut = SIGMA_BY_POLICY[policy]

    # best = evolve_brain_only(
    #     policy=policy,
    #     pop_size=24,
    #     gens=20,
    #     sigma_mut=sigma_mut,
    #     crossover_p=0.5,
    #     seeds=(0, 1),
    #     # lifetime learning
    #     learn_steps=2,          # try 1–3
    #     learn_sigma=0.02,       # tune 0.01–0.05
    #     learn_duration=8.0,     # short inner rollouts
    #     lamarckian=False,       # keep Darwinian by default
    # )
    # console.log(f"[EA+Learn] Best fitness: {best['fitness']:.4f}")

    # # Re-eval & video from true start with full duration
    # full_dur = duration_from_best(best["fitness"])
    # fit, history = evaluate_brain_on_gecko_with_params(
    #     best["params"], policy=policy, duration=full_dur, record_video=True
    # )
    # console.log(f"[Best video] Fitness (re-eval): {fit:.4f}")
    # show_xpos_history(history)

    policy = "cpg_paired"  # or "mlp"

    # --- CMA-ES run ---
    best = evolve_brain_cmaes(
        policy=policy,
        iters=40,
        seeds=(0, 1),
        learn_steps=2,
        learn_sigma=0.02,
        learn_duration=8.0,
        lamarckian=False,
        sigma0_mlp=0.3,
        sigma0_cpg=0.1,
        popsize=16,  # let it default to 4 + floor(3*ln(n))
    )
    console.log(f"[CMA done] Best fitness: {best['fitness']:.4f}")

    # Re-eval + video from the true start with full duration
    full_dur = duration_from_best(best["fitness"])
    fit, history = evaluate_brain_on_gecko_with_params(
        best["params"], policy=policy, duration=full_dur, record_video=True
    )
    console.log(f"[Best video] Fitness (re-eval): {fit:.4f}")
    show_xpos_history(history)


if __name__ == "__main__":
    main()
