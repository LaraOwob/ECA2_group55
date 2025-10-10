"""Assignment 3 template code."""

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


def fitness_function(history: list[tuple[float, float, float]]) -> float:
    xt, yt, zt = TARGET_POSITION
    xc, yc, zc = history[-1]

    # Minimize the distance --> maximize the negative distance
    cartesian_distance = np.sqrt(
        (xt - xc) ** 2 + (yt - yc) ** 2 + (zt - zc) ** 2,
    )
    return -cartesian_distance

def shaped_fitness(
    history: list[tuple[float, float, float]],
    *,
    goal_x: float = TARGET_POSITION[0],
    w_progress: float = 1.0,
    w_finish: float = 2.0,
    w_time: float = 1.0,
    w_lat: float = 0.5,
    w_drop: float = 0.5,
) -> float:
    """
    Reward forward progress + finishing early; penalize lateral drift and falling.

    Returns a scalar to MAXIMIZE. Scales are modest so CMA-ES behaves well.
    """
    arr = np.asarray(history, dtype=np.float64)  # [T, 3]
    x = arr[:, 0]; y = arr[:, 1]; z = arr[:, 2]

    # forward progress along +x
    progress = float(x[-1] - x[0])  # meters

    # finish bonus (first index where we pass goal_x)
    finish_bonus = 0.0
    time_bonus = 0.0
    passed = np.nonzero(x >= goal_x)[0]
    if passed.size > 0:
        t_reach = int(passed[0])
        T = len(x)
        finish_bonus = w_finish
        # more bonus if it reaches earlier in the episode
        time_bonus = w_time * (1.0 - t_reach / max(T - 1, 1))

    # lateral drift/zig-zag penalty: std dev in y
    lat_pen = w_lat * float(np.std(y))

    # vertical drop penalty: if it slams low, discourage that
    # (OlympicArena floor is around z ~ 0; core is ~0.1 at spawn)
    min_z = float(np.min(z))
    drop_pen = w_drop * float(max(0.0, 0.08 - min_z))  # penalize going below ~8 cm

    # final shaped reward
    return (w_progress * progress) + finish_bonus + time_bonus - lat_pen - drop_pen



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


#------------------------------------------------------------------#
#------------------------------------------------------------------#
class MLPPolicy:
    """
    Stateless MLP controller with tanh activations.
    Flat params layout: [W1 (nq*h), W2 (h*h), W3 (h*nu)]
    """
    def __init__(self, model: mj.MjModel, rng: np.random.Generator, hidden: int = 32):
        self.in_dim = model.nq
        self.h = hidden
        self.out_dim = model.nu
        # random init (can be overwritten with set_flat_params later)
        self.W1 = rng.normal(0, 0.2, size=(self.in_dim, self.h))
        self.W2 = rng.normal(0, 0.2, size=(self.h, self.h))
        self.W3 = rng.normal(0, 0.2, size=(self.h, self.out_dim))

    @property
    def flat_size(self) -> int:
        return self.in_dim*self.h + self.h*self.h + self.h*self.out_dim

    def set_flat_params(self, w: npt.NDArray[np.float64]) -> None:
        assert w.size == self.flat_size, f"Expected {self.flat_size}, got {w.size}"
        i = 0
        n = self.in_dim*self.h
        self.W1 = w[i:i+n].reshape(self.in_dim, self.h); i += n
        n = self.h*self.h
        self.W2 = w[i:i+n].reshape(self.h, self.h); i += n
        n = self.h*self.out_dim
        self.W3 = w[i:i+n].reshape(self.h, self.out_dim)

    def __call__(self, model: mj.MjModel, data: mj.MjData) -> npt.NDArray[np.float64]:
        x = np.asarray(data.qpos)             # [nq]
        h1 = np.tanh(x @ self.W1)
        h2 = np.tanh(h1 @ self.W2)
        y  = np.tanh(h2 @ self.W3)            # [-1, 1]
        return y * np.pi                      # scale to [-π, π]
#------------------------------------------------------------------#
#------------------------------------------------------------------#


#------------------------------------------------------------------#
#------------------------------------------------------------------#
def evaluate_brain_on_gecko_once(
    duration: float = 15.0,
    spawn_pos: list[float] = SPAWN_POS,
    hidden: int = 32,
    record_video: bool = False,
) -> tuple[float, list[tuple[float, float, float]]]:
    """
    Build OlympicArena + Gecko, run a single headless rollout with an MLP brain.
    Returns (fitness, history_of_core_positions).
    """
    # Reset MuJoCo control callback
    mj.set_mjcb_control(None)

    # World + fixed body (Gecko)
    world = OlympicArena()
    core = gecko()
    world.spawn(core.spec, position=spawn_pos)

    # Model/Data/Tracker
    model = world.spec.compile()
    data = mj.MjData(model)
    tracker = Tracker(mj.mjtObj.mjOBJ_GEOM, "core")

    # Controller (MLP brain)
    brain = MLPPolicy(model, RNG, hidden=hidden)
    ctrl = Controller(controller_callback_function=brain, tracker=tracker)

    # Reset, setup tracker, set callback
    mj.mj_resetData(model, data)
    tracker.setup(world.spec, data)
    mj.set_mjcb_control(lambda m, d: ctrl.set_control(m, d))

    # Headless run (or record)
    if record_video:
        path_to_video_folder = str(DATA / "videos")
        video_recorder = VideoRecorder(output_folder=path_to_video_folder)
        video_renderer(model, data, duration=duration, video_recorder=video_recorder)
    else:
        simple_runner(model, data, duration=duration)

    history = tracker.history["xpos"][0]  # list of (x,y,z)
    fit = fitness_function(history)
    return float(fit), history
#------------------------------------------------------------------#
#------------------------------------------------------------------#

# === MLP CMA-ES helpers =======================================================

def probe_mlp_flat_size(hidden: int = 32) -> int:
    """Build a probe model to learn the MLP flat parameter size."""
    world = OlympicArena()
    core = gecko()
    world.spawn(core.spec, position=SPAWN_POS)
    model = world.spec.compile()
    brain = MLPPolicy(model, RNG, hidden=hidden)
    return brain.flat_size

def init_mlp_vector(size: int) -> npt.NDArray[np.float64]:
    """Small Gaussian init for MLP weights."""
    return RNG.normal(0.0, 0.2, size=size).astype(np.float64)

def mlp_bounds(size: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Simple weight box bounds to keep things sane."""
    low  = np.full(size, -3.0, dtype=np.float64)
    high = np.full(size,  3.0, dtype=np.float64)
    return low, high

def duration_from_best(best_f: float) -> float:
    """Map best fitness (−distance) to episode length (s)."""
    dist = -best_f
    if dist > 3.0: return 15.0   # early: just reach rugged
    if dist > 1.0: return 45.0   # mid: clear rugged
    return 100.0                 # late: uphill to finish
# ==============================================================================

#------------------------------------------------------------------#
#------------------------------------------------------------------#
def evaluate_mlp_params(
    params: npt.NDArray[np.float64],
    duration: float = 15.0,
    spawn_pos: list[float] = SPAWN_POS,
    seeds: tuple[int, ...] = (0, 1),
    hidden: int = 32,
    record_video: bool = False,
    use_shaping: bool = True,
) -> tuple[float, list[tuple[float, float, float]]]:
    """
    Evaluate *given* MLP weights on Gecko in OlympicArena.
    Returns (mean_fitness_over_seeds, history_from_last_seed).
    If use_shaping=True -> uses shaped_fitness; else -> official fitness_function.
    """
    last_history: list[tuple[float, float, float]] = []
    scores: list[float] = []

    for s in seeds:
        rng_local = np.random.default_rng(SEED + s)
        mj.set_mjcb_control(None)

        # World + Gecko
        world = OlympicArena()
        core = gecko()
        world.spawn(core.spec, position=spawn_pos)

        # Model/Data/Tracker
        model = world.spec.compile()
        data = mj.MjData(model)
        tracker = Tracker(mj.mjtObj.mjOBJ_GEOM, "core")

        # MLP brain with provided params
        brain = MLPPolicy(model, rng_local, hidden=hidden)
        brain.set_flat_params(params)
        ctrl = Controller(controller_callback_function=brain, tracker=tracker)

        # Reset + attach
        mj.mj_resetData(model, data)
        tracker.setup(world.spec, data)
        mj.set_mjcb_control(lambda m, d: ctrl.set_control(m, d))

        # Run
        if record_video and s == seeds[-1]:
            path = str(DATA / "videos")
            video_recorder = VideoRecorder(output_folder=path)
            video_renderer(model, data, duration=duration, video_recorder=video_recorder)
        else:
            simple_runner(model, data, duration=duration)

        history = tracker.history["xpos"][0]
        score = shaped_fitness(history) if use_shaping else fitness_function(history)
        scores.append(float(score))
        last_history = history

    return float(np.mean(scores)), last_history

#------------------------------------------------------------------#
#------------------------------------------------------------------#

# ----------------------------------------------------------#
# ----------------------------------------------------------#
def evolve_mlp_cmaes(
    iters: int = 50,
    hidden: int = 32,
    seeds: tuple[int, ...] = (0, 1),
    sigma0: float = 0.3,
    popsize: int | None = None,
) -> dict[str, Any]:
    """
    CMA-ES over MLP weights (Gecko body, OlympicArena).
    Returns {"params": best_weights, "fitness": best_fitness}.
    """
    try:
        import cma  # type: ignore
    except Exception:
        console.log("[CMA-ES] Please install pycma: pip install cma")
        raise

    # Probe dimensionality
    flat_size = probe_mlp_flat_size(hidden=hidden)
    m0 = init_mlp_vector(flat_size)
    lb, ub = mlp_bounds(flat_size)

    if popsize is None:
        popsize = max(4, 4 + int(3 * np.log(flat_size + 1)))  # safe default ≥4

    opts = {
        "bounds": [lb.tolist(), ub.tolist()],
        "popsize": popsize,
        "seed": SEED,
        "verbose": -9,  # we log manually
    }
    es = cma.CMAEvolutionStrategy(m0, sigma0, opts)

    best = {"params": m0.copy(), "fitness": -1e9}
    duration = 15.0

    for g in range(1, iters + 1):
        xs = es.ask()  # population batch
        fitnesses: list[float] = []

        # Evaluate whole batch
        for x in xs:
            x = np.asarray(x, dtype=np.float64)
            f, _ = evaluate_mlp_params(
                x,
                duration=duration,
                seeds=seeds,
                hidden=hidden,
                record_video=False,
            )
            fitnesses.append(f)

        # CMA minimizes → pass negative fitness
        es.tell(xs, [-f for f in fitnesses])

        # Track & log best of batch
        idx = int(np.argmax(fitnesses))
        if fitnesses[idx] > best["fitness"]:
            best = {"params": np.asarray(xs[idx], dtype=np.float64).copy(), "fitness": float(fitnesses[idx])}

        duration = duration_from_best(best["fitness"])
        console.log(f"[CMA] iter {g}/{iters}: best={best['fitness']:.4f} mean={np.mean(fitnesses):.4f} dur={duration:.0f}s")

        # Simple checkpoint
        ckpt = DATA / f"mlp_cma_iter{g}.npz"
        np.savez_compressed(
            ckpt,
            best_fitness=np.array([best["fitness"]]),
            best_params=best["params"],
            pop_fitness=np.array(fitnesses),
            hidden=np.array([hidden]),
        )

        if es.stop():
            break

    return best
# ----------------------------------------------------------#
# ----------------------------------------------------------#




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
    world.spawn(robot.spec, position=SPAWN_POS)
    #world.spawn(robot.spec, spawn_position=SPAWN_POS)


    # Generate the model and data
    # These are standard parts of the simulation USE THEM AS IS, DO NOT CHANGE
    model = world.spec.compile()
    data = mj.MjData(model)

    # Reset state and time of simulation
    mj.mj_resetData(model, data)

    # Pass the model and data to the tracker
    controller.tracker.setup(world.spec, data)

    # Set the control callback function
    # This is called every time step to get the next action.
    args: list[Any] = []  # IF YOU NEED MORE ARGUMENTS ADD THEM HERE!
    kwargs: dict[Any, Any] = {}  # IF YOU NEED MORE ARGUMENTS ADD THEM HERE!

    mj.set_mjcb_control(
        lambda m, d: controller.set_control(m, d, *args, **kwargs),  # pyright: ignore[reportUnknownLambdaType]
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
    # """Entry point."""
    # # ? ------------------------------------------------------------------ #
    # genotype_size = 64
    # type_p_genes = RNG.uniform(-100, 100, genotype_size).astype(np.float32)
    # conn_p_genes = RNG.uniform(-100, 100, genotype_size).astype(np.float32)
    # rot_p_genes = RNG.uniform(-100, 100, genotype_size).astype(np.float32)

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

    # ? ------------------------------------------------------------------ #
    # """
    # Phase 1 — Step 1:
    # Single headless rollout on OlympicArena using Gecko + MLP controller.
    # (No EA yet. This sets up the exact evaluation we’ll optimize with CMA-ES in Step 2.)
    # """
    # fitness, history = evaluate_brain_on_gecko_once(
    #     duration=15.0,
    #     spawn_pos=SPAWN_POS,
    #     hidden=32,
    #     record_video=False,
    # )
    # console.log(f"[Step 1] Single rollout fitness: {fitness:.4f}")
    # show_xpos_history(history)

    """
    Phase 1 — Step 2:
    CMA-ES optimizes MLP weights on Gecko in OlympicArena (headless eval).
    """
    # best = evolve_mlp_cmaes(
    #     iters=40,          # try 40–80 for a decent first pass
    #     hidden=32,         # you can try 64 later
    #     seeds=(0,),      # average over a couple seeds for robustness
    #     sigma0=0.3,        # initial step-size; 0.3–0.6 works
    #     popsize=None,      # default popsize rule, or set e.g. 12–20
    # )
    # console.log(f"[CMA done] Best fitness: {best['fitness']:.4f}")

    # # Re-evaluate best once with full duration and save a video
    # full_dur = duration_from_best(best["fitness"])
    # fit, history = evaluate_mlp_params(
    #     best["params"],
    #     duration=full_dur,
    #     seeds=(0,),
    #     hidden=32,
    #     record_video=True,
    # )
    # console.log(f"[Best video] Fitness (re-eval): {fit:.4f}")
    # show_xpos_history(history)

    best = evolve_mlp_cmaes(
        iters=40,
        hidden=32,
        seeds=(0,),      # start with 1 seed for speed; add (0,1) later
        sigma0=0.4,
        popsize=16,
    )
    console.log(f"[CMA done] Best shaped score: {best['fitness']:.4f}")

    # Re-evaluate best with the official metric and full duration
    full_dur = 100.0
    # official metric → use_shaping=False
    dist_score, history = evaluate_mlp_params(
        best["params"], duration=full_dur, seeds=(0,), hidden=32, record_video=True, use_shaping=False
    )
    console.log(f"[Official distance score] {dist_score:.4f}")
    show_xpos_history(history)

if __name__ == "__main__":
    main()
