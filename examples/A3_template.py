"""Assignment 3 template code."""

# Standard library
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

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

# Type Checking
if TYPE_CHECKING:
    from networkx import DiGraph

# Type Aliases
type ViewerTypes = Literal["launcher", "video", "simple", "no_control", "frame"]

# --- RANDOM GENERATOR SETUP --- #
SEED = 42
RNG = np.random.default_rng(SEED)
TARGET_POS = np.array([0, -2, 0.1])
SPAWN=[0, 0, 0.1]

# --- DATA SETUP ---
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True)
TOLERANCE = 0.1 
CTRL_MAX = np.pi / 2  # match controller clipping


def show_xpos_history(history: list[float]) -> None:
    # Convert list of [x,y,z] positions to numpy array
    pos_data = np.array(history)

    # Create figure and axis
    plt.figure(figsize=(10, 6))

    # Plot x,y trajectory
    plt.plot(pos_data[:, 0], pos_data[:, 1], "b-", label="Path")
    plt.plot(pos_data[0, 0], pos_data[0, 1], "go", label="Start")
    plt.plot(pos_data[-1, 0], pos_data[-1, 1], "ro", label="End")
    plt.plot(0, 0, "kx", label="Origin")

    # Add labels and title
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Robot Path in XY Plane")
    plt.legend()
    plt.grid(visible=True)

    # Set equal aspect ratio and center at (0,0)
    plt.axis("equal")

    # Show results
    plt.show()


def random_move(
    model: mj.MjModel,
    data: mj.MjData,
) -> npt.NDArray[np.float64]:
    # Get the number of joints
    num_joints = model.nu

    # Hinges take values between -pi/2 and pi/2
    hinge_range = np.pi / 2
    return RNG.uniform(
        low=-hinge_range,  # -pi/2
        high=hinge_range,  # pi/2
        size=num_joints,
    ).astype(np.float64)


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


class MLPController:
    """
    Tiny MLP: [prev_ctrl (nu), sin(wt), cos(wt)] -> ctrl (nu)
    Two layers: (in_dim -> hidden) -> (hidden -> out_dim) with tanh activations.
    Output is scaled to [-CTRL_RANGE, CTRL_RANGE].
    """
    def __init__(self, nu: int, hidden: int = 32, freq_hz: float = 1.0):
        self.nu = nu
        self.hidden = hidden
        self.in_dim = nu + 4
        self.out_dim = nu
        self.freq_hz = freq_hz
        # Parameter shapes
        self.W1_shape = (self.in_dim, hidden)
        self.b1_shape = (hidden,)
        self.W2_shape = (hidden, self.out_dim)
        self.b2_shape = (self.out_dim,)
        self.n_params = (
            self.in_dim*self.hidden +
            self.hidden +
            self.hidden*self.out_dim +
            self.out_dim
        )

    def set_params(self, theta: np.ndarray):
        #Checks length of solution, if it matches the expected number of parameters
        assert theta.shape[0] == self.n_params
        #Saves copy of solutoin, and saves in object
        self.theta = theta.copy()

    def _unpack(self):
        #Retrieve solution (values)
        t = self.theta
        i = 0
        #Unpacks solution in W1, b1, W2, b2
        W1 = t[i:i+self.in_dim*self.hidden].reshape(self.in_dim, self.hidden); i += self.in_dim*self.hidden
        b1 = t[i:i+self.hidden]; i += self.hidden
        W2 = t[i:i+self.hidden*self.out_dim].reshape(self.hidden, self.out_dim); i += self.hidden*self.out_dim
        b2 = t[i:i+self.out_dim]
        return W1, b1, W2, b2
    
    #Applies forward pass and returns MLP output
   
    def forward(self, prev_ctrl, t, target_pos, current_pos):
        w = 2*np.pi*self.freq_hz
        delta = target_pos[:2] - current_pos[:2] 
        obs = np.concatenate([prev_ctrl, [np.sin(w*t), np.cos(w*t)], delta], dtype=np.float64)
        W1, b1, W2, b2 = self._unpack()
        h = np.tanh(obs @ W1 + b1)
        y = np.tanh(h @ W2 + b2)
        return np.pi * y  
    

def mlp_forward(model, data, network, core_bind, *args, **kwargs):
    prev_ctrl = data.ctrl.copy()
    current_pos = data.xpos[core_bind.id]

    t = data.time               # use actual simulation time
    return network.forward(prev_ctrl, t, TARGET_POS, current_pos)


def create_body():
    num_modules = 20

    # ? ------------------------------------------------------------------ #
    genotype_size = 64
    type_p_genes = RNG.random(genotype_size).astype(np.float32)
    conn_p_genes = RNG.random(genotype_size).astype(np.float32)
    rot_p_genes = RNG.random(genotype_size).astype(np.float32)

    genotype = [
        type_p_genes,
        conn_p_genes,
        rot_p_genes,
    ]

    nde = NeuralDevelopmentalEncoding(number_of_modules=num_modules)
    p_matrices = nde.forward(genotype)

    # Decode the high-probability graph
    hpd = HighProbabilityDecoder(num_modules)
    robot_graph: DiGraph[Any] = hpd.probability_matrices_to_graph(
        p_matrices[0],
        p_matrices[1],
        p_matrices[2],
    )

    # ? ------------------------------------------------------------------ #
    # Save the graph to a file
    save_graph_as_json(
        robot_graph,
        DATA / "robot_graph.json",
    )

    # ? ------------------------------------------------------------------ #
    # Print all nodes
    core = construct_mjspec_from_graph(robot_graph)
    core = gecko()
    return core

def create_tracker():
    tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")
    return tracker


def evaluate_fitness(candidate, model, data, ctrl, network, core_bind, duration=30.0, alpha = 0.2):
    
    """
    Evaluate MLP controller fitness using the simulation runner and Controller class.
    
    Parameters
    ----------
    model : mujoco.MjModel
    data : mujoco.MjData
    ctrl : Controller
        Controller instance to use for simulation.
    network : MLPController
        MLP network providing forward() method.
    duration : float
        Simulation duration in seconds.
    
    Returns
    -------
    float
        Fitness (higher is better).
    """
    network.set_params(candidate)

    # Reset simulation
    mj.mj_resetData(model, data)
    
    # Assign network forward as controller callback
    ctrl.controller_callback_function = partial(mlp_forward, network=network, core_bind = core_bind)
    
    # Set controller in MuJoCo
    mj.set_mjcb_control(ctrl.set_control)
    
    # Run simulation
    simple_runner(model, data, duration=duration)
    

    dist = np.linalg.norm(data.xpos[core_bind.id] - TARGET_POS)
   
    
    # Fitness: inverse of distance (closer is better)
    fitness = 1.0 / (dist + 1e-6)  # add epsilon to avoid div by zero

    
    return fitness




def evaluate_fitness2(candidate, model, data, ctrl, network, core_bind, duration=15, alpha=0.1):
    """
    Simple fitness: encourages fast movement toward TARGET_POS on rugged terrain.

    Fitness increases if the robot moves closer to the target quickly.
    """
    beta = 0.05
    alpha = 0.1
    network.set_params(candidate)

    # Reset simulation
    mj.mj_resetData(model, data)

    # Assign network forward as controller callback
    ctrl.controller_callback_function = partial(mlp_forward, network=network, core_bind=core_bind)
    mj.set_mjcb_control(ctrl.set_control)

    start_pos = data.xpos[0].copy()
    dt = 1.0 / network.freq_hz
    steps = int(duration / dt)
    prev_ctrl = np.zeros(network.nu)
    total_control_effort = 0.0

    # --- Run simulation ---
    for step in range(steps):
        t = step * dt
        current_pos = data.xpos[0].copy()

        # Forward pass (target given to NN)
        u = network.forward(prev_ctrl, t, TARGET_POS, current_pos)
        prev_ctrl = u.copy()
        total_control_effort += np.sum(np.abs(u))

        data.ctrl[:] = u

        # Step simulation
        mj.mj_step(model, data)

    # --- Compute fitness ---
    end_pos = data.xpos[0].copy()
    move_vec = end_pos - start_pos
    target_vec = TARGET_POS[:2] - start_pos[:2]
    target_dist = np.linalg.norm(target_vec)

    if target_dist < 1e-6:
        target_dir = np.array([1.0, 0.0])
    else:
        target_dir = target_vec / target_dist

    # Progress along target
    progress = np.dot(move_vec[:2], target_dir)

    # Side drift (perpendicular component)
    side_drift = np.linalg.norm(move_vec[:2] - progress * target_dir)

    # Combine into fitness
    fitness = progress - alpha * side_drift - beta * total_control_effort

    return fitness


def evaluate_fitness3(theta, model, data, ctrl, network, core_bind, T, alpha = 0.2):
    """
    Simple, stable fitness: reward forward walking straight toward +Y.
    Works very well on flat terrain.
    """
    # Set NN parameters
    network.set_params(theta)

    # Reset simulation
    mj.mj_resetData(model, data)

    # Assign controller
    ctrl.controller_callback_function = partial(mlp_forward, network=network, core_bind=core_bind)
    mj.set_mjcb_control(ctrl.set_control)

    # Record start position
    start_pos = data.xpos[core_bind.id].copy()

    dt = 1.0 / network.freq_hz

    simple_runner(model, data, duration=T)

     # Compute displacement
    end_pos = data.xpos[core_bind.id].copy()
    move_vec = end_pos - start_pos

    # Forward (Y-axis) progress
    forward_progress = -move_vec[1]

    # Side drift (X-axis)
    side_drift = abs(move_vec[0])

    # Reward forward motion, penalize sideways drift
    fitness = forward_progress - 0.1 * side_drift

    return fitness











import cma
from functools import partial
import numpy as np


def train_mlp_cma(
    generations=20,
    pop_size=None,
    T=30.0,
    seed=50,
    model=None,
    data=None,
    core_bind=None,
    hidden=32,
    freq_hz=1.0,
    sigma=0.5,
    init_scale=0.5,
    tracker=None,
    alpha=0.2,
    world=None,
):
    """
    CMA-ES optimization for MLP controller parameters (maximizing fitness).
    """
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    # Initialize MLP controller
    nu = model.nu
    network = MLPController(nu=nu, hidden=hidden, freq_hz=freq_hz)
    num_params = network.n_params
    mean_init = rng.normal(0.0, init_scale, size=num_params)

    # Tracker setup
    if tracker is not None and world is not None:
        tracker.setup(world.spec, data)

    # Controller wrapper
    ctrl = Controller(
        controller_callback_function=partial(mlp_forward, network=network, core_bind=core_bind),
        tracker=tracker
    )

    # CMA-ES setup
    es = cma.CMAEvolutionStrategy(mean_init, sigma, {
        "popsize": pop_size,
        "seed": seed,
        "maxiter": generations,
        "verb_disp": 1
    })

    best_candidate = None
    best_fit = -np.inf  # we want to maximize fitness

    # Evolution loop
    for gen in range(generations):
        candidates = es.ask()
        fitnesses = []

        # Evaluate all candidates
        for cand in candidates:
            fit = evaluate_fitness3(cand, model, data, ctrl, network, core_bind, T, alpha=alpha)
            # Negate because CMA minimizes by default
            fitnesses.append(-fit)

        es.tell(candidates, fitnesses)
        es.disp()

        # Correctly identify the best candidate in this generation
        best_idx = int(np.argmin(fitnesses))  # min(-fitness) = max(fitness)
        gen_best_fit = -fitnesses[best_idx]   # revert negation
        gen_best_candidate = candidates[best_idx]

        # Update global best
        if gen_best_fit > best_fit:
            best_fit = gen_best_fit
            best_candidate = gen_best_candidate.copy()

        if (gen + 1) % max(1, generations // 10) == 0:
            print(f"[mlp_cma seed {seed}] Gen {gen+1}/{generations}: best={gen_best_fit:.3f}")

        if es.stop():
            break

    return best_candidate, best_fit









def train_mlp_de(generations=20, pop_size=20, T=20, seed=50,
                 model=None, data=None, core_bind=None,
                 hidden=32, freq_hz=1.0, F=0.7, CR=0.9, init_scale=0.5,
                 tracker=None, alpha=0.2, world = SimpleFlatWorld):
    

  
    """
    Simple, Differential Evolution for MLP parameters.
    """
    #Selects random number
    # 
    # 
    # 
    #  
    rng = np.random.default_rng(seed)
    print(model)
    nu = model.nu
    print(model.nu)
    network = MLPController(nu=nu, hidden=32, freq_hz=1.0)
    number_params = network.n_params
    random_theta = rng.normal(0.0, 0.5, size=number_params)
    network.set_params(random_theta)

    

    if tracker is not None:
        tracker.setup(world.spec, data)
    ctrl = Controller(
        controller_callback_function=partial(mlp_forward, network=network, core_bind = core_bind),
        tracker=tracker
    )

    # Initialize population by sampling normal distribution with mean 0 and stdv of init_scale
    pop = rng.normal(loc=0.0, scale=init_scale, size=(pop_size,number_params ))
    #Initialize empty numby array with length number of candidates
    fits = np.empty(pop_size, dtype=np.float64)
   

    # Calculate fitness for each candidate, pop[i] = candidate
    for i in range(pop_size):
        fits[i] = evaluate_fitness(pop[i], model, data, ctrl, network, core_bind, T, alpha=0.2)
        #fits[i] = -loss  # fitness
   
    best_idx = int(np.argmax(fits))
    best_candidate = pop[best_idx].copy()
    best_fit = float(fits[best_idx])


    for g in range(generations):
        for i in range(pop_size):
            #Select all indices expcept for i, the candidate you want to mutate
            choices = []
            for index in range(pop_size):
                if index != i:
                    choices.append(index)
            #Out of valid choices, choose 3 random candidates 
            a, b, c = rng.choice(choices, size=3, replace=False)
            #Mutant candidate:
            mutant = pop[a] + F * (pop[b] - pop[c])

            # Crossover (binomial), between parent and mutant
            #Creates list of length weights whether each gene will be changed or stays like parent. Returns FAlse or True list
            cross = rng.random(number_params) < CR
            #At least one random gene will be open for crossover with mutant
            cross[rng.integers(0, number_params)] = True 
            #The genes which were chosen by cross validation proability are adjusted
            trial = np.where(cross, mutant, pop[i])

            # Evaluate trial
            f_trial = evaluate_fitness(trial, model, data, ctrl, network,core_bind, T, alpha = 0.2)
            #f_trial = -loss

            # If trial is better than parent than trial enters population
            if f_trial >= fits[i]:
                pop[i] = trial
                fits[i] = f_trial

        # Retrieve stats: best fitness, mean fitness and median fitness and write to csv
        gen_best = float(np.max(fits))
        gen_mean = float(np.mean(fits))
        gen_median = float(np.median(fits))

        # Update and check best candidate with best fitness
        if gen_best > best_fit:
            best_fit = gen_best
            best_candidate = pop[int(np.argmax(fits))].copy()

        #Print every after every 10th generation
        if (g + 1) % max(1, generations // 10) == 0:
            print(f"[mlp_de seed {seed}] Gen {g+1}/{generations}: best={gen_best:.3f} mean={gen_mean:.3f}")

    return best_candidate, best_fit



import numpy as np
import random as pyrand
from deap import base, creator, tools
from functools import partial

def train_mlp_ga(out_csv=None, generations=100, pop_size=60, T=10.0, seed=0,
                 model=None, data=None, core_bind=None, hidden=32, freq_hz=1.0,
                 crossover_prob=0.6, mutation_prob=0.3, mut_sigma=0.3, alpha=0.2,
                 tracker=None, world=None):
    """
    Genetic Algorithm on MLP parameters using DEAP. Maximizes fitness.
    Returns (best_theta, best_fitness).
    """
    rng = np.random.default_rng(seed)
    pyrand.seed(seed)
    np.random.seed(seed)

    # Initialize MLP controller
    nu = model.nu
    network = MLPController(nu=nu, hidden=hidden, freq_hz=freq_hz)
    n_params = network.n_params
    network.set_params(rng.normal(0.0, 0.5, size=n_params))

    # Tracker setup if provided
    if tracker is not None and world is not None:
        tracker.setup(world.spec, data)

    ctrl = Controller(
        controller_callback_function=partial(mlp_forward, network=network, core_bind=core_bind),
        tracker=tracker
    )

    # DEAP creator setup
    try:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    except Exception:
        pass
    try:
        creator.create("Individual", list, fitness=creator.FitnessMax)
    except Exception:
        pass

    # DEAP toolbox
    toolbox = base.Toolbox()
    toolbox.register("attr_float", pyrand.gauss, 0.0, 0.5)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n_params)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Evaluation
    def eval_ind(individual):
        theta = np.asarray(individual, dtype=np.float64)
        fitness = evaluate_fitness(theta, model, data, ctrl, network, core_bind, T, alpha=alpha)
        return (fitness,)

    toolbox.register("evaluate", eval_ind)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=mut_sigma, indpb=1.0/n_params)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Initialize population
    population = toolbox.population(n=pop_size)

    # Evaluate initial population
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)

    # Evolution loop
    for g in range(generations):
        # Selection
        offspring = list(map(toolbox.clone, toolbox.select(population, len(population))))

        # Crossover
        for i in range(0, len(offspring)-1, 2):
            if pyrand.random() < crossover_prob:
                toolbox.mate(offspring[i], offspring[i+1])
                del offspring[i].fitness.values
                del offspring[i+1].fitness.values

        # Mutation
        for ind in offspring:
            if pyrand.random() < mutation_prob:
                toolbox.mutate(ind)
                del ind.fitness.values

        # Evaluate invalid individuals
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)

        # Replace population
        population = offspring

        # Logging
        fits = np.array([ind.fitness.values[0] for ind in population])
        if (g + 1) % max(1, generations // 10) == 0:
            print(f"[mlp_ga seed {seed}] Gen {g+1}/{generations}: best={np.max(fits):.3f}, mean={np.mean(fits):.3f}")

    # Return best individual
    best_ind = tools.selBest(population, 1)[0]
    best_theta = np.asarray(best_ind, dtype=np.float64)
    best_fit = float(best_ind.fitness.values[0])

    return best_theta, best_fit


import numpy as np
from functools import partial

def train_mlp_pso_mujoco(
    generations=100,
    pop_size=30,
    T=25.0,
    seed=0,
    model=None,
    data=None,
    core_bind=None,
    hidden=32,
    freq_hz=1.0,
    w=0.5,
    c1=1.5,
    c2=1.5,
    alpha=0.2,
):
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    nu = model.nu
    network = MLPController(nu=nu, hidden=hidden, freq_hz=freq_hz)
    n_params = network.n_params

    # Initialize particles
    population = [rng.normal(0.0, 0.5, size=n_params) for _ in range(pop_size)]
    velocities = [rng.normal(0.0, 0.1, size=n_params) for _ in range(pop_size)]

    # Initialize personal bests
    pbest_positions = [p.copy() for p in population]
    pbest_scores = []
    ctrl = Controller(controller_callback_function=partial(mlp_forward, network=network, core_bind=core_bind))

    for theta in population:
        fitness = evaluate_fitness(theta, model, data, ctrl, network, core_bind, T, alpha)
        pbest_scores.append(fitness)

    # Initialize global best
    gbest_idx = int(np.argmax(pbest_scores))
    gbest_position = pbest_positions[gbest_idx].copy()
    gbest_score = pbest_scores[gbest_idx]

    for gen in range(generations):
        for i in range(pop_size):
            r1 = rng.random(n_params)
            r2 = rng.random(n_params)

            # PSO update
            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (pbest_positions[i] - population[i])
                + c2 * r2 * (gbest_position - population[i])
            )
            population[i] = population[i] + velocities[i]

            # Evaluate new position
            theta = population[i]
            fitness = evaluate_fitness(theta, model, data, ctrl, network, core_bind, T, alpha)

            # Update personal best
            if fitness > pbest_scores[i]:
                pbest_scores[i] = fitness
                pbest_positions[i] = theta.copy()

                # Update global best
                if fitness > gbest_score:
                    gbest_score = fitness
                    gbest_position = theta.copy()

        if (gen + 1) % max(1, generations // 10) == 0:
            print(f"[mlp_pso seed {seed}] Gen {gen+1}/{generations}: best={gbest_score:.3f}")

    return gbest_position, gbest_score




def experiment2():
    core = create_body()
    core = gecko()
    tracker = create_tracker()

    #Inialize world and model
    mj.set_mjcb_control(None) 
    world = SimpleFlatWorld()

    # Spawn robot in the world
    world.spawn(core.spec, spawn_position=[0, 0, 0.1])
    world.spec.worldbody.add_geom(
    type=mj.mjtGeom.mjGEOM_SPHERE,
    name="target_marker",
    size=[0.05, 0.0, 0.0],          
    pos=TARGET_POS, 
    rgba=[1, 0, 0, 1],  
    contype=0,            
    conaffinity=0)

    # Generate the model and data
    model = world.spec.compile()
    data = mj.MjData(model)

    # current_pos is the robot root position right after spawn
    current_pos = data.xpos[0]  # shape (3,)

    # Compute Euclidean distance to target (2D or 3D)
    delta = TARGET_POS[:3] - current_pos  # or [:2] if you only care about XY
    distance = np.linalg.norm(delta)

    print("Distance to target after spawn:", distance)

  
    #Number of hinges
    nu = model.nu
    tracker.setup(core.spec, data)
    core_bind = tracker.to_track[0]
    # theta, best_fit = train_mlp_de(
    #             generations=50, pop_size=50, T=20, seed=5,
    #             model=model, data=data, core_bind=core_bind,
    #             hidden=32, freq_hz=1.0, F=0.7, CR=0.9, init_scale=0.5, tracker = tracker, alpha = 0.2, world = world
    #         )
    
#     theta, best_fit = train_mlp_ga(
#     generations=20,
#     pop_size=20,
#     T=5.0,
#     seed=42,
#     model=model,
#     data=data,
#     core_bind=core_bind,
#     hidden=32,
#     freq_hz=1.0,
#     tracker=tracker,
#     world=world
# )
    theta, best_fit = train_mlp_cma( generations=80,
     pop_size=30,
     T=10.0,
     seed=50,
     model=model,
     data=data,
     core_bind=core_bind,
     hidden=32,
     freq_hz=1.0,
     sigma=0.5,
     init_scale=0.5,
     tracker=tracker,
     alpha=0.2,
     world=world)
    


    # theta, best_fit =  train_mlp_pso_mujoco(
    # generations=10,
    # pop_size=10,
    # T=20.0,
    # seed=0,
    # model=model,
    # data=data,
    # core_bind=core_bind,
    # hidden=32,
    # freq_hz=1.0,
    # w=0.5,
    # c1=1.5,
    # c2=1.5,
    # alpha=0.2)
    
    mj.mj_resetData(model, data)
    network = MLPController(nu=nu, hidden=32, freq_hz=1.0)
    network.set_params(theta)
    if tracker is not None:
        tracker.setup(world.spec, data)

    ctrl = Controller(
        controller_callback_function=partial(mlp_forward, network=network, core_bind = core_bind),
        tracker=tracker
    )
    
    # Assign network forward as controller callback
    ctrl.controller_callback_function = partial(mlp_forward, network=network, core_bind = core_bind)
    
    # Set controller in MuJoCo
    mj.set_mjcb_control(ctrl.set_control)
    
    # Run simulation
    simple_runner(model, data, duration=50)

    viewer.launch(model=model, data=data)


    show_xpos_history(tracker.history["xpos"][0])









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
    #world = OlympicArena()
    world = RuggedTerrainWorld()

    # Spawn robot in the world
    # Check docstring for spawn conditions
    world.spawn(robot.spec, spawn_position=[0, 0, 0.1])

    # Generate the model and data
    # These are standard parts of the simulation USE THEM AS IS, DO NOT CHANGE
    model = world.spec.compile()
    data = mj.MjData(model)
    current_pos = data.xpos[0]
    print(current_pos)
    print('okeee')

    nu = model.nu

    network = None
    T = 10
    rng = np.random.default_rng(50)
    network = MLPController(nu = nu, hidden = 32, freq_hz = 1.0)
    number_params = network.n_params
    random_theta = rng.normal(loc=0.0, scale=0.5, size=number_params)


    network.set_params(random_theta)

   


    prev_ctrl = data.ctrl.copy()
    ctrl = Controller(network.forward,prev_ctrl, T, TARGET_POS, current_pos)
    mj.set_mjcb_control(ctrl.set_control)
    print('fine')
    simple_runner(model, data, duration = 30)
    print(simple_runner)
     
    print('yell focking yea')


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
    print('heheh')

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
    """Entry point."""
    # ? ------------------------------------------------------------------ #
    # System parameters
    num_modules = 20

    # ? ------------------------------------------------------------------ #
    genotype_size = 64
    type_p_genes = RNG.random(genotype_size).astype(np.float32)
    conn_p_genes = RNG.random(genotype_size).astype(np.float32)
    rot_p_genes = RNG.random(genotype_size).astype(np.float32)

    genotype = [
        type_p_genes,
        conn_p_genes,
        rot_p_genes,
    ]

    nde = NeuralDevelopmentalEncoding(number_of_modules=num_modules)
    p_matrices = nde.forward(genotype)

    # Decode the high-probability graph
    hpd = HighProbabilityDecoder(num_modules)
    robot_graph: DiGraph[Any] = hpd.probability_matrices_to_graph(
        p_matrices[0],
        p_matrices[1],
        p_matrices[2],
    )

    # ? ------------------------------------------------------------------ #
    # Save the graph to a file
    save_graph_as_json(
        robot_graph,
        DATA / "robot_graph.json",
    )

    # ? ------------------------------------------------------------------ #
    # Print all nodes
    core = construct_mjspec_from_graph(robot_graph)

    # ? ------------------------------------------------------------------ #
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

    #experiment(robot=core, controller=ctrl, mode="simple")
    experiment2()

    # show_xpos_history(tracker.history["xpos"][0])


if __name__ == "__main__":
    main()
