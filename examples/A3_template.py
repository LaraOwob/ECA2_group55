from pyswarm import pso
from scipy.optimize import differential_evolution
import numpy as np
import time
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko




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
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
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
from pyswarm import pso
import glob

import time
import matplotlib.pyplot as plt
import pandas as pd
import os

# Type Checking
if TYPE_CHECKING:
    from networkx import DiGraph
type ViewerTypes = Literal["launcher", "video", "simple", "no_control", "frame","timestep"]

SEED = 42
RNG = np.random.default_rng(SEED)

# --- DATA SETUP ---
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True)

CTRL_RANGE = 1.0
WORKING_BODIES =set()
# ----------------------------
# Neural Network Controller
# ----------------------------
class NNController(nn.Module):
    def __init__(self, nu: int, hidden_size: int = 8):
        super().__init__()
        self.nu = nu
        self.in_dim = nu + 4   # prev_ctrl + sin/cos + delta_x, delta_y
        self.hidden = hidden_size
        self.out_dim = nu

        self.fc1 = nn.Linear(self.in_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, self.out_dim)
        self.tanh = nn.Tanh()

    def forward(self, prev_ctrl: torch.Tensor, t: float, target_pos: torch.Tensor, current_pos: torch.Tensor):
        t_tensor = torch.tensor(t, dtype=torch.float32)
        w = 2 * torch.pi
        delta = target_pos[:2] - current_pos[:2]
        
        # Use t_tensor here
        time_features = torch.stack([torch.sin(w * t_tensor), torch.cos(w * t_tensor)])
        
        obs = torch.cat([prev_ctrl, time_features, delta], dim=0)
        h = self.tanh(self.fc1(obs))
        h = self.tanh(self.fc2(h))
        y = self.tanh(self.fc3(h))
        return CTRL_RANGE * y
    
    
    
    
    
    

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
    mode: ViewerTypes = "simple",

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
    world.spawn(robot.spec, spawn_position=[0, 0, 0.1])
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
    if num_hinges < 3 or num_blocks < 3:
        print("insufficient body")
        qacc_fitness = 0
        return None, qacc_fitness
    
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
    
    # ✅ Record starting position
    start_position = np.copy(data.qpos[:2])  # store x,y
    
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
                  #  print(maxQacc)

           
     # ✅ After simulation — record end position
    end_position = np.copy(data.qpos[:2])  # final x,y

    # ✅ Compute Euclidean distance travelled
    distance = np.linalg.norm(end_position - start_position)
    
    # Return the scalar distance
    return distance, qacc_fitness



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
    distance, weirdfitness = experiment(robot=core, controller=ctrl, mode=simulation,duration = duration)
    if weirdfitness:
        print("got weird fitness")
        return core,weirdfitness
    else:
        WORKING_BODIES.add(core)
        return core, None


def makeIndividual(body,genotype,QACC_fitness):
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
        
        trained_nn, fitness = TrainDummyNet(nn_obj)
        #train_Net(model, world, nn_obj, core, target_pos, duration = 5.0, iterations=50, lr=0.01)
        individual ={
                "genotype": genotype,
                "robot_spec": body,
                "nn": trained_nn,
                "fitness": fitness
            }
       
        return individual
 
def flatten_genotype(genotype):
    """Flatten the genotype (3 vectors of size 64) into a single 192-element vector."""
    return np.concatenate(genotype)

def unflatten_genotype(flat_vector):
    """Convert a 192-element vector back into a genotype (3 vectors of size 64)."""
    return [flat_vector[0:64], flat_vector[64:128], flat_vector[128:192]]

def train_mlp_cmaes(out_csv, generations=100, pop_size=30, seed=0, sigma=0.5):
    """
    CMA-ES for evolving robot genotypes with dummy fitness evaluation.
    """
    rng = np.random.default_rng(seed)
    dim = 192  # 3 vectors of size 64
    x0 = rng.random(dim)  # Initial mean
    es = cma.CMAEvolutionStrategy(x0, sigma, {'popsize': pop_size, 'seed': seed})

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    
    
    best_fit = np.inf
    best_candidate = None

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["generation", "best_fitness", "mean_fitness", "median_fitness"])

        for g in range(generations):
            solutions = es.ask()
            fitnesses = []
            for sol in solutions:
                genotype = unflatten_genotype(np.clip(sol, 0.1, 0.9).astype(np.float32))
                body,Qacc_fitness = makeBody(num_modules=20, genotype=genotype)
                individual = makeIndividual(body, genotype,Qacc_fitness)
                fitness = individual["fitness"]
                fitnesses.append(fitness) 
                if fitness < best_fit:
                    best_fit = fitness
                    best_candidate = individual
            es.tell(solutions, fitnesses)

            # Stats for CSV
            gen_best = float(np.min(fitnesses))
            gen_mean = float(np.mean(fitnesses))
            gen_median = float(np.median(fitnesses))
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



def mutation(a,b,c,F):
    #mutant = pop[a] + F * (pop[b] - pop[c])
    mutant_genotye = [np.zeros(64)]*3
    for vector in range(3):
        for gene in range(64):
            difference = b[vector][gene] - c[vector][gene]
            mutant_genotye[vector][gene] = a[vector][gene] + F * difference
    return mutant_genotye

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
    best_idx = int(np.argmin(fits))
    best_candidate = pop[best_idx].copy()
    best_fit = float(fits[best_idx])

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["generation", "best_fitness", "mean_fitness", "median_fitness"])
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
                mutant = mutation(pop[a]["genotype"],pop[b]["genotype"],pop[c]["genotype"],F)
                trial = crossover(pop[i]["genotype"], mutant, CR, vector_size=3, gene_size=64)
                # Evaluate trial
                trialbody,Qacc_fitness = makeBody(num_modules=20, genotype=trial)
                trialindividual = makeIndividual(trialbody, trial,Qacc_fitness)
                f_trial = trialindividual["fitness"] #fitness is negative distance to target

                # If trial is better than parent than trial enters population
                if f_trial <= fits[i]:
                    pop[i] = trialindividual
                    fits[i] = f_trial

            # Retrieve stats: best fitness, mean fitness and median fitness and write to csv
            gen_best = float(np.min(fits))
            gen_mean = float(np.mean(fits))
            gen_median = float(np.median(fits))
            writer.writerow([g, gen_best, gen_mean, gen_median])

            # Update and check best candidate with best fitness
            if gen_best < best_fit:
                best_fit = gen_best
                best_candidate = pop[int(np.argmin(fits))].copy()

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





    
def main(action, generations = 100, pop_size = 30, seed = [0,1,2]):
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
            file_path =  f"./results/DE/DE{generations}_{pop_size}_seed{s}.csv"
            print(f"Starting DE algorithm with {generations} generations and population {pop_size} and seed {s}...")
            start_de = time.time()
            best_candidate_DE, best_fit_DE = train_mlp_de(out_csv=file_path, generations=generations, pop_size=pop_size, seed=s)
            end_de = time.time()
            print(f"Seed {s}")
            print("Best DE candidate fitness:", best_fit_DE)
            print("DE Time:", end_de - start_de)
            #simulate core quickly to see if it is feasible
            print(f"DE found {len(WORKING_BODIES)} bodies in total \n\n\n")
        if best_candidate_DE["robot_spec"]:
            print("find a body")
            makeBody(genotype = best_candidate_DE["genotype"], simulation = "launcher", duration = 15)
            
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
  
            
if __name__ == "__main__":
    action = "plot DE" # 'plot CMAES', 'plot DE', 'DE', 'CMAES'
    main(action)
    
