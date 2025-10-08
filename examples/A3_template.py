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
BODY_COLLECTION = {}
BEST_FITNESS= 0
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

    # Count blocks (bodies)
    # Exclude the world body (index 0) if you only want the robot parts
    num_blocks = model.nbody - 1  

    print("Number of hinge joints:", num_hinges)
    print("Number of blocks (bodies) used:", num_blocks)
    if num_hinges < 3 or num_blocks < 3:
        print("insufficient body")
        return None
    
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
                    unstable = True
                    break
                #if np.any(np.abs(data.qacc) > maxQacc):
                 #   maxQacc =max(np.abs(data.qacc))
                  #  print(maxQacc)

            if unstable:
                # Optionally: return early with zero distance or a flag
                return None
     # ✅ After simulation — record end position
    end_position = np.copy(data.qpos[:2])  # final x,y

    # ✅ Compute Euclidean distance travelled
    distance = np.linalg.norm(end_position - start_position)

    # Return the scalar distance
    return distance



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
    distance = experiment(robot=core, controller=ctrl, mode=simulation,duration = duration)
    if distance ==None:
        return None
    else:
        WORKING_BODIES.add(core)
        return core


def makeIndividual(body,genotype):
    #Create a body
    #Spawn core to get model, for nu
    if body == None:
        individual ={
                "genotype": genotype,
                "robot_spec": body,
                "nn": None,
                "fitness": 1.5
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
    
    
def is_genotype_valid(genotype):
    # e.g., check number of hinges
    num_modules =20
    nde = NeuralDevelopmentalEncoding(number_of_modules=num_modules)
    p_matrices = nde.forward(genotype)
    hpd = HighProbabilityDecoder(num_modules)
    robot_graph = hpd.probability_matrices_to_graph(p_matrices[0], p_matrices[1], p_matrices[2])
    no_hinges = sum(1 for joint in robot_graph['joints'] if joint['type'] == 'hinge')
    no_bodies = len(robot_graph['bodies'])
    if no_hinges == 0:
        return False
    if no_bodies< 3:
        return False
    return True


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
            print(f"this is generation {g} \n\n\n")
            for sol in solutions:
                genotype = unflatten_genotype(np.clip(sol, 0.1, 0.9).astype(np.float32))
                body = makeBody(num_modules=20, genotype=genotype)
                individual = makeIndividual(body, genotype)
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
    print("finished")
    return best_candidate, best_fit


def TrainDummyNet(nn_obj): #(nn_obj: NNController):   
    RNG = np.random.default_rng(SEED)  # No fixed seed
    fitness = RNG.random()
    return None, fitness



def plot(file_path,output_path):
        # --- Load your results file ---
    # Works for both .csv and .xlsx

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
    
    
    
def main(action, generations = 100, pop_size = 150):
    file_path =  f"MLP_CMAES_{generations}_{pop_size}_results.csv"
    if action == "algorithm":
        
        print("Starting evolutionary algorithm...")
        start_cmaes = time.time()
        best_candidate_CMAES, best_fit_CMAES = train_mlp_cmaes(out_csv=str(DATA /file_path ), generations=generations, pop_size=pop_size, seed=SEED)
        end_cmaes = time.time()
        print("Best CMAE candidate fitness:", best_fit_CMAES)
        print("CMAES Time:", end_cmaes - start_cmaes)
        #simulate core quickly to see if it is feasible
        print(f"CMAES found {len(WORKING_BODIES)} bodies in total")
        if best_candidate_CMAES["robot_spec"]:
            print("find a body")
            makeBody(genotype = best_candidate_CMAES["genotype"], simulation = "launcher", duration = 15)
    if action == "plot":
        output_path = "plot" + file_path[:-4] + ".png"
        plot(str(DATA / file_path),output_path)
            
            
if __name__ == "__main__":
    action = "plot" # 'plot' or 'algorithm'
    main(action)
    
