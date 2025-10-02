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
from stable_baselines3 import SAC
import gym
from gym import spaces
import os
import cma
from scipy.optimize import differential_evolution
import pyswarms as ps

# Type Checking
if TYPE_CHECKING:
    from networkx import DiGraph
print("helololololololo")


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

CTRL_RANGE = 1.0

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

# ----------------------------
# Body Creation
# ----------------------------
def makeBody(num_modules: int = 20):
    genotype_size = 64
    type_p = RNG.random(genotype_size).astype(np.float32)
    conn_p = RNG.random(genotype_size).astype(np.float32)
    rot_p = RNG.random(genotype_size).astype(np.float32)
    genotype = [type_p, conn_p, rot_p]

    nde = NeuralDevelopmentalEncoding(number_of_modules=num_modules)
    p_matrices = nde.forward(genotype)

    hpd = HighProbabilityDecoder(num_modules)
    robot_graph = hpd.probability_matrices_to_graph(
        p_matrices[0], p_matrices[1], p_matrices[2]
    )

    core = construct_mjspec_from_graph(robot_graph)



    return core, genotype


# ----------------------------
# Brain Creation
# ----------------------------
def makeNeuralNetwork(nu: int, hidden_size: int = 8):
    return NNController(nu=nu, hidden_size=hidden_size)


# ----------------------------
# Fitness Evaluation
# ----------------------------
# def evaluateFitness(
#     model: mj.MjModel,
#     world: OlympicArena,
#     nn_obj: NNController,
#     core: object,
#     target_pos: np.ndarray = np.array([1.0, 0.0], dtype=np.float32),
#     duration: float = 5.0
# ) -> float:
#     """
#     Evaluate the fitness of a neural network controller by simulating the robot
#     and measuring its final distance to the target.
#     """
#     # --- Setup world ---
   
#     data = mj.MjData(model)
#     mj.mj_resetData(model, data)

#     # --- Setup tracker ---
#     tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")
#     tracker.setup(world.spec, data)

#     # --- NN control callback ---
#     def nn_control_fn(m: mj.MjModel, d: mj.MjData):
#         prev_ctrl = torch.zeros(nn_obj.nu)
#         current_pos = torch.tensor(d.geom_xpos[0], dtype=torch.float32)
#         target_tensor = torch.tensor(target_pos, dtype=torch.float32)
#         action = nn_obj(prev_ctrl=prev_ctrl, t=d.time, target_pos=target_tensor, current_pos=current_pos)
#         d.ctrl[:] = action.detach().numpy()

#         # record position
#         tracker.history["xpos"].append(current_pos)

#     mj.set_mjcb_control(nn_control_fn)

#     # --- Run simulation ---
#     simple_runner(model, data, duration=duration)

#     # --- Compute fitness ---
#     final_pos = np.array(tracker.history["xpos"][-1])[:2]
#     fitness = -np.linalg.norm(final_pos - target_pos[:2])

#     return fitness

def nn_obj_control(nn_obj: NNController, model: mj.MjModel, data: mj.MjData, target_pos: np.ndarray):
    # Previous control (just zeros for simplicity or could be tracked)
    prev_ctrl = torch.zeros(nn_obj.nu, dtype=torch.float32)

    # Current position (assume first geom of robot)
    current_pos = torch.tensor(data.geom_xpos[0], dtype=torch.float32)

    # Forward pass through NN
    action = nn_obj(prev_ctrl=prev_ctrl, t=data.time, 
                    target_pos=torch.tensor(target_pos, dtype=torch.float32),
                    current_pos=current_pos)
    return action.detach().numpy()



def evaluateFitness(
    model: mj.MjModel,
    world: OlympicArena,
    nn_obj: NNController,
    core: object,
    # ctrl: Controller
    target_pos: np.ndarray = np.array([1.0, 0.0], dtype=np.float32),
    duration: float = 5.0
):
    # data = mj.MjData(model)
    # mj.mj_resetData(model, data)
    # ctrl.callback_function = nn_obj.forward()
    # mj.set_mjcb_control(ctrl.set_control)
    # simple_runner(model, data, duration = duration)
    


    mujoco_type_to_find = mj.mjtObj.mjOBJ_GEOM
    name_to_bind = "core"
    tracker = Tracker(mujoco_obj_to_find=mujoco_type_to_find,name_to_bind=name_to_bind,)
    ctrl = Controller(controller_callback_function=lambda m, d: nn_obj_control(nn_obj, m, d, target_pos),tracker=tracker)
    data = mj.MjData(model)
    mj.mj_resetData(model, data)

    # Pass the model and data to the tracker
    if ctrl.tracker is not None:
        ctrl.tracker.setup(world.spec, data)
    mj.set_mjcb_control(lambda m, d: ctrl.set_control(m, d))
   
    first_geom = list(tracker.history["xpos"].keys())[0]  
    xpos_history = tracker.history["xpos"][first_geom]   
    final_pos = xpos_history[-1][:2] 
    fitness = -np.linalg.norm(final_pos - target_pos[:2])

    return fitness




class OlympicEnv(gym.Env):
    def __init__(self, robot,world):
        super().__init__()
        # Init world + robot
        mj.set_mjcb_control(None)
        self.world = world

        # Build MuJoCo model/data
        self.model = self.world.spec.compile()
        self.data = mj.MjData(self.model)

        # Action/obs spaces
        self.n_actions = self.model.nu
        self.n_obs = self.model.nq + self.model.nv
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.n_actions,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_obs,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mj.mj_resetData(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        self.data.ctrl[:] = action
        mj.mj_step(self.model, self.data)
        obs = self._get_obs()

        reward = self._compute_reward(obs, action)
        done = self._check_done()
        truncated = False

        return obs, reward, done, truncated, {}

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)

    def _compute_reward(self, obs, action):
        # Placeholder: define task-specific reward
        return -np.linalg.norm(obs)

    def _check_done(self):
        # Placeholder: define when episode ends
        return False








def train_Net(
    model: mj.MjModel,
    world: OlympicArena,
    nn_obj: NNController,
    core: object,
    target_pos: np.ndarray,
    duration: float,
    iterations: int = 50,
    lr: float = 0.01
):
    """Simple finite-difference optimizer for NN weights."""
    for name, param in nn_obj.named_parameters():
            print(name, param.shape)
            print(param) 
    nnModel = SAC("MlpPolicy", env=OlympicEnv(core,world), verbose=1)
    nnModel.learn(total_timesteps=10000)
    
    print("did that run")
    
    """
    for _ in range(iterations):
        # Evaluate current fitness
        
        baseline = evaluateFitness(model, world, nn_obj, core, target_pos, duration)

        # For each weight, perturb slightly and keep if fitness improves
        with torch.no_grad():
            for param in nn_obj.parameters():
                noise = 0.01 * torch.randn_like(param)
                param += noise
                new_fitness = evaluateFitness(model, world, nn_obj, core, target_pos, duration)
                if new_fitness < baseline:
                    # Revert if worse
                    param -= noise
                else:
                    baseline = new_fitness
    print(baseline)
    """
    return nn_obj#, baseline


def TrainDummyNet(nn_obj: NNController):    
    RNG = np.random.default_rng(SEED)
    fitness = RNG.random()
    return None, fitness


def mutation(a,b,c,rate,F):
    #mutant = pop[a] + F * (pop[b] - pop[c])
    individual = []
    for vector in range(3):
        for gene in range(64):
            individual[vector][gene] = a[vector][gene] + F * (b[vector][gene] - c[vector][gene])
    return individual

def crossover(parent, mutant, CR, vector_size=3, gene_size=64):
    """
    Make a crossover between the parent and the mutant per vector
    so that each vector has at least one gene from the mutant
    
    """
    rng = np.random.default_rng(SEED)
    trial = []
    for i in range(vector_size):
        cross = rng.random(gene_size) < CR
        cross[rng.integers(0, gene_size)] = True 
        trial_vector = np.where(cross, mutant[i], parent[i])
        trial[i] = trial_vector
    
    return trial


#  Differential Evolution on MLP (1b)
def train_mlp_de(out_csv, generations=100, pop_size=30, T=10.0, seed=0, model=None, data=None, core_bind=None,
                 hidden=32, freq_hz=1.0, F=0.7, CR=0.9, init_scale=0.5, alpha=0.2):
    """
    Simple, Differential Evolution for MLP parameters.
    """
    rng = np.random.default_rng(seed)

    # Initialize population by sampling normal distribution with mean 0 and stdv of init_scale
    pop = initializePopulation(pop_size=pop_size, num_modules=20)
    #Initialize empty numby array with length number of candidates
    fits = np.empty(pop_size, dtype=np.float64)

    # Calculate fitness for each candidate, pop[i] = candidate
    for i in range(pop_size):
        #evaluate_mlp_headless(pop[i], model, data, core_bind, T=T, hidden=hidden, freq_hz=freq_hz, alpha=alpha)
        fits[i] = pop[i]["fitness"] 

    #Finds best fitnesses
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    best_idx = int(np.argmax(fits))
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
                
                #mutant = pop[a] + F * (pop[b] - pop[c])
                mutant = mutation(pop[a]["genotype"],pop[b]["genotype"],pop[c]["genotype"],F)
                # Crossover (binomial), between parent and mutant
                #Creates list of length weights whether each gene will be changed or stays like parent. Returns FAlse or True list
                cross = rng.random(number_params) < CR
                #At least one random gene will be open for crossover with mutant
                cross[rng.integers(0, number_params)] = True 
                #The genes which were chosen by cross validation proability are adjusted
                trial = np.where(cross, mutant, pop[i])

                # Evaluate trial
                loss = evaluate_mlp_headless(trial, model, data, core_bind, T=T, hidden=hidden, freq_hz=freq_hz, alpha=alpha)
                f_trial = -loss

                # If trial is better than parent than trial enters population
                if f_trial >= fits[i]:
                    pop[i] = trial
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
            if (g + 1) % max(1, generations // 10) == 0:
                print(f"[mlp_de seed {seed}] Gen {g+1}/{generations}: best={gen_best:.3f} mean={gen_mean:.3f}")

    return best_candidate, best_fit










#DE
def DeMethod(fitnessscore,dim):
    result = differential_evolution(fitness, bounds=[(-5, 5)] * dim, maxiter=50)
    print("Best solution:", result.x)
    return result.x


#CMAES
def CMAESmethod(fitness,dim):
    es = cma.CMAEvolutionStrategy(dim * [0], 0.5)  # mean, sigma
    es.optimize(fitness, iterations=50)
    print("Best solution:", es.result.xbest)
    return es.result.xbest


def PSOmethod(fitness,dim):
    
    options = {'c1': 1.5, 'c2': 1.5, 'w': 0.7}
    optimizer = ps.single.GlobalBestPSO(n_particles=20, dimensions=dim, options=options)
    best_cost, best_pos = optimizer.optimize(fitness, iters=50)
    print("Best PSO solution:", best_pos)

def MapElites(fitness, dim):
    # Archive: 2D map based on (mean vec1, mean vec2)
    archive = GridArchive(
    dims=[10, 10],  # 10x10 grid
    ranges=[(-1,1), (-1,1)],  # descriptors
    learning_rate=1.0,
    )

    emitter = GaussianEmitter(archive, x0=np.zeros(dim), sigma=0.1)
    opt = Optimizer([emitter])

    for gen in range(100):
        sols = opt.ask()
        fits, descs = zip(*[evaluate(s) for s in sols])
        opt.tell(sols, fits, descs)

    best = archive._solutions[np.argmax(archive.f)]
    print("Best MAP-Elites fitness:", np.max(archive.f))



def evaluate(x):
    indiv = [x[:vec_dim], x[vec_dim:2*vec_dim], x[2*vec_dim:]]
    fit = evaluateFitness(model, world, nn_obj, core, target_pos, duration)
    desc = [np.mean(indiv[0]), np.mean(indiv[1])]
    return fit, desc

# ----------------------------
# Population Initialization
# ----------------------------
def initializePopulation(pop_size: int = 10, num_modules: int = 20):
    population = []
    for _ in range(pop_size):
        #Create a body
        core, genotype = makeBody(num_modules)
        #Spawn core to get model, for nu
        mj.set_mjcb_control(None)
        world = OlympicArena()
        world.spawn(core.spec, spawn_position=[2.0, 0, 0.1], correct_for_bounding_box=False)
        model = world.spec.compile() 
        nu = model.nu
        #Pass numm hinges as inputsize NN 
        nn_obj = makeNeuralNetwork(nu=nu, hidden_size=8)
        #Set target_position
        target_pos = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        # Train and evaluate
        trained_nn, fitness = TrainDummyNet(nn_obj)
        #train_Net(model, world, nn_obj, core, target_pos, duration = 5.0, iterations=50, lr=0.01)

        population.append({
            "genotype": genotype,
            "robot_spec": core,
            "nn": trained_nn,
            "fitness": fitness
        })
        print("Individual fitness:", fitness)
    return population


# ----------------------------
# Evolutionary Algorithm
# ----------------------------
def evolvePopulation(population, elite_fraction=0.2, mutation_rate=0.1):
    population = sorted(population, key=lambda x: x["fitness"], reverse=True)
    n_elite = max(1, int(len(population) * elite_fraction))
    new_population = population[:n_elite]

    while len(new_population) < len(population):
        parent1, parent2 = RNG.choice(population[:n_elite], 2, replace=False)
        child_genotype = []
        for g1, g2 in zip(parent1["genotype"], parent2["genotype"]):
            mask = RNG.random(g1.shape) < 0.5
            child_gene = np.where(mask, g1, g2)
            mutation = RNG.normal(0, mutation_rate, size=g1.shape)
            child_gene += mutation
            child_genotype.append(child_gene)
        nde = NeuralDevelopmentalEncoding(number_of_modules=20)
        p_matrices = nde.forward(child_genotype)
        hpd = HighProbabilityDecoder(20)
        child_graph = hpd.probability_matrices_to_graph(p_matrices[0], p_matrices[1], p_matrices[2])
        child_robot_spec = construct_mjspec_from_graph(child_graph)
        child_nn = makeNeuralNetwork(child_robot_spec)
        trained_nn, fitness = trainNet(child_nn, child_robot_spec)
        new_population.append({
            "genotype": child_genotype,
            "robot_spec": child_robot_spec,
            "nn": child_nn,
            "fitness": fitness
        })
    return new_population

# ----------------------------
# Main Loop
# ----------------------------
def main():
    population = initializePopulation(pop_size=10)
    for gen in range(5):
        population = evolvePopulation(population)
        best = max(population, key=lambda x: x["fitness"])
        print(f"Generation {gen}: Best fitness = {best['fitness']:.3f}")

if __name__ == "__main__":
    main()


# # class NN:
# #     def __init__(self, nu: int, hidden_size: int = 8):
# #         self.nu = nu
# #         self.hidden = hidden_size
# #         self.in_dim = nu + 4
# #         self.out_dim = nu
# #         #self.freq_hz = freq_hz
# #         # Parameter shapes
# #         self.W1_shape = (self.in_dim, hidden_size)
# #         self.b1_shape = (hidden_size,)
# #         self.W2_shape = (hidden_size, self.out_dim)
# #         self.b2_shape = (self.out_dim,)
# #         self.n_params = (
# #             self.in_dim*self.hidden +
# #             self.hidden +
# #             self.hidden*self.out_dim +
# #             self.out_dim
# #         )


# import torch
# import torch.nn as nn
# import torch.optim as optim

# CTRL_RANGE = 1.0

# class NNController(nn.Module):
#     def __init__(self, nu: int, hidden_size: int = 8):
#         """
#         Simple 2-hidden-layer MLP controller.
#         Inputs: prev_ctrl + sin/cos + delta to target
#         Outputs: actuator commands
#         """
#         super().__init__()
#         self.nu = nu
#         self.in_dim = nu + 4   # prev_ctrl + sin(wt) + cos(wt) + delta_x,delta_y
#         self.hidden = hidden_size
#         self.out_dim = nu

#         self.fc1 = nn.Linear(self.in_dim, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, hidden_size)
#         self.fc3 = nn.Linear(hidden_size, self.out_dim)
#         self.tanh = nn.Tanh()

#     def forward(self, prev_ctrl: torch.Tensor, t: float, target_pos: torch.Tensor, current_pos: torch.Tensor):
#         w = 2 * torch.pi  # frequency can be parameterized
#         delta = target_pos[:2] - current_pos[:2]
#         obs = torch.cat([prev_ctrl, torch.tensor([torch.sin(w*t), torch.cos(w*t)], dtype=torch.float32), delta], dim=0)
#         h = self.tanh(self.fc1(obs))
#         h = self.tanh(self.fc2(h))
#         y = self.tanh(self.fc3(h))
#         return CTRL_RANGE * y
    


# def makeNeuralNetwork(input_size: int, output_size: int):

    
#     """
#     create a net that takes in the positions of the joints and outputs
#     new positions for the joints
#     with the hidden layers etc predefined
  
    
#     """
#     return 0    


# def makeBody():
#     """
#     create a body using the nde and hpd classes
#     returns the body a genotype of 3 vectors of 64 floats each
#     Here to make a body evolutionary algorithms are used
#     so DE for example
#     """
#     pass



# def trainNet(nn_obj):

#     """
#     Takes input of an NN and trains it to the max fitness
#     and returns the trained net and fitness 
    
#     """
#     return 0.0, 0.0

# def trainNet(nn_obj: nn.Module, prev_ctrl: torch.Tensor, target_pos: torch.Tensor,
#              current_pos: torch.Tensor, lr: float = 0.01, epochs: int = 200):
#     """
#     Train the neural network to maximize fitness (minimize distance to target)
#     using Adam optimizer.
#     """
#     optimizer = optim.Adam(nn_obj.parameters(), lr=lr)
#     criterion = nn.MSELoss()

#     for epoch in range(epochs):
#         optimizer.zero_grad()
#         outputs = nn_obj(prev_ctrl, t=0.0, target_pos=target_pos, current_pos=current_pos)
#         loss = criterion(outputs, target_pos)  # simple fitness: match target
#         loss.backward()
#         optimizer.step()

#     # Compute final fitness
#     with torch.no_grad():
#         final_outputs = nn_obj(prev_ctrl, t=0.0, target_pos=target_pos, current_pos=current_pos)
#         fitness = -criterion(final_outputs, target_pos).item()  # negative because higher fitness = better

#     return nn_obj, fitness

# def initializePopulation(pop_size: int):
#     """
#     initializes a population of pop_size individuals
#     each individual is a tuple of (genotype, neural network)
#     genotype is a list of 3 vectors of 64 floats each
#     neural network is a neural network that takes in the positions of the joints and outputs
#     new positions for the joints
#     with the hidden layers etc predefined
    
#     returns a list of individuals
#     """
#     population = []
#     for _ in range(pop_size):
#         genotype = makeBody() #unique bodies
#         neural_net = makeNeuralNetwork(input_size=genotype, output_size=genotype)
#         trained_net, fitness = trainNet(neural_net)
#         population.append((genotype, neural_net, fitness))
#     return population



# def makePopulation(): #start of evolutionary algorithm
    
#     """
#     This is the first step of the evolutionary algorithm.
#     Create a population of genotypes (bodies).
#     uses the function makebody and makeneuralnetwork
#     then trains the neural network in the function train net
#     and returns the fitness of each individual in the population
#     and the genotype and neural network of each individual in the population
    
#     This is done for each individual in the population
#     and the population is return with fitness, genotype and neural network
    
    
#     """
#     pass




























# def show_xpos_history(history: list[float]) -> None:
#     # Convert list of [x,y,z] positions to numpy array
#     pos_data = np.array(history)

#     # Create figure and axis
#     plt.figure(figsize=(10, 6))

#     # Plot x,y trajectory
#     plt.plot(pos_data[:, 0], pos_data[:, 1], "b-", label="Path")
#     plt.plot(pos_data[0, 0], pos_data[0, 1], "go", label="Start")
#     plt.plot(pos_data[-1, 0], pos_data[-1, 1], "ro", label="End")
#     plt.plot(0, 0, "kx", label="Origin")

#     # Add labels and title
#     plt.xlabel("X Position")
#     plt.ylabel("Y Position")
#     plt.title("Robot Path in XY Plane")
#     plt.legend()
#     plt.grid(visible=True)

#     # Set equal aspect ratio and center at (0,0)
#     plt.axis("equal")

#     # Show results
#     plt.show()


# def random_move(
#     model: mj.MjModel,
#     data: mj.MjData,
# ) -> npt.NDArray[np.float64]:
#     # Get the number of joints
#     num_joints = model.nu

#     # Hinges take values between -pi/2 and pi/2
#     hinge_range = np.pi / 2
#     return RNG.uniform(
#         low=-hinge_range,  # -pi/2
#         high=hinge_range,  # pi/2
#         size=num_joints,
#     ).astype(np.float64)


# def nn_controller(
#     model: mj.MjModel,
#     data: mj.MjData,
# ) -> npt.NDArray[np.float64]:
#     # Simple 3-layer neural network
#     input_size = len(data.qpos)
#     hidden_size = 8
#     output_size = model.nu

#     # Initialize the networks weights randomly
#     # Normally, you would use the genes of an individual as the weights,
#     # Here we set them randomly for simplicity.
#     w1 = RNG.normal(loc=0.0138, scale=0.5, size=(input_size, hidden_size))
#     w2 = RNG.normal(loc=0.0138, scale=0.5, size=(hidden_size, hidden_size))
#     w3 = RNG.normal(loc=0.0138, scale=0.5, size=(hidden_size, output_size))

#     # Get inputs, in this case the positions of the actuator motors (hinges)
#     inputs = data.qpos

#     # Run the inputs through the lays of the network.
#     layer1 = np.tanh(np.dot(inputs, w1))
#     layer2 = np.tanh(np.dot(layer1, w2))
#     outputs = np.tanh(np.dot(layer2, w3))

#     # Scale the outputs
#     return outputs * np.pi


# def experiment(
#     robot: Any,
#     controller: Controller,
#     duration: int = 15,
#     mode: ViewerTypes = "viewer",
# ) -> None:
#     """Run the simulation with random movements."""
#     # ==================================================================== #
#     # Initialise controller to controller to None, always in the beginning.
#     mj.set_mjcb_control(None)  # DO NOT REMOVE

#     # Initialise world
#     # Import environments from ariel.simulation.environments
#     world = OlympicArena()

#     # Spawn robot in the world
#     # Check docstring for spawn conditions
#     world.spawn(robot.spec, spawn_position=[0, 0, 0.1])

#     # Generate the model and data
#     # These are standard parts of the simulation USE THEM AS IS, DO NOT CHANGE
#     model = world.spec.compile()
#     data = mj.MjData(model)

#     # Reset state and time of simulation
#     mj.mj_resetData(model, data)

#     # Pass the model and data to the tracker
#     if controller.tracker is not None:
#         controller.tracker.setup(world.spec, data)

#     # Set the control callback function
#     # This is called every time step to get the next action.
#     args: list[Any] = []  # IF YOU NEED MORE ARGUMENTS ADD THEM HERE!
#     kwargs: dict[Any, Any] = {}  # IF YOU NEED MORE ARGUMENTS ADD THEM HERE!

#     mj.set_mjcb_control(
#         lambda m, d: controller.set_control(m, d, *args, **kwargs),
#     )

#     # ------------------------------------------------------------------ #
#     match mode:
#         case "simple":
#             # This disables visualisation (fastest option)
#             simple_runner(
#                 model,
#                 data,
#                 duration=duration,
#             )
#         case "frame":
#             # Render a single frame (for debugging)
#             save_path = str(DATA / "robot.png")
#             single_frame_renderer(model, data, save=True, save_path=save_path)
#         case "video":
#             # This records a video of the simulation
#             path_to_video_folder = str(DATA / "videos")
#             video_recorder = VideoRecorder(output_folder=path_to_video_folder)

#             # Render with video recorder
#             video_renderer(
#                 model,
#                 data,
#                 duration=duration,
#                 video_recorder=video_recorder,
#             )
#         case "launcher":
#             # This opens a liver viewer of the simulation
#             viewer.launch(
#                 model=model,
#                 data=data,
#             )
#         case "no_control":
#             # If mj.set_mjcb_control(None), you can control the limbs manually.
#             mj.set_mjcb_control(None)
#             viewer.launch(
#                 model=model,
#                 data=data,
#             )
#     # ==================================================================== #



# def main() -> None:
#     """Entry point."""
#     # ? ------------------------------------------------------------------ #
#     # System parameters
#     num_modules = 20

#     # ? ------------------------------------------------------------------ #
#     genotype_size = 64
#     type_p_genes = RNG.random(genotype_size).astype(np.float32)
#     conn_p_genes = RNG.random(genotype_size).astype(np.float32)
#     rot_p_genes = RNG.random(genotype_size).astype(np.float32)

#     genotype = [
#         type_p_genes,
#         conn_p_genes,
#         rot_p_genes,
#     ] #evolution on this 

#     nde = NeuralDevelopmentalEncoding(number_of_modules=num_modules)
#     p_matrices = nde.forward(genotype) #not the p_matrices

#     # Decode the high-probability graph
#     hpd = HighProbabilityDecoder(num_modules)
#     robot_graph: DiGraph[Any] = hpd.probability_matrices_to_graph(
#         p_matrices[0],
#         p_matrices[1],
#         p_matrices[2],
#     )

#     # ? ------------------------------------------------------------------ #
#     # Save the graph to a file
#     save_graph_as_json(
#         robot_graph,
#         DATA / "robot_graph.json",
#     )

#     # ? ------------------------------------------------------------------ #
#     # Print all nodes
#     core = construct_mjspec_from_graph(robot_graph) # this is the gecko but now called core

#     # ? ------------------------------------------------------------------ #
#     mujoco_type_to_find = mj.mjtObj.mjOBJ_GEOM
#     name_to_bind = "core"
#     tracker = Tracker(
#         mujoco_obj_to_find=mujoco_type_to_find,
#         name_to_bind=name_to_bind,
#     )

#     # ? ------------------------------------------------------------------ #
#     # Simulate the robot
#     ctrl = Controller(
#         controller_callback_function=nn_controller,
#         #controller_callback_function=random_move,
#         tracker=tracker,
#     )

#     experiment(robot=core, controller=ctrl, mode="launcher")

#     show_xpos_history(tracker.history["xpos"][0])
#     print(tracker.history["xpos"][0])


# if __name__ == "__main__":
#     main()
