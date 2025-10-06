from pyswarm import pso
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

# Type Checking
if TYPE_CHECKING:
    from networkx import DiGraph
print("helololololololo")

SEED = 42
RNG = np.random.default_rng(SEED)

# --- DATA SETUP ---
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True)

CTRL_RANGE = 1.0
BODY_COLLECTION = {}
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
    

def makeBody(num_modules: int = 20, genotype=None):
    nde = NeuralDevelopmentalEncoding(number_of_modules=num_modules)
    p_matrices = nde.forward(genotype)

    hpd = HighProbabilityDecoder(num_modules)
    robot_graph = hpd.probability_matrices_to_graph(
        p_matrices[0], p_matrices[1], p_matrices[2]
    )

    core = construct_mjspec_from_graph(robot_graph)

    return core

def makeGenotype(num_modules: int = 20):
    genotype_size = 64
    type_p = RNG.random(genotype_size).astype(np.float32)
    conn_p = RNG.random(genotype_size).astype(np.float32)
    rot_p = RNG.random(genotype_size).astype(np.float32)
    genotype = [type_p, conn_p, rot_p]
    body = makeBody(num_modules, genotype)
    

    return body, genotype



def makeNeuralNetwork(nu: int, hidden_size: int = 8):
    return NNController(nu=nu, hidden_size=hidden_size)

def makeIndividual(body,genotype):
    #Create a body
    #Spawn core to get model, for nu
    if body in BODY_COLLECTION:
        return BODY_COLLECTION[body]
    
    else:
        mj.set_mjcb_control(None)
        world = OlympicArena()
        world.spawn(body.spec, spawn_position=[2.0, 0, 0.1], correct_for_bounding_box=False)
        model = world.spec.compile() 
        nu = model.nu
        #Pass numm hinges as inputsize NN 
        nn_obj = makeNeuralNetwork(nu=nu, hidden_size=8)
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
        BODY_COLLECTION[body] = individual
        return individual
# ----------------------------
# Population Initialization
# ----------------------------
def initializePopulation(pop_size: int = 10, num_modules: int = 20):
    # add unique bodies to the population
    population = []
    all_fitness = []
    for _ in range(pop_size):
        core, genotype = makeGenotype(num_modules)

        indiv = makeIndividual(core,genotype)
        all_fitness.append(indiv["fitness"])
        population.append(indiv)
        
    return population, all_fitness



def train_mlp_pso(out_csv, generations=100, pop_size=30, T=10.0, seed=0, model=None, data=None, core_bind=None,
                  hidden=32, freq_hz=1.0, w=0.5, c1=1.5, c2=1.5):
    rng = np.random.default_rng(seed)

    # Initialize population
    population = []
    velocities = []
    fitnesses = []
    for _ in range(pop_size):
        body, genotype = makeGenotype()
        individual = makeIndividual(body, genotype)
        population.append(individual)
        fitnesses.append(individual["fitness"])
        velocities.append([np.random.uniform(-1, 1, 64).astype(np.float32) for _ in range(3)])

    # Initialize personal bests
    pbest_positions = [ind["genotype"] for ind in population]
    pbest_scores = fitnesses.copy()

    # Initialize global best
    gbest_idx = int(np.argmax(pbest_scores))
    gbest_position = pbest_positions[gbest_idx]
    gbest_score = pbest_scores[gbest_idx]
    best_candidate = population[gbest_idx]

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["generation", "best_fitness", "mean_fitness", "median_fitness"])

        for g in range(generations):
            for i in range(pop_size):
                new_velocity = []
                new_position = []
                for d in range(3):
                    r1 = rng.random(64)
                    r2 = rng.random(64)
                    inertia = w * velocities[i][d]
                    cognitive = c1 * r1 * (pbest_positions[i][d] - population[i]["genotype"][d])
                    social = c2 * r2 * (gbest_position[d] - population[i]["genotype"][d])
                    v_new = inertia + cognitive + social
                    x_new = population[i]["genotype"][d] + v_new
                    x_new = np.clip(x_new, 0, 1)
                    new_velocity.append(v_new)
                    new_position.append(x_new)

                velocities[i] = new_velocity
                new_body = makeBody(num_modules=20, genotype=new_position)
                new_individual = makeIndividual(new_body, new_position)
                new_fitness = new_individual["fitness"]

                if new_fitness > pbest_scores[i]:
                    pbest_scores[i] = new_fitness
                    pbest_positions[i] = new_position
                    population[i] = new_individual

                    if new_fitness > gbest_score:
                        gbest_score = new_fitness
                        gbest_position = new_position
                        best_candidate = new_individual

            gen_best = float(np.max(pbest_scores))
            gen_mean = float(np.mean(pbest_scores))
            gen_median = float(np.median(pbest_scores))
            writer.writerow([g, gen_best, gen_mean, gen_median])

            #if (g + 1) % max(1, generations // 10) == 0:
            #    print(f"[mlp_pso seed {seed}] Gen {g+1}/{generations}: best={gen_best:.3f} mean={gen_mean:.3f}")

    return best_candidate, gbest_score




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
                mutant = mutation(pop[a]["genotype"],pop[b]["genotype"],pop[c]["genotype"],F)
                trial = crossover(pop[i]["genotype"], mutant, CR, vector_size=3, gene_size=64)
                # Evaluate trial
                trialbody = makeBody(num_modules=20, genotype=trial)
                trialindividual = makeIndividual(trialbody,trial)
                f_trial = -trialindividual["fitness"] #fitness is negative distance to target

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
            #if (g + 1) % max(1, generations // 10) == 0:
                #print(f"[mlp_de seed {seed}] Gen {g+1}/{generations}: best={gen_best:.3f} mean={gen_mean:.3f}")

    return best_candidate, best_fit




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
    best_fit = -np.inf
    best_candidate = None

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["generation", "best_fitness", "mean_fitness", "median_fitness"])

        for g in range(generations):
            solutions = es.ask()
            fitnesses = []

            for sol in solutions:
                genotype = unflatten_genotype(np.clip(sol, 0, 1).astype(np.float32))
                body = makeBody(num_modules=20, genotype=genotype)
                individual = makeIndividual(body, genotype)
                fitness = individual["fitness"]
                fitnesses.append(-fitness)  # CMA-ES minimizes

                if fitness > best_fit:
                    best_fit = fitness
                    best_candidate = individual

            es.tell(solutions, fitnesses)

            gen_best = float(np.max([-f for f in fitnesses]))
            gen_mean = float(np.mean([-f for f in fitnesses]))
            gen_median = float(np.median([-f for f in fitnesses]))
            writer.writerow([g, gen_best, gen_mean, gen_median])

    return best_candidate, best_fit


def TrainDummyNet(nn_obj: NNController):   
    RNG = np.random.default_rng(SEED)  # No fixed seed
    fitness = RNG.random()
    return None, fitness


def Testruntime():
    print("Starting Test time...")
    global SEED 
    for i in range(50,500,50):
        print("Generations:", i,SEED)
        start_de = time.time()
        best_candidate_DE, best_fit_DE = train_mlp_de(out_csv=str(DATA / "mlp_de3_results.csv"), generations=i, pop_size=10, seed=SEED)
        end_de = time.time()
        start_pso = time.time()
        best_candidate_PSO, best_fit_PSO = train_mlp_pso(out_csv=str(DATA / "mlp_pso3_results.csv"), generations=i, pop_size=10, seed=SEED)
        end_pso = time.time()
        start_cmaes = time.time()
        best_candidate_CMAES, best_fit_CMAES = train_mlp_cmaes("results/mlp_cmaes_results.csv", generations=i, pop_size=30, seed=42)
        end_cmaes = time.time()
        print("Best DE candidate fitness:", best_fit_DE)
        print("Best PSO candidate fitness:", best_fit_PSO)
        print("Best CMAES candidate fitness:", best_fit_CMAES)
        print("DE Time:", end_de - start_de, i/(end_de - start_de))
        print("PSO Time:", end_pso - start_pso, i/(end_pso - start_pso))
        print("CMAES Time:", end_cmaes - start_cmaes, i/(end_cmaes - start_cmaes))
        
        SEED +=1




def main():
    Testruntime()
    print("Starting evolutionary algorithm...")
    start_de = time.time()
    best_candidate_DE, best_fit_DE = train_mlp_de(out_csv=str(DATA / "mlp_de3_results.csv"), generations=10, pop_size=10, seed=SEED)
    end_de = time.time()
    start_pso = time.time()
    best_candidate_PSO, best_fit_PSO = train_mlp_pso(out_csv=str(DATA / "mlp_pso3_results.csv"), generations=10, pop_size=10, seed=SEED)
    end_pso = time.time()
    print("Best DE candidate fitness:", best_fit_DE)
    print("Best PSO candidate fitness:", best_fit_PSO)
    print("DE Time:", end_de - start_de)
    print("PSO Time:", end_pso - start_pso)


if __name__ == "__main__":
    main()
"""

# Dummy fitness function that returns a random score for each particle
def dummy_fitness_function(x):
    return np.random.rand()

# Particle Swarm Optimization (PSO) with dummy fitness
def pso_dummy_fitness(dim, num_particles, max_iter, bounds, w=0.5, c1=1.5, c2=1.5):
    positions = np.random.uniform(bounds[0], bounds[1], (num_particles, dim))
    velocities = np.random.uniform(-1, 1, (num_particles, dim))

    pbest_positions = positions.copy()
    pbest_scores = np.array([dummy_fitness_function(p) for p in positions])

    gbest_index = np.argmin(pbest_scores)
    gbest_position = pbest_positions[gbest_index].copy()
    gbest_score = pbest_scores[gbest_index]

    for iteration in range(max_iter):
        for i in range(num_particles):
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            velocities[i] = (w * velocities[i] +
                             c1 * r1 * (pbest_positions[i] - positions[i]) +
                             c2 * r2 * (gbest_position - positions[i]))

            positions[i] += velocities[i]
            positions[i] = np.clip(positions[i], bounds[0], bounds[1])

            score = dummy_fitness_function(positions[i])

            if score < pbest_scores[i]:
                pbest_scores[i] = score
                pbest_positions[i] = positions[i].copy()

                if score < gbest_score:
                    gbest_score = score
                    gbest_position = positions[i].copy()

        print(f"Iteration {iteration+1}/{max_iter}, Best Dummy Score: {gbest_score}")

    return gbest_position, gbest_score

# Example usage
dim = 5  # Number of parameters defining the robot body
num_particles = 20
max_iter = 50
bounds = [-10, 10]

best_position, best_score = pso_dummy_fitness(dim, num_particles, max_iter, bounds)
print("Best Position (Body):", best_position)
print("Best Dummy Fitness Score:", best_score)

BODY = {}


def objective_function(body):
    global BODY
    if body[0] in BODY:
       return BODY[body[0]]
    else:
        RNG = np.random.default_rng()  # No fixed seed
        fitness = RNG.random()
        BODY[body[0]] = fitness
        return fitness



def multimodal_fitness(x):
    return np.sum(np.sin(x) + 0.1 * x**2)




# ----------------------------
# Main Loop
# ----------------------------
def main():
    print("Starting evolutionary algorithm...")
    #train_mlp_de(out_csv=str(DATA / "mlp_de3_results.csv"), generations=10, pop_size=10, seed=SEED)
    
    bounds = [(-10, 10), (-10, 10)]
    lb = [b[0] for b in bounds]
    ub = [b[1] for b in bounds]

    
    # Differential Evolution
    start_de = time.time()
    result_de = differential_evolution(objective_function, bounds)
    end_de = time.time()

    # Particle Swarm Optimization
    start_pso = time.time()
    pso_position, pso_score = pso(objective_function, lb, ub, swarmsize=30, maxiter=100)
    end_pso = time.time()

    print("DE:", result_de.x, result_de.fun, "Time:", end_de - start_de)
    print("PSO:", pso_position, pso_score, "Time:", end_pso - start_pso)

    
if __name__ == "__main__":
    main()
"""