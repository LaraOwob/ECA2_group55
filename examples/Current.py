# Third-party libraries
import numpy as np
import mujoco
from mujoco import viewer
import matplotlib.pyplot as plt

# Local libraries
from ariel.utils.renderers import video_renderer
from ariel.utils.video_recorder import VideoRecorder
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld

# import prebuilt robot phenotypes
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
import torch
import torch.nn as nn
import evotorch as et

from deap import base, creator, tools

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
# from ariel.simulation.tasks.targeted_locomotion import distance_to_target




# Keep track of data / history
HISTORY = []
# === Neural Network ===
class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
            nn.Tanh() 
        )

    def forward(self, x):
        return self.model(x)


def set_model_parameters_from_flat(model, flat_tensor):
    pointer = 0
    for p in model.parameters():
        numel = p.numel()
        chunk = flat_tensor[pointer: pointer + numel]
        chunk = chunk.view_as(p)
        with torch.no_grad():
            p.copy_(chunk)
        pointer += numel

def controller(model, data, to_track, solution_flat):
    inputs = torch.tensor(data.qpos, dtype=torch.float32).unsqueeze(0)
  
  # THere is also inbuilt function set_parameters, but that did not work for me
    set_model_parameters_from_flat(model, solution_flat)

    with torch.no_grad():
        output = model(inputs).squeeze()

    print(output)
    data.ctrl = np.clip(output * (np.pi / 2), -np.pi/2, np.pi/2)
    HISTORY.append(to_track[0].xpos.copy())



def fitness(net, model, solution):
        # Initialize net with random weigths
        set_model_parameters_from_flat(net, solution)
        data_eval = mujoco.MjData(model)
        start_x = data_eval.qpos[0]
        for _ in range(200):
            obs = torch.tensor(data_eval.qpos, dtype=torch.float32).unsqueeze(0)
            action = net(obs).detach().flatten().tolist()
            data_eval.ctrl[:] = [max(min(a * (np.pi/2), np.pi/2), -np.pi/2) for a in action]

            mujoco.mj_step(model, data_eval)
        return data_eval.qpos[0] - start_x

def search_for_weights(model, net, generations=10, popsize=20):
    """
    Optimise the parameters of `net` using CMA-ES and return the best solution.

    Parameters
    ----------
    model : mujoco.MjModel
        The compiled MuJoCo model of your world.
    net : Net
        Your neural network (only its shape is used to count params).
    fitness_fn : callable
        Function with signature  fitness(solution_flat: np.ndarray) -> float
    generations : int
        Number of CMA-ES generations to run.
    popsize : int
        Population size per generation.

    Returns
    -------
    np.ndarray
        Flat vector of parameters giving the best fitness.
    """
    # --- count how many parameters the NN has ---
    num_params = sum(p.numel() for p in net.parameters())
    random_sol = torch.randn(num_params)

    fitness_function = fitness(net, model, random_sol)
    print(fitness_function)
    print('HAHAHHAHA')

    # --- define the optimisation problem ---
    problem = et.Problem(
        "max",                      # we want to maximise fitness
        fitness_function,
        solution_length=num_params
    )

    # --- create CMA-ES searcher ---
    searcher = et.CMAES(problem, popsize=popsize)

    # --- run the search ---
    for gen in range(generations):
        searcher.step()
        print(
            f"Generation {gen+1}/{generations} | "
            f"best fitness: {searcher.status['best_fitness']:.4f}"
        )

    # --- return the best solution ---
    return searcher.status["best_solution"]

# def fitness():
#     net = Net()
#     et.nn.set_parameters(net, solution)  # load genome into model
#     for i in time:
#         controller(net,data,totrack,solution)
#     #result is how fast it went from a to b
#     return result
        

# def algorithm1():
#     #algorithm that just optimizes the weights of the NN
#     # Define search space (genome length = number of parameters in Net)
#     dummy_net = Net()
#     num_params = sum(p.numel() for p in dummy_net.parameters())

#     problem = et.Problem(
#         "max",         # maximize fitness
#         fitness,
#         solution_length=num_params
#     )

#     # Use CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
#     searcher = et.CMAES(problem, popsize=20)

#     # Run evolution
#     for gen in range(10):
#         searcher.step()
#         print(f"Generation {gen+1}, best fitness: {searcher.status['best_fitness']:.4f}")




# # Define genome: [num_hidden_layers, size_layer1, size_layer2, activation]
# def random_architecture():
#     num_layers = random.randint(1, 3)
#     layers = [random.randint(4, 32) for _ in range(num_layers)]
#     activation = random.choice(["relu", "tanh"])
#     return [num_layers] + layers + [activation]

# # Build a network from genome
# def build_net(genome, input_size, output_size):
#     num_layers = genome[0]
#     layers = []
#     in_features = input_size
    
#     for i in range(num_layers):
#         out_features = genome[i+1]
#         layers.append(nn.Linear(in_features, out_features))
#         if genome[-1] == "relu":
#             layers.append(nn.ReLU())
#         else:
#             layers.append(nn.Tanh())
#         in_features = out_features
    
#     layers.append(nn.Linear(in_features, output_size))
#     return nn.Sequential(*layers)

# # Fitness function: evaluate NN performance on task
# def evaluate(genome,data):
#     inputs = len(data.qpos)
#     outputs = 0  #number of hinges
#     net = build_net(genome, input_size=inputs, output_size=outputs)
#     controller()
#     # Run robot simulator here...
#     fitness = random.random()  # fitness calculator 
#     return (fitness,)

# def algorithm2():
#     # --- DEAP Setup ---
#     creator.create("FitnessMax", base.Fitness, weights=(1.0,))
#     creator.create("Individual", list, fitness=creator.FitnessMax)

#     toolbox = base.Toolbox()
#     toolbox.register("individual", tools.initIterate, creator.Individual, random_architecture)
#     toolbox.register("population", tools.initRepeat, list, toolbox.individual)
#     toolbox.register("evaluate", evaluate)
#     toolbox.register("mate", tools.cxTwoPoint)
#     toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
#     toolbox.register("select", tools.selTournament, tournsize=3)

#     # Evolve
#     pop = toolbox.population(n=10)
#     for gen in range(5):
#         offspring = toolbox.select(pop, len(pop))
#         offspring = list(map(toolbox.clone, offspring))
#         for child1, child2 in zip(offspring[::2], offspring[1::2]):
#             if random.random() < 0.5:
#                 toolbox.mate(child1, child2)
#                 del child1.fitness.values, child2.fitness.values
#         for mutant in offspring:
#             if random.random() < 0.2:
#                 toolbox.mutate(mutant)
#                 del mutant.fitness.values
#         invalid = [ind for ind in offspring if not ind.fitness.valid]
#         fitnesses = map(toolbox.evaluate, invalid)
#         for ind, fit in zip(invalid, fitnesses):
#             ind.fitness.values = fit
#         pop[:] = offspring



# def random_move(model, data, to_track) -> None:
#     """Generate random movements for the robot's joints.
    
#     The mujoco.set_mjcb_control() function will always give 
#     model and data as inputs to the function. Even if you don't use them,
#     you need to have them as inputs.

#     Parameters
#     ----------

#     model : mujoco.MjModel
#         The MuJoCo model of the robot.
#     data : mujoco.MjData
#         The MuJoCo data of the robot.

#     Returns
#     -------
#     None
#         This function modifies the data.ctrl in place.
#     """

#     # Get the number of joints
#     num_joints = model.nu 
    
#     # Hinges take values between -pi/2 and pi/2
#     hinge_range = np.pi/2
#     rand_moves = np.random.uniform(low= -hinge_range, # -pi/2
#                                    high=hinge_range, # pi/2
#                                    size=num_joints) 

#     # There are 2 ways to make movements:
#     # 1. Set the control values directly (this might result in junky physics)
#     # data.ctrl = rand_moves

#     # 2. Add to the control values with a delta (this results in smoother physics)
#     delta = 0.05
#     data.ctrl += rand_moves * delta 

#     # Bound the control values to be within the hinge limits.
#     # If a value goes outside the bounds it might result in jittery movement.
#     data.ctrl = np.clip(data.ctrl, -np.pi/2, np.pi/2)

#     # Save movement to history
#     HISTORY.append(to_track[0].xpos.copy())

    ##############################################
    #
    # Take all the above into consideration when creating your controller
    # The input size, output size, output range
    # Your network might return ranges [-1,1], so you will need to scale it
    # to the expected [-pi/2, pi/2] range.
    # 
    # Or you might not need a delta and use the direct controller outputs
    #
    ##############################################

def show_qpos_history(history:list):
    # Convert list of [x,y,z] positions to numpy array
    pos_data = np.array(history)
    
    # Create figure and axis
    plt.figure(figsize=(10, 6))
    
    # Plot x,y trajectory
    plt.plot(pos_data[:, 0], pos_data[:, 1], 'b-', label='Path')
    plt.plot(pos_data[0, 0], pos_data[0, 1], 'go', label='Start')
    plt.plot(pos_data[-1, 0], pos_data[-1, 1], 'ro', label='End')
    
    # Add labels and title
    plt.xlabel('X Position')
    plt.ylabel('Y Position') 
    plt.title('Robot Path in XY Plane')
    plt.legend()
    plt.grid(True)
    
    # Set equal aspect ratio and center at (0,0)
    plt.axis('equal')
    max_range = max(abs(pos_data).max(), 0.3)  # At least 1.0 to avoid empty plots
    plt.xlim(-max_range, max_range)
    plt.ylim(-max_range, max_range)
    
    plt.show()

def main():
    """Main function to run the simulation with random movements."""
    # Initialise controller to controller to None, always in the beginning.
    mujoco.set_mjcb_control(None) # DO NOT REMOVE
    
    # Initialise world
    # Import environments from ariel.simulation.environments
    world = SimpleFlatWorld()

    # Initialise robot body
    # YOU MUST USE THE GECKO BODY
    gecko_core = gecko()     # DO NOT CHANGE

    # Spawn robot in the world
    # Check docstring for spawn conditions
    world.spawn(gecko_core.spec, spawn_position=[0, 0, 0])
    
    # Generate the model and data
    # These are standard parts of the simulation USE THEM AS IS, DO NOT CHANGE
    model = world.spec.compile()
    data = mujoco.MjData(model) # type: ignore

    # Initialise data tracking
    # to_track is automatically updated every time step
    # You do not need to touch it.
    geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]

    input_size = len(data.qpos)
    output_size = 3
    net = Net(input_size, output_size)

   
    
    # 4) search for good weights
    best_solution = search_for_weights(model, net, generations=10)

    # Set the control callback function
    # This is called every time step to get the next action. 
    mujoco.set_mjcb_control(lambda m, d: controller(net, data, to_track, best_solution))

    # This opens a viewer window and runs the simulation with the controller you defined
    # If mujoco.set_mjcb_control(None), then you can control the limbs yourself.
    viewer.launch(
        model=model,  # type: ignore
        data=data,
    )

    show_qpos_history(HISTORY)
    # If you want to record a video of your simulation, you can use the video renderer.

    # # Non-default VideoRecorder options
    # PATH_TO_VIDEO_FOLDER = "./__videos__"
    # video_recorder = VideoRecorder(output_folder=PATH_TO_VIDEO_FOLDER)

    # # Render with video recorder
    # video_renderer(
    #     model,
    #     data,
    #     duration=30,
    #     video_recorder=video_recorder,
    # )

if __name__ == "__main__":
    main()








