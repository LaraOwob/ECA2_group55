# Third-party libraries
import os
import csv
import glob
import numpy as np
import mujoco
from mujoco import viewer
import matplotlib.pyplot as plt
import cma
from deap import base, creator, tools
import random as pyrand
import pandas as pd
from pathlib import Path
from scipy.stats import ttest_ind


# Local libraries
from ariel.utils.renderers import video_renderer
from ariel.utils.video_recorder import VideoRecorder
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
from ariel.simulation.tasks.targeted_locomotion import distance_to_target
# import prebuilt robot phenotypes
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko

# Keep track of data / history
HISTORY = []
CTRL_RANGE = np.pi / 2
TARGET_POS = np.array([0.7, 0.3, 0.1])  # x,y,z
TOLERANCE = 0.3  # distance to target to be considered "reached"


def load_final_best(pattern):
    """Load final best_fitness values from CSVs matching the pattern."""
    vals = []
    for p in sorted(Path().glob(pattern)):
        df = pd.read_csv(p)
        if "best_fitness" in df.columns:
            vals.append(float(df["best_fitness"].iloc[-1]))
    return vals

def run_ttest():
    """Run Welch's t-test on final results of DE vs GA."""
    de = load_final_best("./results/mlp_de/mlp_de_seed*.csv")
    ga = load_final_best("./results/mlp_ga/mlp_ga_seed*.csv")

    print("Final DE runs:", de)
    print("Final GA runs:", ga)

    stat, p = ttest_ind(de, ga, equal_var=False)
    print(f"Welch's t-test: stat={stat:.3f}, p={p:.4f}")


#  BASELINE (Random) 
def run_random_baseline(out_csv, generations=200, pop_size=32, T=10.0, seed=0, model=None, data=None, to_track=None):
    """Evaluate pop_size random rollouts per generation; log best/mean/median to CSV."""
    rng = np.random.default_rng(seed)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["generation", "best_fitness", "mean_fitness", "median_fitness"])
        for g in range(generations):
            fits = [random_move(model, data, to_track) for _ in range(pop_size)]      
            best, mean, median = float(np.max(fits)), float(np.mean(fits)), float(np.median(fits))
            writer.writerow([g, best, mean, median])
            if (g + 1) % max(1, generations // 10) == 0:
                print(f"[baseline seed {seed}] Gen {g+1}/{generations}: best={best:.3f} mean={mean:.3f}")


def random_move(model, data, to_track) -> None:
    """Generate random movements for the robot's joints.
    
    The mujoco.set_mjcb_control() function will always give 
    model and data as inputs to the function. Even if you don't use them,
    you need to have them as inputs.

    Parameters
    ----------

    model : mujoco.MjModel
        The MuJoCo model of the robot.
    data : mujoco.MjData
        The MuJoCo data of the robot.

    Returns
    -------
    None
        This function modifies the data.ctrl in place.
    """

    # Get the number of joints
    num_joints = model.nu 
    
    # Hinges take values between -pi/2 and pi/2
    hinge_range = np.pi/2
    rand_moves = np.random.uniform(low= -hinge_range, # -pi/2
                                   high=hinge_range, # pi/2
                                   size=num_joints) 

    # There are 2 ways to make movements:
    # 1. Set the control values directly (this might result in junky physics)
    # data.ctrl = rand_moves

    # 2. Add to the control values with a delta (this results in smoother physics)
    delta = 0.05
    data.ctrl += rand_moves * delta 

    # Bound the control values to be within the hinge limits.
    # If a value goes outside the bounds it might result in jittery movement.
    data.ctrl = np.clip(data.ctrl, -np.pi/2, np.pi/2)

    # Save movement to history
    HISTORY.append(to_track[0].xpos.copy())
    
    final_pos =  to_track[0].xpos.copy()
    fitness = -np.linalg.norm( final_pos - TARGET_POS)
    return - float(fitness)
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
    plt.scatter(TARGET_POS[0], TARGET_POS[1], c='magenta', marker='*', s=200, label='Target')  # Add this line

    # Add labels and title
    plt.xlabel('X Position')
    plt.ylabel('Y Position') 
    plt.title('Robot Path in XY Plane')
    plt.legend()
    plt.grid(True)
    
    # Set equal aspect ratio and center at (0,0)
    plt.axis('equal')
    max_range = max(abs(pos_data).max(), 2)  # At least 1.0 to avoid empty plots
    plt.xlim(-max_range, max_range)
    plt.ylim(-max_range, max_range)
    
    plt.show()


# MLP Controller (shared) for Experiments 1a, 1b, and 3
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
    def forward(self, prev_ctrl: np.ndarray, t: float, target_pos: np.ndarray, current_pos: np.ndarray) -> np.ndarray:
        w = 2*np.pi*self.freq_hz
        #includes the distance to target as input
        delta = target_pos[:2] - current_pos[:2] 
        obs = np.concatenate([prev_ctrl, [np.sin(w*t), np.cos(w*t)], delta], dtype=np.float64)
        W1, b1, W2, b2 = self._unpack()
        h = np.tanh(obs @ W1 + b1)
        y = np.tanh(h @ W2 + b2)
        return CTRL_RANGE * y

def evaluate_mlp_headless(theta, model, data, core_bind, T=10.0, hidden=32, freq_hz=1.0, alpha=0.2):
    """
    Headless rollout for an MLP controller with params 'theta'.
    Returns loss = -fitness (so optimizers can minimize).
    alpha: smoothing factor for actions (0=no smoothing, 1=full new action).
    """
    #creates mujoco data for evaluation
    mujoco.set_mjcb_control(None)
    data_eval = mujoco.MjData(model)
    mujoco.mj_resetData(model, data_eval)
    
    dt = float(model.opt.timestep)
    steps = int(T / dt)
    nu = model.nu

    ctrlr = MLPController(nu, hidden=hidden, freq_hz=freq_hz)
    ctrlr.set_params(theta)
    prev = np.zeros(nu, dtype=np.float64)
    start_pos = data_eval.xpos[core_bind.id].copy()
    initial_dist = np.linalg.norm(start_pos - TARGET_POS)
    #starting values for fitness calculation
    energy_acc = 0.0
    dctrl_acc = 0.0
    progress_reward = 0.0
    reached =False
    time_to_target = None
    
    for k in range(steps):
        t = k * dt
        current_pos = data_eval.xpos[core_bind.id]
        raw = ctrlr.forward(prev, t,TARGET_POS,current_pos)
        ctrl = (1 - alpha) * prev + alpha * raw
        ctrl = np.clip(ctrl, -CTRL_RANGE, CTRL_RANGE)
        
        dctrl_acc += float(np.mean(np.abs(ctrl - prev)))
        energy_acc += float(np.mean(np.abs(ctrl)))
        
        prev = ctrl
        data_eval.ctrl[:] = ctrl
        mujoco.mj_step(model, data_eval)
        
        dist = np.linalg.norm(current_pos - TARGET_POS)
        if dist <TOLERANCE:
            reached = True
            time_to_target = k*dt
        
        if not np.isfinite(core_bind.xpos[0]):
            return 1e6  # big loss on crash
    if reached:
        # penalty for movement after reaching
        extra_movement = energy_acc / steps  
        # big bonus for reaching target quickly
        fitness = 1000.0 + (T - time_to_target) * 100.0 - 50.0 * extra_movement  

    else:
        #if target not reached, calculate fitness based on distance to target
        current_pos = data_eval.xpos[core_bind.id]
        final_dist = np.linalg.norm(current_pos - TARGET_POS)
        progress_reward = (initial_dist - final_dist)/steps
        dx = float(current_pos[0] -start_pos[0])
        lam_energy = 0.01
        lam_smooth = 0.01
        energy = energy_acc / steps
        smooth = dctrl_acc / steps
        fitness = dx - lam_energy * energy - lam_smooth * smooth - progress_reward*lam_smooth
    return -float(fitness)  




def mlp_viewer_callback_factory(theta, to_track, hidden=32, freq_hz=1.0, alpha=0.2, dt=0.01):
    """Viewer callback to run the trained MLP policy."""
    #Initialize step, prev_controller and current controller
    step = 0
    prev_controller = None
    controller = None

    def callback(model, data):
        nonlocal step, prev_controller, controller

        if controller is None:
            # Initialize controller on first call
            nu = model.nu
            controller = MLPController(nu, hidden=hidden, freq_hz=freq_hz)
            #Set parameters of solution (weigths)
            controller.set_params(theta)
            prev_controller = np.zeros(nu, dtype=np.float64)

        t = step * dt
        current_pos = to_track[0].xpos.copy() 
        #Calcualte mlp output
        raw_ctrl = controller.forward(prev_controller, t,TARGET_POS,current_pos)
        #Smoothing movement
        ctrl = (1 - alpha) * prev_controller + alpha * raw_ctrl
        #Set range of movement between interval
        ctrl = np.clip(ctrl, -CTRL_RANGE, CTRL_RANGE)

        #Update data
        data.ctrl[:] = ctrl
        final_dist = np.linalg.norm(current_pos - TARGET_POS)
        print(f"Step {t}, pos {current_pos}, dist to target {final_dist:.3f}")
        #Add new step to history
        HISTORY.append(current_pos)
        #Set previous_controller to current controller
        prev_controller[:] = ctrl
        #Increase step
        step += 1

    return callback

#  Differential Evolution on MLP (1b)
def train_mlp_de(out_csv, generations=100, pop_size=30, T=10.0, seed=0, model=None, data=None, core_bind=None,
                 hidden=32, freq_hz=1.0, F=0.7, CR=0.9, init_scale=0.5, alpha=0.2):
    """
    Simple, Differential Evolution for MLP parameters.
    """
    #Selects random number 
    rng = np.random.default_rng(seed)
    nu = model.nu
    number_params = MLPController(nu, hidden=hidden, freq_hz=freq_hz).n_params
    # number_params = ctrlr.n_params

    # Initialize population by sampling normal distribution with mean 0 and stdv of init_scale
    pop = rng.normal(loc=0.0, scale=init_scale, size=(pop_size, number_params))
    #Initialize empty numby array with length number of candidates
    fits = np.empty(pop_size, dtype=np.float64)

    # Calculate fitness for each candidate, pop[i] = candidate
    for i in range(pop_size):
        loss = evaluate_mlp_headless(pop[i], model, data, core_bind, T=T, hidden=hidden, freq_hz=freq_hz, alpha=alpha)
        fits[i] = -loss  # fitness

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
                mutant = pop[a] + F * (pop[b] - pop[c])

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

#  Experiment 3: MLP + GA
def train_mlp_ga(out_csv, generations=100, pop_size=60, T=10.0, seed=0, model=None, data=None, core_bind=None,
                 hidden=32, freq_hz=1.0, crossover_prob=0.6, mutation_prob=0.3, mut_sigma=0.3, alpha=0.2):
    """
    Genetic Algorithm on MLP parameters using DEAP (recognized EA library).
    Maximizes fitness. Returns (best_theta, best_fitness).
    """
   
    np.random.seed(seed)
    pyrand.seed(seed)

    nu = model.nu

    n_params = MLPController(nu, hidden=hidden, freq_hz=freq_hz).n_params

    # Safe creator (avoid duplicate class creation on re-runs)
    try:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    except Exception:
        pass
    try:
        creator.create("Individual", list, fitness=creator.FitnessMax)
    except Exception:
        pass

    toolbox = base.Toolbox()
    #Registers function attr_float, which will return a sample from Normal distribution with mean 0 and stdv 0.5
    toolbox.register("attr_float", pyrand.gauss, 0.0, 0.5)
    #Registers function individual(), which will create a list of length n_params, and values the parameters of the MLP
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n_params)
    #Registers function population(),  which will return list of the population, each candidate with its values and fitness
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    #Evaluate a candidate
    def eval_ind(individual):
        theta = np.asarray(individual, dtype=np.float64)
        loss = evaluate_mlp_headless(theta, model, data, core_bind, T=T, hidden=hidden, freq_hz=freq_hz, alpha=alpha)
        #Casted to tuple, because DEAP needs it to be a tuple
        return (-loss,) 
    #Register the evaluation function
    toolbox.register("evaluate", eval_ind)
    #Registers the cross over function: BLX-alpha crossover
    toolbox.register("mate", tools.cxBlend, alpha=0.5) 
    #Registers the mutation function: adds gaussian noise to genes, mean 0 and stdv mut_sigma, and mutation probability 1/number params
    toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=mut_sigma, indpb=1.0/n_params)
    #Registers selection procedure: Tornament: selects best out of 3 candidates
    toolbox.register("select", tools.selTournament, tournsize=3)

    #Registers and initializes the population to be of size pop_size
    population = toolbox.population(n=pop_size)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["generation", "best_fitness", "mean_fitness", "median_fitness"])

        # Evaluate each inidivual candidate
        for individual in population:
            individual.fitness.values = toolbox.evaluate(individual)

        for g in range(generations):
            # Selection poule for cross_over and mutation
            offspring = toolbox.select(population, len(population))
            # Parent is cloned to be offspring, to make sure parents still exists
            offspring = list(map(toolbox.clone, offspring))

            # Crossover
            #Step size of 2 because we select 2 parents
            for i in range(0, len(offspring)-1, 2):
                #Draw random number to check with cross over probability
                if pyrand.random() < crossover_prob:
                    #Apply cross_over
                    toolbox.mate(offspring[i], offspring[i+1])
                    #Delete old fitness values of offsprings
                    del offspring[i].fitness.values
                    del offspring[i+1].fitness.values

            # Mutation
            for individual in offspring:
                #Draw random number to check with mutation probability
                if pyrand.random() < mutation_prob:
                    #Apply mutation
                    toolbox.mutate(individual)
                    #Delete old fitness of offspring
                    del individual.fitness.values

            #The fitness of the new offsprings still need to be updated
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            for individual in invalid:
                individual.fitness.values = toolbox.evaluate(individual)

            # Survivor replacement
            population = offspring

            # Logging
            fits = np.array([ind.fitness.values[0] for ind in population], dtype=np.float64)
            writer.writerow([g, float(np.max(fits)), float(np.mean(fits)), float(np.median(fits))])

            if (g + 1) % max(1, generations // 10) == 0:
                print(f"[mlp_ga seed {seed}] Gen {g+1}/{generations}: best={np.max(fits):.3f} mean={np.mean(fits):.3f}")

    # Champion
    
    best_ind = tools.selBest(population, 1)[0]
    best_theta = np.asarray(best_ind, dtype=np.float64)
    best_fit = float(best_ind.fitness.values[0])
    return best_theta, best_fit


# PLOTTING 
def moving_avg(x, w=10):
    if w <= 1:
        return x
    # same-length centered moving average
    pad = w // 2
    xpad = np.pad(x, (pad, w - 1 - pad), mode='edge')
    kern = np.ones(w) / w
    return np.convolve(xpad, kern, mode="valid")

def load_series(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path)
    return df["best_fitness"].to_numpy()

def plot_experiment(exp_name, exp_csvs, rand_csvs, rand_label, final_exp ,out_png, window=10, title=None, xlabel="Generation", ylabel="Fitness"):
    # Load runs
    exp_runs = [load_series(p) for p in exp_csvs]
    if len(exp_runs) == 0:
        print(f"[plot] No runs found for {exp_name}")
        return
    L = min(len(r) for r in exp_runs)
    exp_runs = [r[:L] for r in exp_runs]
    exp_mean = np.mean(exp_runs, axis=0)
    exp_std  = np.std(exp_runs, axis=0)

    

    rand_runs = [load_series(p) for p in rand_csvs]
    if len(rand_runs) > 0:
        Lr = min(len(r) for r in rand_runs)
        L = min(L, Lr)
        exp_mean, exp_std = exp_mean[:L], exp_std[:L]
        rand_runs = [r[:L] for r in rand_runs]
        rand_mean = np.mean(rand_runs, axis=0)
        
    else:
        rand_mean = None

    x = np.arange(len(exp_mean))
    plt.figure(figsize=(8,4.5))
    # individual runs (light)
    k = 1 
    if not  final_exp:
        for r in exp_runs:
            plt.plot(np.arange(len(r[:L])), r[:L], linewidth=6, alpha=0.35, label = f"Run {exp_name} {k}")
            k+=1 
    # mean + std
    plt.plot(x, exp_mean, linewidth=2.5, label=f"{exp_name} Mean")
    if not final_exp:
        plt.fill_between(x, exp_mean-exp_std, exp_mean+exp_std, alpha=0.25, label=f"{exp_name} ± std")

    # random baseline overlay
    if rand_mean is not None:
        plt.plot(x, rand_mean, linestyle="--", linewidth=2, label=rand_label +" Mean")
    else:
        print(f"[plot] No random runs found")
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    if title is None:
        title = f"{exp_name} vs Random"
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.show()
    print(f"[plot] Saved {out_png}")




def main(experiment,
         generations = 60, pop_size=24, T=10.0, seed=[0,1,2],
         mlp_hidden=32, mlp_freq_hz=1.0, mlp_alpha=0.25,
         de_F=0.7, de_CR=0.9, de_init_scale=0.5,
         ga_cxpb=0.6, ga_mutpb=0.3, ga_mut_sigma=0.3,    
         plot_window=10):
    """Main function to run the simulation with random movements.
        Experiments:
      - "baseline": random moves CSVs + viewer of baseline
      - "mlp_de":   MLP + Differential Evolution CSVs + viewer of champion
        - "mlp_ga":   MLP + Genetic Algorithm (DEAP) 
      - "plot":     Generate 3 plots (1a, 1b, 2) overlaying Random baseline"""
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
    # Add target marker
    world.spec.worldbody.add_geom(
    type=mujoco.mjtGeom.mjGEOM_SPHERE,
    name="target_marker",
    size=[0.05, 0.0, 0.0],          
    pos=TARGET_POS, 
    rgba=[1, 0, 0, 1],  
    contype=0,            
    conaffinity=0)
    
    # Generate the model and data
    # These are standard parts of the simulation USE THEM AS IS, DO NOT CHANGE
    model = world.spec.compile()
    data = mujoco.MjData(model) # type: ignore

    # Initialise data tracking
    # to_track is automatically updated every time step
    # You do not need to touch it.
    geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]

    core_bind = to_track[0]
    dt = float(model.opt.timestep)

    # Run random baseline experiment
    if experiment == "baseline":
        for s in seed:
            out_csv = f"./results/random/random_seed{s}.csv"
            run_random_baseline(out_csv, generations=generations, pop_size=pop_size, T=T, seed=s,
                                model=model, data=data, to_track=to_track)
            
         # 2) (Optional) visualize one baseline rollout in the viewer
        #    Use the SAME distribution as in the headless baseline.
        mujoco.mj_resetData(model, data)         # start fresh

        # Set the control callback function
        # This is called every time step to get the next action. 
        mujoco.set_mjcb_control(lambda m,d: random_move(m, d, to_track))
        viewer.launch(model=model, data=data)
        show_qpos_history(HISTORY)
        return  # don't also run the demo callback
    
    # MLP + DE
    if experiment == "mlp_de":
        best_theta_viz = None
        for i, s in enumerate(seed):
            out_csv = f"./results/mlp_de/mlp_de_seed{s}.csv"
            theta, best_fit = train_mlp_de(
                out_csv=out_csv, generations=generations, pop_size=pop_size, T=T, seed=s,
                model=model, data=data, core_bind=core_bind,
                hidden=mlp_hidden, freq_hz=mlp_freq_hz, F=de_F, CR=de_CR, init_scale=de_init_scale, alpha=mlp_alpha
            )
            print(f"[mlp_de seed {s}] best_fitness={best_fit:.3f}")
            if i == 0: best_theta_viz = theta
        if best_theta_viz is not None:
            mujoco.mj_resetData(model, data)
            cb = mlp_viewer_callback_factory(best_theta_viz, to_track, hidden=mlp_hidden, freq_hz=mlp_freq_hz, alpha=mlp_alpha, dt=dt)
            mujoco.set_mjcb_control(cb)
            viewer.launch(model=model, data=data)
            show_qpos_history(HISTORY)
        return

    

    #  MLP + GA (DEAP)
    if experiment == "mlp_ga":
        best_theta_viz = None
        for i, s in enumerate(seed):
            out_csv = f"./results/mlp_ga/mlp_ga_seed{s}.csv"
            theta, best_fit = train_mlp_ga(
                out_csv=out_csv, generations=generations, pop_size=max(pop_size, 60), T=T, seed=s,
                model=model, data=data, core_bind=core_bind,
                hidden=mlp_hidden, freq_hz=mlp_freq_hz,
                crossover_prob=ga_cxpb, mutation_prob=ga_mutpb, mut_sigma=ga_mut_sigma, alpha=mlp_alpha
            )
            print(f"[mlp_ga seed {s}] best_fitness={best_fit:.3f}")
            if i == 0: best_theta_viz = theta
        if best_theta_viz is not None:
            mujoco.mj_resetData(model, data)
            cb = mlp_viewer_callback_factory(best_theta_viz, to_track, hidden=mlp_hidden, freq_hz=mlp_freq_hz, alpha=mlp_alpha, dt=dt)
            mujoco.set_mjcb_control(cb)
            viewer.launch(model=model, data=data)
           
        show_qpos_history(HISTORY)
        return


    # PLOTS (all three)
    if experiment == "plot":
        # Random baseline CSVs (3 runs)
        rand_csvs = sorted(glob.glob("./results/random/random_seed*.csv"))[:3]

        # 1 MLP+CMA vs Random
        exp_csvs = sorted(glob.glob("./results/mlp_ga/mlp_ga_seed*.csv"))[:3]
        plot_experiment(
            exp_name="MLP + GA",
            exp_csvs=exp_csvs,
            rand_csvs=rand_csvs,
            rand_label="Random",
            final_exp = False,
            out_png="./results/plots/exp1_mlp_ga_vs_random.png",
            window=plot_window,
            title="Experiment 1: MLP + GA vs Random"
        )

        # 2 MLP+DE vs Random
        exp_csvs = sorted(glob.glob("./results/mlp_de/mlp_de_seed*.csv"))[:3]
        plot_experiment(
            exp_name="MLP + DE",
            exp_csvs=exp_csvs,
            rand_csvs=rand_csvs,
            rand_label="Random",
            final_exp = False,
            out_png="./results/plots/exp2_mlp_de_vs_random.png",
            window=plot_window,
            title="Experiment 2: MLP + DE vs Random"
        )

        # 2 MLP + GA vs MLP + DE
        exp_csvs1 = sorted(glob.glob("./results/mlp_ga/mlp_ga_seed*.csv"))[:3]
        exp_csvs2= sorted(glob.glob("./results/mlp_de/mlp_de_seed*.csv"))[:3]
        plot_experiment(
            exp_name="MLP + GA",
            exp_csvs=exp_csvs1,
            rand_csvs=exp_csvs2,
            rand_label="MLP + DE",
            final_exp = True,
            out_png="./results/plots/exp3_mlp_ga_vs_mlp_de.png",
            window=plot_window,
            title="Experiment 3: MLP + GA vs MLP + DE"
        )

        run_ttest()

        return


    # Set the control callback function
    # This is called every time step to get the next action. 
    mujoco.set_mjcb_control(lambda m,d: random_move(m, d, to_track))

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
    experiment = "mlp_ga"  # "baseline" or "mlp_cma" or "mlp_de" or "cpg_cma" or "mlp_ga" or "plot"
    main(experiment=experiment)


