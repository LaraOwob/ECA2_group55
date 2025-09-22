# Third-party libraries
import os
import csv
import glob
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

# Keep track of data / history
HISTORY = []
CTRL_RANGE = np.pi / 2


#  BASELINE (Random) 
def rollout_random_headless(model, data, core_bind, T=10.0, rng=None):
    """One headless episode with step-wise random actions; returns a scalar fitness."""
    if rng is None:
        rng = np.random.default_rng()
    mujoco.set_mjcb_control(None)  # ensure callback isn't used
    mujoco.mj_resetData(model, data)

    dt = float(model.opt.timestep)
    steps = int(T / dt)
    nu = model.nu

    x0 = float(core_bind.xpos[0])
    energy_acc = 0.0
    dctrl_acc = 0.0
    prev = np.zeros(nu, dtype=np.float64)

    for _ in range(steps):
        ctrl = rng.uniform(-CTRL_RANGE, CTRL_RANGE, size=nu)
        ctrl = np.clip(ctrl, -CTRL_RANGE, CTRL_RANGE)
        dctrl_acc += float(np.mean(np.abs(ctrl - prev)))
        energy_acc += float(np.mean(np.abs(ctrl)))
        prev = ctrl

        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)
        if not np.isfinite(core_bind.xpos[0]):
            return -1e6  # crash guard

    dx = float(core_bind.xpos[0] - x0)
    lam_energy = 0.01
    lam_smooth = 0.01
    energy = energy_acc / steps
    smooth = dctrl_acc / steps
    fitness = dx - lam_energy * energy - lam_smooth * smooth
    return float(fitness)



def run_random_baseline(out_csv, generations=200, pop_size=32, T=10.0, seed=0, model=None, data=None, core_bind=None):
    """Evaluate pop_size random rollouts per generation; log best/mean/median to CSV."""
    rng = np.random.default_rng(seed)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["generation", "best_fitness", "mean_fitness", "median_fitness"])
        for g in range(generations):
            fits = [
                rollout_random_headless(model, data, core_bind, T=T, rng=rng)
                for _ in range(pop_size)
            ]
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


def random_baseline_callback_factory(seed, to_track):
    """
    Returns a MuJoCo control callback that matches the baseline:
    at each step, set ctrl ~ Uniform[-pi/2, pi/2] independently.
    """
    rng = np.random.default_rng(seed)
    def cb(model, data):
        nu = model.nu
        ctrl = rng.uniform(-CTRL_RANGE, CTRL_RANGE, size=nu)
        data.ctrl[:] = np.clip(ctrl, -CTRL_RANGE, CTRL_RANGE)
        HISTORY.append(to_track[0].xpos.copy())
    return cb



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
        self.in_dim = nu + 2
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
        assert theta.shape[0] == self.n_params
        self.theta = theta.copy()

    def _unpack(self):
        t = self.theta
        i = 0
        W1 = t[i:i+self.in_dim*self.hidden].reshape(self.in_dim, self.hidden); i += self.in_dim*self.hidden
        b1 = t[i:i+self.hidden]; i += self.hidden
        W2 = t[i:i+self.hidden*self.out_dim].reshape(self.hidden, self.out_dim); i += self.hidden*self.out_dim
        b2 = t[i:i+self.out_dim]
        return W1, b1, W2, b2

    def forward(self, prev_ctrl: np.ndarray, t: float) -> np.ndarray:
        w = 2*np.pi*self.freq_hz
        obs = np.concatenate([prev_ctrl, [np.sin(w*t), np.cos(w*t)]], dtype=np.float64)
        W1, b1, W2, b2 = self._unpack()
        h = np.tanh(obs @ W1 + b1)
        y = np.tanh(h @ W2 + b2)  # in [-1,1]
        return CTRL_RANGE * y

def evaluate_mlp_headless(theta, model, data, core_bind, T=10.0, hidden=32, freq_hz=1.0, alpha=0.2):
    """
    Headless rollout for an MLP controller with params 'theta'.
    Returns loss = -fitness (so optimizers can minimize).
    alpha: smoothing factor for actions (0=no smoothing, 1=full new action).
    """
    mujoco.set_mjcb_control(None)
    mujoco.mj_resetData(model, data)

    dt = float(model.opt.timestep)
    steps = int(T / dt)
    nu = model.nu

    ctrlr = MLPController(nu, hidden=hidden, freq_hz=freq_hz)
    ctrlr.set_params(theta)
    prev = np.zeros(nu, dtype=np.float64)

    x0 = float(core_bind.xpos[0])
    energy_acc = 0.0
    dctrl_acc = 0.0

    for k in range(steps):
        t = k * dt
        raw = ctrlr.forward(prev, t)
        ctrl = (1 - alpha) * prev + alpha * raw
        ctrl = np.clip(ctrl, -CTRL_RANGE, CTRL_RANGE)
        dctrl_acc += float(np.mean(np.abs(ctrl - prev)))
        energy_acc += float(np.mean(np.abs(ctrl)))
        prev = ctrl
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)
        if not np.isfinite(core_bind.xpos[0]):
            return 1e6  # big loss on crash

    dx = float(core_bind.xpos[0] - x0)
    lam_energy = 0.01
    lam_smooth = 0.01
    energy = energy_acc / steps
    smooth = dctrl_acc / steps
    fitness = dx - lam_energy * energy - lam_smooth * smooth
    return -float(fitness)  # minimize loss

def mlp_viewer_callback_factory(theta, to_track, hidden=32, freq_hz=1.0, alpha=0.2, dt=0.01):
    """Viewer callback to run the trained MLP policy."""
    step_k = {'k': 0}
    ctrlr_cache = {'ctrlr': None}
    prev = np.zeros(1, dtype=np.float64)  # resized on first call
    def cb(model, data):
        nonlocal prev
        if ctrlr_cache['ctrlr'] is None:
            nu = model.nu
            ctrlr_cache['ctrlr'] = MLPController(nu, hidden=hidden, freq_hz=freq_hz)
            ctrlr_cache['ctrlr'].set_params(theta)
            prev = np.zeros(nu, dtype=np.float64)
        k = step_k['k']; t = k * dt
        raw = ctrlr_cache['ctrlr'].forward(prev, t)
        ctrl = (1 - alpha) * prev + alpha * raw
        ctrl = np.clip(ctrl, -CTRL_RANGE, CTRL_RANGE)
        data.ctrl[:] = ctrl
        HISTORY.append(to_track[0].xpos.copy())
        prev[:] = ctrl
        step_k['k'] = k + 1
    return cb


# CMA-ES on MLP (1a) 

def train_mlp_cma(out_csv, generations=100, pop_size=None, T=10.0, seed=0, model=None, data=None, core_bind=None,
                  hidden=32, freq_hz=1.0, sigma0=0.5, alpha=0.2):
    """CMA-ES optimization of MLP weights. Logs per-generation best/mean/median fitness. Returns (best_theta, best_fitness)."""
    try:
        import cma
    except ImportError as e:
        raise RuntimeError("CMA-ES requires the 'cma' package. `pip install cma`.") from e

    nu = model.nu
    ctrlr = MLPController(nu, hidden=hidden, freq_hz=freq_hz)
    n_params = ctrlr.n_params
    if pop_size is None:
        pop_size = int(4 + np.floor(3 * np.log(n_params)))

    x0 = np.zeros(n_params, dtype=np.float64)
    es = cma.CMAEvolutionStrategy(x0, sigma0, {'popsize': pop_size, 'seed': seed})

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    best_theta = None
    best_fit = -np.inf

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["generation", "best_fitness", "mean_fitness", "median_fitness"])
        g = 0
        while not es.stop() and g < generations:
            thetas = es.ask()
            losses = []
            fits = []
            for th in thetas:
                loss = evaluate_mlp_headless(
                    np.asarray(th, dtype=np.float64),
                    model, data, core_bind,
                    T=T, hidden=hidden, freq_hz=freq_hz, alpha=alpha
                )
                losses.append(loss)
                fits.append(-loss)
            es.tell(thetas, losses)
            gen_best = float(np.max(fits))
            gen_mean = float(np.mean(fits))
            gen_median = float(np.median(fits))
            writer.writerow([g, gen_best, gen_mean, gen_median])
            if gen_best > best_fit:
                best_fit = gen_best
                best_theta = np.asarray(thetas[int(np.argmax(fits))], dtype=np.float64)
            if (g + 1) % max(1, generations // 10) == 0:
                print(f"[mlp_cma seed {seed}] Gen {g+1}/{generations}: best={gen_best:.3f} mean={gen_mean:.3f}")
            g += 1

    return best_theta, best_fit



#  Differential Evolution on MLP (1b)

def train_mlp_de(out_csv, generations=100, pop_size=30, T=10.0, seed=0, model=None, data=None, core_bind=None,
                 hidden=32, freq_hz=1.0, F=0.7, CR=0.9, init_scale=0.5, alpha=0.2):
    """
    Simple, dependency-free Differential Evolution (DE/rand/1/bin) for MLP params.
    Maximizes fitness (like our logs). Returns (best_theta, best_fitness).
    """
    rng = np.random.default_rng(seed)
    nu = model.nu
    ctrlr = MLPController(nu, hidden=hidden, freq_hz=freq_hz)
    n = ctrlr.n_params

    # Initialize population around 0 with scale
    pop = rng.normal(loc=0.0, scale=init_scale, size=(pop_size, n))
    fits = np.empty(pop_size, dtype=np.float64)

    # Evaluate initial population
    for i in range(pop_size):
        loss = evaluate_mlp_headless(pop[i], model, data, core_bind, T=T, hidden=hidden, freq_hz=freq_hz, alpha=alpha)
        fits[i] = -loss  # fitness

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    best_idx = int(np.argmax(fits))
    best_theta = pop[best_idx].copy()
    best_fit = float(fits[best_idx])

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["generation", "best_fitness", "mean_fitness", "median_fitness"])
        for g in range(generations):
            for i in range(pop_size):
                # Mutation: select a,b,c distinct from i
                idxs = list(range(pop_size))
                idxs.remove(i)
                a, b, c = rng.choice(idxs, size=3, replace=False)
                mutant = pop[a] + F * (pop[b] - pop[c])

                # Crossover (binomial)
                cross = rng.random(n) < CR
                jrand = rng.integers(0, n)
                cross[jrand] = True
                trial = np.where(cross, mutant, pop[i])

                # Evaluate trial
                loss = evaluate_mlp_headless(trial, model, data, core_bind, T=T, hidden=hidden, freq_hz=freq_hz, alpha=alpha)
                f_trial = -loss

                # Selection
                if f_trial >= fits[i]:
                    pop[i] = trial
                    fits[i] = f_trial

            # Log stats
            gen_best = float(np.max(fits))
            gen_mean = float(np.mean(fits))
            gen_median = float(np.median(fits))
            writer.writerow([g, gen_best, gen_mean, gen_median])

            # Track global best
            if gen_best > best_fit:
                best_fit = gen_best
                best_theta = pop[int(np.argmax(fits))].copy()

            if (g + 1) % max(1, generations // 10) == 0:
                print(f"[mlp_de seed {seed}] Gen {g+1}/{generations}: best={gen_best:.3f} mean={gen_mean:.3f}")

    return best_theta, best_fit



# CPG Controller (Exp 2)

class CPGController:
    """
    Open-loop per-joint oscillator:
      ctrl_j(t) = b_j + A_j * sin(2π f_j t + phi_j)
    Genome per joint: [A, f, phi, b]
    """
    def __init__(self, nu: int, A_max=CTRL_RANGE, f_min=0.1, f_max=2.0, b_max=CTRL_RANGE):
        self.nu = nu
        self.A_max = A_max
        self.f_min = f_min
        self.f_max = f_max
        self.b_max = b_max
        self.params = None  # shape (nu, 4)

    @property
    def n_params(self):
        return self.nu * 4

    def set_params(self, theta: np.ndarray):
        assert theta.shape[0] == self.n_params
        self.params = theta.reshape(self.nu, 4).copy()

    def forward(self, t: float) -> np.ndarray:
        A = np.clip(self.params[:, 0], 0.0, self.A_max)
        f = np.clip(self.params[:, 1], self.f_min, self.f_max)
        phi = self.params[:, 2]
        b = np.clip(self.params[:, 3], -self.b_max, self.b_max)
        ctrl = b + A * np.sin(2*np.pi*f*t + phi)
        return np.clip(ctrl, -CTRL_RANGE, CTRL_RANGE)

def evaluate_cpg_headless(theta, model, data, core_bind, T=10.0, A_max=CTRL_RANGE, f_min=0.1, f_max=2.0, b_max=CTRL_RANGE, alpha=0.2):
    """Headless rollout for a CPG controller. Returns loss = -fitness."""
    mujoco.set_mjcb_control(None)
    mujoco.mj_resetData(model, data)

    dt = float(model.opt.timestep)
    steps = int(T / dt)
    nu = model.nu

    ctrlr = CPGController(nu, A_max=A_max, f_min=f_min, f_max=f_max, b_max=b_max)
    ctrlr.set_params(theta)
    prev = np.zeros(nu, dtype=np.float64)

    x0 = float(core_bind.xpos[0])
    energy_acc = 0.0
    dctrl_acc = 0.0

    for k in range(steps):
        t = k * dt
        raw = ctrlr.forward(t)
        ctrl = (1 - alpha) * prev + alpha * raw
        ctrl = np.clip(ctrl, -CTRL_RANGE, CTRL_RANGE)
        dctrl_acc += float(np.mean(np.abs(ctrl - prev)))
        energy_acc += float(np.mean(np.abs(ctrl)))
        prev = ctrl
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)
        if not np.isfinite(core_bind.xpos[0]):
            return 1e6

    dx = float(core_bind.xpos[0] - x0)
    lam_energy = 0.01
    lam_smooth = 0.01
    energy = energy_acc / steps
    smooth = dctrl_acc / steps
    fitness = dx - lam_energy * energy - lam_smooth * smooth
    return -float(fitness)

def cpg_viewer_callback_factory(theta, to_track, dt=0.01, A_max=CTRL_RANGE, f_min=0.1, f_max=2.0, b_max=CTRL_RANGE, alpha=0.2):
    """Viewer callback for a trained CPG."""
    step_k = {'k': 0}
    ctrlr_cache = {'ctrlr': None}
    prev = np.zeros(1, dtype=np.float64)
    def cb(model, data):
        nonlocal prev
        if ctrlr_cache['ctrlr'] is None:
            nu = model.nu
            ctrlr_cache['ctrlr'] = CPGController(nu, A_max=A_max, f_min=f_min, f_max=f_max, b_max=b_max)
            ctrlr_cache['ctrlr'].set_params(theta)
            prev = np.zeros(nu, dtype=np.float64)
        k = step_k['k']; t = k * dt
        raw = ctrlr_cache['ctrlr'].forward(t)
        ctrl = (1 - alpha) * prev + alpha * raw
        ctrl = np.clip(ctrl, -CTRL_RANGE, CTRL_RANGE)
        data.ctrl[:] = ctrl
        HISTORY.append(to_track[0].xpos.copy())
        prev[:] = ctrl
        step_k['k'] = k + 1
    return cb

def train_cpg_cma(out_csv, generations=100, pop_size=None, T=10.0, seed=0, model=None, data=None, core_bind=None,
                  A_max=CTRL_RANGE, f_min=0.1, f_max=2.0, b_max=CTRL_RANGE, sigma0=0.5, alpha=0.2):
    """CMA-ES optimization of CPG params. Returns (best_theta, best_fitness)."""
    try:
        import cma
    except ImportError as e:
        raise RuntimeError("CMA-ES requires the 'cma' package. `pip install cma`.") from e

    nu = model.nu
    n_params = nu * 4
    if pop_size is None:
        pop_size = int(4 + np.floor(3 * np.log(n_params)))

    x0 = np.zeros(n_params, dtype=np.float64)
    es = cma.CMAEvolutionStrategy(x0, sigma0, {'popsize': pop_size, 'seed': seed})

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    best_theta = None
    best_fit = -np.inf

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["generation", "best_fitness", "mean_fitness", "median_fitness"])
        g = 0
        while not es.stop() and g < generations:
            thetas = es.ask()
            losses, fits = [], []
            for th in thetas:
                loss = evaluate_cpg_headless(
                    np.asarray(th, dtype=np.float64),
                    model, data, core_bind,
                    T=T, A_max=A_max, f_min=f_min, f_max=f_max, b_max=b_max, alpha=alpha
                )
                losses.append(loss)
                fits.append(-loss)
            es.tell(thetas, losses)
            gen_best = float(np.max(fits))
            gen_mean = float(np.mean(fits))
            gen_median = float(np.median(fits))
            writer.writerow([g, gen_best, gen_mean, gen_median])
            if gen_best > best_fit:
                best_fit = gen_best
                best_theta = np.asarray(thetas[int(np.argmax(fits))], dtype=np.float64)
            if (g + 1) % max(1, generations // 10) == 0:
                print(f"[cpg_cma seed {seed}] Gen {g+1}/{generations}: best={gen_best:.3f} mean={gen_mean:.3f}")
            g += 1

    return best_theta, best_fit


#  Experiment 3: MLP + GA

def train_mlp_ga(out_csv, generations=100, pop_size=60, T=10.0, seed=0, model=None, data=None, core_bind=None,
                 hidden=32, freq_hz=1.0, cxpb=0.6, mutpb=0.3, mut_sigma=0.3, alpha=0.2):
    """
    Genetic Algorithm on MLP parameters using DEAP (recognized EA library).
    Maximizes fitness. Returns (best_theta, best_fitness).
    """
    try:
        from deap import base, creator, tools
    except ImportError as e:
        raise RuntimeError("DEAP is required for GA. Install with `pip install deap`.") from e

    import random as pyrand
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
    # init around 0.0 (like CMA-ES mean)
    toolbox.register("attr_float", pyrand.gauss, 0.0, 0.5)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n_params)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def eval_ind(individual):
        theta = np.asarray(individual, dtype=np.float64)
        loss = evaluate_mlp_headless(theta, model, data, core_bind, T=T, hidden=hidden, freq_hz=freq_hz, alpha=alpha)
        return (-loss,)  # DEAP expects a tuple

    toolbox.register("evaluate", eval_ind)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)  # BLX-α crossover for reals
    toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=mut_sigma, indpb=1.0/n_params)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=pop_size)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["generation", "best_fitness", "mean_fitness", "median_fitness"])

        # Evaluate initial
        for ind in pop:
            ind.fitness.values = toolbox.evaluate(ind)

        for g in range(generations):
            # Selection & variation
            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))

            # Crossover
            for i in range(0, len(offspring)-1, 2):
                if pyrand.random() < cxpb:
                    toolbox.mate(offspring[i], offspring[i+1])
                    del offspring[i].fitness.values
                    del offspring[i+1].fitness.values

            # Mutation
            for ind in offspring:
                if pyrand.random() < mutpb:
                    toolbox.mutate(ind)
                    del ind.fitness.values

            # Evaluate invalid
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            for ind in invalid:
                ind.fitness.values = toolbox.evaluate(ind)

            # Survivor replacement
            pop = offspring

            # Logging
            fits = np.array([ind.fitness.values[0] for ind in pop], dtype=np.float64)
            writer.writerow([g, float(np.max(fits)), float(np.mean(fits)), float(np.median(fits))])

            if (g + 1) % max(1, generations // 10) == 0:
                print(f"[mlp_ga seed {seed}] Gen {g+1}/{generations}: best={np.max(fits):.3f} mean={np.mean(fits):.3f}")

    # Champion
    from deap import tools as deap_tools
    best_ind = deap_tools.selBest(pop, 1)[0]
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

def plot_experiment(exp_name, exp_csvs, rand_csvs, out_png, window=10, title=None, xlabel="Generation", ylabel="Fitness"):
    # Load runs
    exp_runs = [load_series(p) for p in exp_csvs]
    if len(exp_runs) == 0:
        print(f"[plot] No runs found for {exp_name}")
        return
    L = min(len(r) for r in exp_runs)
    exp_runs = [r[:L] for r in exp_runs]
    exp_sm = [moving_avg(r, window) for r in exp_runs]
    exp_mean = np.mean(exp_sm, axis=0)
    exp_std  = np.std(exp_sm, axis=0)

    rand_runs = [load_series(p) for p in rand_csvs]
    if len(rand_runs) > 0:
        Lr = min(len(r) for r in rand_runs)
        L = min(L, Lr)
        exp_sm = [r[:L] for r in exp_sm]
        exp_mean, exp_std = exp_mean[:L], exp_std[:L]
        rand_runs = [r[:L] for r in rand_runs]
        rand_sm = [moving_avg(r, window) for r in rand_runs]
        rand_mean = np.mean(rand_sm, axis=0)
        rand_std  = np.std(rand_sm, axis=0)
    else:
        rand_mean = None

    x = np.arange(len(exp_mean))
    plt.figure(figsize=(8,4.5))
    # individual runs (light)
    for r in exp_runs:
        plt.plot(np.arange(len(r[:L])), r[:L], linewidth=1, alpha=0.35)
    # mean + std
    plt.plot(x, exp_mean, linewidth=2.5, label=f"{exp_name} (mean MA)")
    plt.fill_between(x, exp_mean-exp_std, exp_mean+exp_std, alpha=0.25, label=f"{exp_name} ± std")

    # random baseline overlay
    if rand_mean is not None:
        plt.plot(x, rand_mean, linestyle="--", linewidth=2, label="Random (mean MA)")
        plt.fill_between(x, rand_mean-rand_std, rand_mean+rand_std, alpha=0.15, label="Random ± std")

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
         generations=10, pop_size=24, T=10.0, seed=[0,1,2],
         # MLP hyperparams
         mlp_hidden=32, cma_sigma0=0.7, mlp_freq_hz=1.0, mlp_alpha=0.25,
         # DE hyperparams
         de_F=0.7, de_CR=0.9, de_init_scale=0.5,
         # GA (DEAP) hyperparams
         ga_cxpb=0.6, ga_mutpb=0.3, ga_mut_sigma=0.3,
         # CPG hyperparams
         cpg_A_max=CTRL_RANGE, cpg_f_min=0.1, cpg_f_max=2.0, cpg_b_max=CTRL_RANGE, cpg_sigma0=0.5, cpg_alpha=0.2,
         # plotting
         plot_window=10):
    """Main function to run the simulation with random movements.
        Experiments:
      - "baseline": random baseline CSVs + viewer of baseline
      - "mlp_cma":  MLP + CMA-ES (Exp 1a) CSVs + viewer of champion
      - "mlp_de":   MLP + Differential Evolution (Exp 1b) CSVs + viewer of champion
      - "cpg_cma":  CPG + CMA-ES (Exp 2) CSVs + viewer of champion
            - "mlp_ga":   MLP + Genetic Algorithm (DEAP) (Exp 3)
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
            out_csv = f"./results/baseline/baseline_seed{s}.csv"
            run_random_baseline(out_csv, generations=generations, pop_size=pop_size, T=T, seed=s,
                                model=model, data=data, core_bind=core_bind)
            
         # 2) (Optional) visualize one baseline rollout in the viewer
        #    Use the SAME distribution as in the headless baseline.
        mujoco.mj_resetData(model, data)         # start fresh
        cb = random_baseline_callback_factory(seed[0], to_track)
        mujoco.set_mjcb_control(cb)
        viewer.launch(model=model, data=data)
        show_qpos_history(HISTORY)
        return  # don't also run the demo callback
    
    # MLP + CMA-ES (1a) 
    if experiment == "mlp_cma":
        best_theta_viz = None
        for i, s in enumerate(seed):
            out_csv = f"./results/mlp_cma/mlp_cma_seed{s}.csv"
            theta, best_fit = train_mlp_cma(
                out_csv=out_csv, generations=generations, pop_size=pop_size, T=T, seed=s,
                model=model, data=data, core_bind=core_bind,
                hidden=mlp_hidden, freq_hz=mlp_freq_hz, sigma0=cma_sigma0, alpha=mlp_alpha
            )
            print(f"[mlp_cma seed {s}] best_fitness={best_fit:.3f}")
            if i == 0: best_theta_viz = theta
        if best_theta_viz is not None:
            mujoco.mj_resetData(model, data)
            cb = mlp_viewer_callback_factory(best_theta_viz, to_track, hidden=mlp_hidden, freq_hz=mlp_freq_hz, alpha=mlp_alpha, dt=dt)
            mujoco.set_mjcb_control(cb)
            viewer.launch(model=model, data=data)
            show_qpos_history(HISTORY)
        return

    # MLP + DE (1b)
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

    #  CPG + CMA-ES (2)
    if experiment == "cpg_cma":
        best_theta_viz = None
        for i, s in enumerate(seed):
            out_csv = f"./results/cpg_cma/cpg_cma_seed{s}.csv"
            theta, best_fit = train_cpg_cma(
                out_csv=out_csv, generations=generations, pop_size=pop_size, T=T, seed=s,
                model=model, data=data, core_bind=core_bind,
                A_max=cpg_A_max, f_min=cpg_f_min, f_max=cpg_f_max, b_max=cpg_b_max, sigma0=cpg_sigma0, alpha=cpg_alpha
            )
            print(f"[cpg_cma seed {s}] best_fitness={best_fit:.3f}")
            if i == 0: best_theta_viz = theta
        if best_theta_viz is not None:
            mujoco.mj_resetData(model, data)
            cb = cpg_viewer_callback_factory(best_theta_viz, to_track, dt=dt, A_max=cpg_A_max, f_min=cpg_f_min, f_max=cpg_f_max, b_max=cpg_b_max, alpha=cpg_alpha)
            mujoco.set_mjcb_control(cb)
            viewer.launch(model=model, data=data)
            
        show_qpos_history(HISTORY)
        return
    

    #  MLP + GA (DEAP) (3)
    if experiment == "mlp_ga":
        best_theta_viz = None
        for i, s in enumerate(seed):
            out_csv = f"./results/mlp_ga/mlp_ga_seed{s}.csv"
            theta, best_fit = train_mlp_ga(
                out_csv=out_csv, generations=generations, pop_size=max(pop_size, 40), T=T, seed=s,
                model=model, data=data, core_bind=core_bind,
                hidden=mlp_hidden, freq_hz=mlp_freq_hz,
                cxpb=ga_cxpb, mutpb=ga_mutpb, mut_sigma=ga_mut_sigma, alpha=mlp_alpha
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

        # 1a MLP+CMA vs Random
        exp_csvs = sorted(glob.glob("./results/mlp_cma/mlp_cma_seed*.csv"))[:3]
        plot_experiment(
            exp_name="MLP + CMA-ES",
            exp_csvs=exp_csvs,
            rand_csvs=rand_csvs,
            out_png="./results/plots/exp1a_mlp_cma_vs_random.png",
            window=plot_window,
            title="Experiment 1a: MLP + CMA-ES vs Random"
        )

        # 1b MLP+DE vs Random
        exp_csvs = sorted(glob.glob("./results/mlp_de/mlp_de_seed*.csv"))[:3]
        plot_experiment(
            exp_name="MLP + DE",
            exp_csvs=exp_csvs,
            rand_csvs=rand_csvs,
            out_png="./results/plots/exp1b_mlp_de_vs_random.png",
            window=plot_window,
            title="Experiment 1b: MLP + DE vs Random"
        )

        # 2 CPG+CMA vs Random
        exp_csvs = sorted(glob.glob("./results/cpg_cma/cpg_cma_seed*.csv"))[:3]
        plot_experiment(
            exp_name="CPG + CMA-ES",
            exp_csvs=exp_csvs,
            rand_csvs=rand_csvs,
            out_png="./results/plots/exp2_cpg_cma_vs_random.png",
            window=plot_window,
            title="Experiment 2: CPG + CMA-ES vs Random"
        )
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
    experiment = "plot"  # "baseline" or "mlp_cma" or "mlp_de" or "cpg_cma" or "mlp_ga" or "plot"
    main(experiment=experiment)


