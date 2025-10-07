"""Assignment 3 - Robot Olympics: Complete Implementation
Evolves robot morphology (body) with Differential Evolution
Optimizes controller (brain) with CMA-ES
"""

# Standard library
from pathlib import Path
from typing import Literal
import os
import csv

# Third-party libraries
#import matplotlib.pyplot as plt
import mujoco as mj
import numpy as np
import torch
import torch.nn as nn
import cma # type: ignore
from mujoco import viewer

# Local libraries
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
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

# Type Aliases
type ViewerTypes = Literal["launcher", "video", "simple", "no_control", "frame"]

# --- CONFIGURATION --- #
SEED = 42
RNG = np.random.default_rng(SEED)
CTRL_RANGE = 0.2

# --- DATA SETUP ---
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True, parents=True)

print(f"[INFO] Data directory: {DATA}")

# Global variables
SPAWN_POS = [-0.8, 0, 0.1]
NUM_OF_MODULES = 30
TARGET_POSITION = [5, 0, 0.5]

# ============================================================================
# NEURAL NETWORK CONTROLLER
# ============================================================================
class NNController(nn.Module):
    """Neural network controller with rich sensory input."""
    
    def __init__(self, nu: int, hidden_size: int = 32):
        super().__init__()
        self.nu = nu
        # Input: joint_pos + joint_vel + time_features + target_delta
        self.in_dim = nu + nu + 2 + 2  # qpos + qvel + sin/cos + delta_x/y
        self.hidden = hidden_size
        
        self.fc1 = nn.Linear(self.in_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, self.nu)
        self.tanh = nn.Tanh()
    
    def forward(self, qpos: torch.Tensor, qvel: torch.Tensor, t: float,
                target_pos: torch.Tensor, current_pos: torch.Tensor):
        """
        Args:
            qpos: Joint positions [nu]
            qvel: Joint velocities [nu]
            t: Current time
            target_pos: Target position [3]
            current_pos: Current robot position [3]
        Returns:
            action: Control signal [nu]
        """
        t_tensor = torch.tensor(t, dtype=torch.float32)
        w = 2 * torch.pi
        delta = target_pos[:2] - current_pos[:2]
        
        time_features = torch.stack([
            torch.sin(w * t_tensor), 
            torch.cos(w * t_tensor)
        ])
        
        obs = torch.cat([qpos, qvel, time_features, delta], dim=0)
        
        h = self.tanh(self.fc1(obs))
        h = self.tanh(self.fc2(h))
        y = self.tanh(self.fc3(h))
        
        return CTRL_RANGE * y


# ============================================================================
# BODY GENERATION (NDE)
# ============================================================================
def makeBody(num_modules: int = NUM_OF_MODULES, genotype=None): # type: ignore
    """Create robot body from genotype using NDE."""
    nde = NeuralDevelopmentalEncoding(number_of_modules=num_modules)
    p_matrices = nde.forward(genotype)
    hpd = HighProbabilityDecoder(num_modules)
    robot_graph = hpd.probability_matrices_to_graph(
        p_matrices[0], p_matrices[1], p_matrices[2]
    )
    core = construct_mjspec_from_graph(robot_graph)
    return core, robot_graph


def makeGenotype(num_modules: int = NUM_OF_MODULES, max_attempts: int = 100):
    """
    Generate a valid genotype with bias toward connectivity.
    Returns: (body_spec, genotype, robot_graph)
    """
    for attempt in range(max_attempts): # type: ignore
        genotype_size = 64
        
        # Bias toward valid, connected structures
        type_p = RNG.random(genotype_size).astype(np.float32) * 0.7 + 0.15
        conn_p = RNG.random(genotype_size).astype(np.float32) * 0.5 + 0.25
        rot_p = RNG.random(genotype_size).astype(np.float32)
        
        genotype = [type_p, conn_p, rot_p]
        
        try:
            body, graph = makeBody(num_modules, genotype)
            if is_valid_robot_spec(body.spec):
                # opzionale: stampa compatta durante init
                tmp_world = OlympicArena(); tmp_world.spawn(body.spec, SPAWN_POS, False); m = tmp_world.spec.compile()
                print(f"[INIT GEN] nu={m.nu}, bodies={m.nbody}, geoms={m.ngeom}")
                return body, genotype, graph
        except Exception as e: # type: ignore
            continue
    
    # Fallback: simple structure
    print("[WARNING] Using fallback genotype")
    type_p = np.full(64, 0.5, dtype=np.float32)
    conn_p = np.full(64, 0.6, dtype=np.float32)
    rot_p = np.full(64, 0.5, dtype=np.float32)
    genotype = [type_p, conn_p, rot_p]
    body, graph = makeBody(num_modules, genotype)
    return body, genotype, graph


def is_valid_robot_spec_strict(body_spec, min_nu: int = 8) -> bool: # type: ignore
    try:
        world = OlympicArena()
        world.spawn(body_spec.spec, spawn_position=SPAWN_POS, correct_for_bounding_box=True)
        model = world.spec.compile()
        apply_stability_patch(model)
        # controlli più robusti
        has_joints = model.nu >= min_nu
        has_bodies = model.nbody > 1
        has_geoms  = model.ngeom > 0
        return bool(has_joints and has_bodies and has_geoms)
    except Exception:
        return False


def is_valid_robot_spec(mj_spec) -> bool: # type: ignore
    """Check if robot morphology is valid."""
    try:
        if mj_spec is None or mj_spec.worldbody is None:
            return False
        has_geoms = hasattr(mj_spec.worldbody, "geoms") and len(mj_spec.worldbody.geoms) > 0
        has_bodies = hasattr(mj_spec.worldbody, "bodies") and len(mj_spec.worldbody.bodies) > 0
        return has_geoms or has_bodies
    except Exception:
        return False


def repair_connectivity(gen):  # type: ignore # gen is [type_p, conn_p, rot_p]
    gen[0] = np.clip(gen[0], 0.0, 1.0)
    gen[1] = np.clip(gen[1], 0.0, 1.0)
    gen[2] = np.clip(gen[2], 0.0, 1.0)
    # keep a floor on connection probabilities
    gen[1] = np.maximum(gen[1], 0.6)   # don’t let them collapse
    return gen


def apply_stability_patch(model): # type: ignore
    # smaller timestep + better integrator
    model.opt.timestep = max(0.001, float(model.opt.timestep))  # e.g. 0.001–0.002
    try:
        model.opt.integrator = mj.mjtIntegrator.mjINT_RK4
    except Exception:
        pass  # if RK4 not available, ignore

    # stronger solver
    model.opt.iterations   = max(50, model.opt.iterations)
    model.opt.ls_iterations = max(10, model.opt.ls_iterations)
    model.opt.warmstart = 1

    # add damping to every DOF (prevents high-frequency explosions)
    import numpy as _np
    if hasattr(model, "dof_damping"):
        model.dof_damping[:] = _np.maximum(model.dof_damping, 0.5)

    # clamp actuator ctrl range to something sane
    if getattr(model, "nu", 0) > 0 and hasattr(model, "actuator_ctrlrange"):
        lohi = model.actuator_ctrlrange
        lohi[:, 0] = _np.maximum(lohi[:, 0], -0.5)
        lohi[:, 1] = _np.minimum(lohi[:, 1],  0.5)

    # a touch of joint/viscous friction helps too
    if hasattr(model, "dof_frictionloss"):
        model.dof_frictionloss[:] = _np.maximum(model.dof_frictionloss, 0.01)

# ============================================================================
# CONTROLLER OPTIMIZATION (CMA-ES)
# ============================================================================
def flatten_params(nn: nn.Module) -> np.ndarray:
    """Flatten all neural network parameters into a single vector."""
    return np.concatenate([p.detach().cpu().numpy().ravel() for p in nn.parameters()])


def unflatten_params(nn: nn.Module, flat: np.ndarray) -> None:
    """Load flattened parameters back into neural network."""
    offset = 0
    with torch.no_grad():
        for p in nn.parameters():
            numel = p.numel()
            p.copy_(torch.tensor(
                flat[offset:offset+numel].reshape(p.shape), 
                dtype=p.dtype
            ))
            offset += numel


def _bind_geom_name(model: mj.MjModel) -> str:
    """Find valid geom name for tracking."""
    try:
        _ = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "core")
        return "core"
    except Exception:
        return mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, 0)


def nn_obj_control(nn_obj: NNController, model: mj.MjModel, data: mj.MjData, 
                   target_pos: np.ndarray, geom_id: int):
    current_pos = torch.tensor(data.geom_xpos[geom_id], dtype=torch.float32)
    qpos = torch.tensor(data.qpos[:nn_obj.nu], dtype=torch.float32)
    qvel = torch.tensor(data.qvel[:nn_obj.nu], dtype=torch.float32)
    action = nn_obj(
        qpos=qpos, qvel=qvel, t=data.time,
        target_pos=torch.tensor(target_pos, dtype=torch.float32),
        current_pos=current_pos
    )
    return action.detach().numpy()


def evaluateFitness(model: mj.MjModel,
                    world: OlympicArena,
                    nn_obj: NNController,
                    target_pos: np.ndarray = np.array(TARGET_POSITION, dtype=np.float32),
                    duration: float = 8.0) -> float:
    try:
        mj.set_mjcb_control(None)
        data = mj.MjData(model)
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)

        def _control_cb(m, d):  # type: ignore
            com = np.array(d.subtree_com[0], dtype=np.float32)
            qpos = torch.tensor(d.qpos[:nn_obj.nu], dtype=torch.float32)
            qvel = torch.tensor(d.qvel[:nn_obj.nu], dtype=torch.float32)
            act = nn_obj(
                qpos=qpos, qvel=qvel, t=d.time,
                target_pos=torch.tensor(target_pos, dtype=torch.float32),
                current_pos=torch.tensor(com, dtype=torch.float32),
            )
            return act.detach().numpy()

        # Tracker minimale (richiesto da Controller)
        tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_BODY, name_to_bind="")
        tracker.setup(world.spec, data)

        ctrl = Controller(controller_callback_function=_control_cb, tracker=tracker)
        mj.set_mjcb_control(ctrl.set_control)

        try:
            simple_runner(model, data, duration=duration)
            if (not np.isfinite(data.qacc).all()) or (not np.isfinite(data.qpos).all()):
                return -1e6
        except Exception as e:
            print(f"[WARNING] Simulation failed: {e}")
            return -1e6

        xc, yc, zc = map(float, data.subtree_com[0])
        xt, yt, zt = map(float, target_pos)
        dist = np.sqrt((xt - xc)**2 + (yt - yc)**2 + (zt - zc)**2)
        fit = -dist
        if not np.isfinite(fit):
            return -1e6
        return float(fit)
    except Exception as e:
        print(f"[WARNING] evaluateFitness failed: {e}")
        return -1e6

def optimize_controller_with_cmaes(nn_obj: NNController,  # type: ignore
                                   model: mj.MjModel, 
                                   world: OlympicArena,
                                   target_pos: np.ndarray, 
                                   duration: float,
                                   iterations: int = 12, 
                                   sigma: float = 0.3, 
                                   popsize: int = 16) -> tuple: # type: ignore
    """
    Optimize neural network controller using CMA-ES.
    Returns: (best_params, best_fitness)
    """
    x0 = flatten_params(nn_obj)
    
    def fitness_fn(x): # type: ignore
        """CMA-ES fitness function (minimizes, so negate)."""
        unflatten_params(nn_obj, x)
        f = evaluateFitness(model, world, nn_obj, target_pos=target_pos, duration=duration)
        return -f  # CMA-ES minimizes
    
    # Run CMA-ES
    es = cma.CMAEvolutionStrategy(
        x0, 
        sigma, 
        {
            'popsize': popsize, 
            'maxiter': iterations, 
            'verb_disp': 0,
            'verb_log': 0
        }
    )
    
    es.optimize(fitness_fn)
    
    best_x = es.result.xbest
    best_f = -es.result.fbest  # Convert back to positive fitness
    
    # Load best parameters
    unflatten_params(nn_obj, best_x)
    
    return best_x, best_f


# ============================================================================
# INDIVIDUAL CREATION
# ============================================================================
def makeIndividual(body_spec, genotype, graph, # type: ignore
                   target_pos=np.array(TARGET_POSITION, dtype=np.float32), # type: ignore
                   duration=8.0, # type: ignore
                   cmaes_iterations=12, # type: ignore
                   cmaes_popsize=16): # type: ignore
    """
    Create and evaluate an individual robot.
    Builds world, optimizes controller with CMA-ES, returns complete individual.
    """
    # Validate morphology
    if not is_valid_robot_spec_strict(body_spec):
        return {
            "genotype": genotype,
            "robot_spec": body_spec,
            "robot_graph": graph,
            "nn": None,
            "controller_params": np.array([]),
            "fitness": -1e6,
        }
    
    # Create fresh simulation context
    mj.set_mjcb_control(None)
    world = OlympicArena()
    
    try:
        world.spawn(
            body_spec.spec, 
            spawn_position=SPAWN_POS, 
            correct_for_bounding_box=True
        )
        model = world.spec.compile()
        apply_stability_patch(model)
    except Exception as e:
        print(f"[ERROR] Failed to spawn/compile robot: {e}")
        return {
            "genotype": genotype,
            "robot_spec": body_spec,
            "robot_graph": graph,
            "nn": None,
            "controller_params": np.array([]),
            "fitness": -1e6,
        }
    
    # Create neural network controller
    nn_obj = NNController(nu=model.nu, hidden_size=32)
    
    # Optimize with CMA-ES
    print(f"    [CMA-ES] Starting optimization (nu={model.nu})...")
    best_params, best_fitness = optimize_controller_with_cmaes(
        nn_obj, model, world, 
        target_pos=target_pos, 
        duration=duration,
        iterations=cmaes_iterations, 
        sigma=0.3, 
        popsize=cmaes_popsize
    )
    print(f"    [CMA-ES] Completed. Fitness: {best_fitness:.3f}")
    
    return {
        "genotype": genotype,
        "robot_spec": body_spec,
        "robot_graph": graph,
        "nn": nn_obj,
        "controller_params": best_params,
        "fitness": best_fitness,
    }


# ============================================================================
# DIFFERENTIAL EVOLUTION (BODY EVOLUTION)
# ============================================================================
def mutation_DE(a, b, c, F): # type: ignore
    """Differential mutation: mutant = a + F * (b - c)."""
    mutant_genotype = []
    for vec_idx in range(3):
        diff = b[vec_idx] - c[vec_idx]
        mutant_vec = a[vec_idx] + F * diff
        mutant_vec = np.clip(mutant_vec, 0.0, 1.0)  # Keep in valid range
        mutant_genotype.append(mutant_vec)
    return mutant_genotype


def crossover_DE(parent, mutant, CR, vector_size=3, gene_size=64): # type: ignore
    """Binomial crossover between parent and mutant."""
    rng = np.random.default_rng()
    crossover_genotype = []
    
    for vec_idx in range(vector_size):
        mask = rng.random(gene_size) < CR
        # Ensure at least one gene from mutant
        mask[rng.integers(0, gene_size)] = True
        trial_vec = np.where(mask, mutant[vec_idx], parent[vec_idx])
        crossover_genotype.append(trial_vec)
    
    return crossover_genotype


def train_body_de(out_csv,  # type: ignore
                  generations=30,  # type: ignore
                  pop_size=15,  # type: ignore
                  num_modules=NUM_OF_MODULES, # type: ignore
                  F=0.7,  # type: ignore
                  CR=0.9, # type: ignore
                  target_pos=np.array(TARGET_POSITION, dtype=np.float32), # type: ignore
                  duration=8.0, # type: ignore
                  cmaes_iterations=12, # type: ignore
                  cmaes_popsize=16): # type: ignore
    """
    Differential Evolution for morphology evolution.
    Each individual's controller is optimized with CMA-ES.
    """
    rng = np.random.default_rng(SEED)
    
    print("\n" + "="*70)
    print("DIFFERENTIAL EVOLUTION - BODY + BRAIN OPTIMIZATION")
    print("="*70)
    print(f"Population size: {pop_size}")
    print(f"Generations: {generations}")
    print(f"Num modules: {num_modules}")
    print(f"F: {F}, CR: {CR}")
    print(f"CMA-ES iterations: {cmaes_iterations}, popsize: {cmaes_popsize}")
    print("="*70 + "\n")
    
    # ========================================================================
    # INITIALIZATION
    # ========================================================================
    print("[INIT] Generating initial population...")
    population = []  # List of genotypes
    individuals = []  # List of full individual dicts
    fitnesses = []
    
    for i in range(pop_size):
        print(f"\n[INIT] Individual {i+1}/{pop_size}")
        
        body, genotype, graph = makeGenotype(num_modules)
        
        if not is_valid_robot_spec_strict(body):
            print(f"   Invalid morphology, assigning poor fitness")
            population.append(genotype)
            individuals.append(None)
            fitnesses.append(-1e6)
            continue
        
        # Evaluate with CMA-ES optimized controller
        indiv = makeIndividual(
            body, genotype, graph,
            target_pos=target_pos, 
            duration=duration,
            cmaes_iterations=cmaes_iterations,
            cmaes_popsize=cmaes_popsize
        )
        
        population.append(genotype)
        individuals.append(indiv)
        fitnesses.append(indiv["fitness"])
        
        print(f"  ✓ Fitness: {indiv['fitness']:.3f}")
    
    fitnesses = np.array(fitnesses)
    
    # Track best
    best_idx = int(np.argmax(fitnesses))
    best_individual = individuals[best_idx]
    best_fitness = float(fitnesses[best_idx])
    
    print(f"\n[INIT] Initialization complete!")
    print(f"  Best fitness: {best_fitness:.3f}")
    print(f"  Mean fitness: {np.mean(fitnesses):.3f}")
    print(f"  Median fitness: {np.median(fitnesses):.3f}")
    
    # ========================================================================
    # CSV LOGGING
    # ========================================================================
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["generation", "best_fitness", "mean_fitness", "median_fitness"])
        writer.writerow([0, best_fitness, np.mean(fitnesses), np.median(fitnesses)])
        
        # ====================================================================
        # EVOLUTION LOOP
        # ====================================================================
        for g in range(generations):
            print("\n" + "="*70)
            print(f"GENERATION {g+1}/{generations}")
            print("="*70)
            
            for i in range(pop_size):
                # Skip terrible individuals
                if fitnesses[i] < -1e5:
                    print(f"[Gen {g+1}] Individual {i}: Skipping (terrible fitness)")
                    continue
                
                print(f"\n[Gen {g+1}] Individual {i+1}/{pop_size}")
                
                # Select 3 different individuals for mutation
                candidates = [idx for idx in range(pop_size) if idx != i]
                a, b, c = rng.choice(candidates, size=3, replace=False)
                
                # Mutation
                mutant_genotype = mutation_DE(
                    population[a], 
                    population[b], 
                    population[c], 
                    F
                )
                
                # Crossover
                trial_genotype = crossover_DE(
                    population[i], 
                    mutant_genotype, 
                    CR, 
                    vector_size=3, 
                    gene_size=64
                )
                
                # Repair connectivity if needed
                trial_genotype = repair_connectivity(trial_genotype)

                # Evaluate trial
                trial_body, trial_graph = makeBody(num_modules, trial_genotype)
                
                if not is_valid_robot_spec_strict(trial_body, min_nu=8):
                    # fallback: prova una nuova genotipizzazione “fresh” (diversity rescue)
                    fb_body, fb_gen, fb_graph = makeGenotype(num_modules)
                    if is_valid_robot_spec_strict(fb_body, min_nu=8):
                        trial_body, trial_genotype, trial_graph = fb_body, fb_gen, fb_graph
                    else:
                        print(f"   Trial invalid (and fallback invalid), keeping parent (fitness: {fitnesses[i]:.3f})")
                        continue
                
                trial_indiv = makeIndividual(
                    trial_body, trial_genotype, trial_graph,
                    target_pos=target_pos, 
                    duration=duration,
                    cmaes_iterations=cmaes_iterations,
                    cmaes_popsize=cmaes_popsize
                )
                
                trial_fitness = trial_indiv["fitness"]
                
                # Selection: replace if better
                if trial_fitness > fitnesses[i]:
                    population[i] = trial_genotype
                    individuals[i] = trial_indiv
                    fitnesses[i] = trial_fitness
                    print(f"  ✓ IMPROVED: {trial_fitness:.3f} (was {fitnesses[i]:.3f})")
                else:
                    print(f"  → No improvement: {trial_fitness:.3f} <= {fitnesses[i]:.3f}")
            
            # Update best
            gen_best_idx = int(np.argmax(fitnesses))
            gen_best = float(fitnesses[gen_best_idx])
            
            if gen_best > best_fitness:
                best_fitness = gen_best
                best_individual = individuals[gen_best_idx]
                print(f"\n  NEW BEST FITNESS: {best_fitness:.3f}")
            
            # Log statistics
            gen_mean = float(np.mean(fitnesses))
            gen_median = float(np.median(fitnesses))
            writer.writerow([g+1, gen_best, gen_mean, gen_median])
            f.flush()
            
            print(f"\n[Gen {g+1}] Summary:")
            print(f"  Best:   {gen_best:.3f}")
            print(f"  Mean:   {gen_mean:.3f}")
            print(f"  Median: {gen_median:.3f}")
    
    print("\n" + "="*70)
    print("EVOLUTION COMPLETE")
    print("="*70)
    print(f"Final best fitness: {best_fitness:.3f}")
    print("="*70 + "\n")
    
    return best_individual, best_fitness


# ============================================================================
# BASELINE EXPERIMENT
# ============================================================================
def baseline_experiment(out_csv,  # type: ignore
                       pop_size=20,  # type: ignore
                       num_modules=NUM_OF_MODULES,  # type: ignore
                       duration=8.0, # type: ignore
                       target_pos=np.array(TARGET_POSITION, dtype=np.float32)): # type: ignore
    """
    Baseline: Random morphologies with RANDOM (untrained) controllers.
    """
    print("\n" + "="*70)
    print("BASELINE EXPERIMENT - Random Controllers")
    print("="*70)
    
    best_fitness = -1e6
    best_individual = None
    all_fitnesses = []
    
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["individual", "fitness"])
        
        for i in range(pop_size):
            print(f"\n[Baseline] Individual {i+1}/{pop_size}")
            
            body, genotype, graph = makeGenotype(num_modules)
            
            if not is_valid_robot_spec_strict(body):
                writer.writerow([i, -1e6])
                all_fitnesses.append(-1e6)
                print(f"  Invalid morphology")
                continue
            
            # Create world and model
            mj.set_mjcb_control(None)
            world = OlympicArena()
            
            try:
                world.spawn(body.spec, spawn_position=SPAWN_POS, 
                           correct_for_bounding_box=True)
                model = world.spec.compile()
                apply_stability_patch(model)
            except Exception as e:
                print(f"  Spawn/compile failed: {e}")
                writer.writerow([i, -1e6])
                all_fitnesses.append(-1e6)
                continue
            
            # Random controller (NO CMA-ES)
            nn_obj = NNController(nu=model.nu, hidden_size=32)
            
            # Evaluate with random weights
            fitness = evaluateFitness(model, world, nn_obj, 
                                     target_pos=target_pos, duration=duration)
            
            writer.writerow([i, fitness])
            all_fitnesses.append(fitness)
            
            print(f"  Fitness: {fitness:.3f}")
            
            if fitness > best_fitness:
                best_fitness = fitness
                best_individual = {
                    "genotype": genotype,
                    "robot_spec": body,
                    "robot_graph": graph,
                    "nn": nn_obj,
                    "fitness": fitness
                }
    
    print(f"\n[Baseline] Complete!")
    print(f"  Best fitness: {best_fitness:.3f}")
    print(f"  Mean fitness: {np.mean(all_fitnesses):.3f}")
    print(f"  Median fitness: {np.median(all_fitnesses):.3f}")
    
    return best_individual, best_fitness


# ============================================================================
# VISUALIZATION
# ============================================================================
def visualize_best_robot(individual_dict,  # type: ignore
                        target_pos=np.array(TARGET_POSITION, dtype=np.float32), # type: ignore
                        duration=8.0, # type: ignore
                        mode: ViewerTypes = "launcher"):
    """
    Visualize the best robot with its optimized controller.
    """
    if individual_dict is None or individual_dict["nn"] is None:
        print("[ERROR] Cannot visualize: invalid individual")
        return
    
    print(f"\n[Visualize] Loading best robot (fitness: {individual_dict['fitness']:.3f})")
    
    # Setup world
    mj.set_mjcb_control(None)
    world = OlympicArena()
    world.spawn(
        individual_dict["robot_spec"].spec,
        spawn_position=SPAWN_POS,
        correct_for_bounding_box=True
    )
    model = world.spec.compile()
    apply_stability_patch(model)
    data = mj.MjData(model)
    
    # Setup controller
    nn_obj = individual_dict["nn"]
    
    bind_name = _bind_geom_name(model)
    geom_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, bind_name)

    def cb(m, d): # type: ignore
        return nn_obj_control(nn_obj, m, d, target_pos, geom_id)
    
    bind_name = _bind_geom_name(model)
    tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind=bind_name)
    tracker.setup(world.spec, data)
    ctrl = Controller(controller_callback_function=cb, tracker=tracker)
    
    mj.set_mjcb_control(ctrl.set_control)
    
    # Visualize
    if mode == "launcher":
        print("[Visualize] Launching interactive viewer...")
        viewer.launch(model=model, data=data)
    elif mode == "video":
        print(f"[Visualize] Recording video to {DATA / 'videos'}...")
        video_recorder = VideoRecorder(output_folder=str(DATA / "videos"))
        video_renderer(model, data, duration=duration, video_recorder=video_recorder)
        print("[Visualize] Video saved!")
    elif mode == "simple":
        print("[Visualize] Running simulation...")
        simple_runner(model, data, duration=duration)
    elif mode == "frame":
        print(f"[Visualize] Saving frame to {DATA / 'robot.png'}...")
        single_frame_renderer(model, data, save=True, save_path=str(DATA / "robot.png"))
        print("[Visualize] Frame saved!")


# ============================================================================
# MAIN
# ============================================================================
def main() -> None:
    """Main entry point for robot evolution."""
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    MODE = "evolve"  # Options: "evolve", "baseline", "visualize"

    # Evolution parameters
    GENERATIONS = 30
    POP_SIZE = 30
    NUM_MODULES = NUM_OF_MODULES  # use global
    F = 0.7
    CR = 0.9

    # CMA-ES parameters
    CMAES_ITERATIONS = 12
    CMAES_POPSIZE = 16

    # Simulation parameters
    TARGET_POS = np.array(TARGET_POSITION, dtype=np.float32)  # use global target
    DURATION = 8.0

    # Visualization
    VIEW_MODE: ViewerTypes = "launcher"  # "launcher", "video", "simple", "frame"
    
    # ========================================================================
    # EXECUTION
    # ========================================================================
    
    if MODE == "baseline": # type: ignore
        print("\n RUNNING BASELINE EXPERIMENT")
        best_indiv, best_fit = baseline_experiment(
            out_csv=str(DATA / "baseline_results.csv"),
            pop_size=20,
            num_modules=NUM_MODULES,
            duration=DURATION,
            target_pos=TARGET_POS
        )
        print(f"\n Baseline complete! Best fitness: {best_fit:.3f}")
        
        # Save baseline result
        if best_indiv is not None:
            np.save(DATA / "baseline_best_fitness.npy", best_fit)
    
    elif MODE == "evolve":
        print("\n RUNNING EVOLUTIONARY EXPERIMENT")
        best_individual, best_fitness = train_body_de(
            out_csv=str(DATA / "evolution_results.csv"),
            generations=GENERATIONS,
            pop_size=POP_SIZE,
            num_modules=NUM_MODULES,
            F=F,
            CR=CR,
            target_pos=TARGET_POS,
            duration=DURATION,
            cmaes_iterations=CMAES_ITERATIONS,
            cmaes_popsize=CMAES_POPSIZE
        )
        
        print(f"\n Evolution complete! Best fitness: {best_fitness:.3f}")
        
        # ====================================================================
        # SAVE BEST ROBOT FOR SUBMISSION
        # ====================================================================
        if best_individual is not None and best_individual["robot_graph"] is not None:
            # Save robot morphology as JSON
            robot_json_path = DATA / "best_robot.json"
            save_graph_as_json(
                best_individual["robot_graph"],
                str(robot_json_path)
            )
            print(f"\n Saved robot morphology: {robot_json_path}")
            
            # Save controller parameters
            controller_path = DATA / "best_controller_params.npy"
            np.save(controller_path, best_individual["controller_params"])
            print(f" Saved controller params: {controller_path}")
            
            # Save genotype (lightweight)
            genotype_path = DATA / "best_genotype.npz"
            np.savez(
                genotype_path,
                type_p=best_individual["genotype"][0],
                conn_p=best_individual["genotype"][1],
                rot_p=best_individual["genotype"][2],
            )
            print(f" Saved genotype: {genotype_path}")

            # Save fitness for reference
            fitness_path = DATA / "best_fitness.npy"
            np.save(fitness_path, best_fitness)
            print(f" Saved fitness: {fitness_path}")
            
            print("\n" + "="*70)
            print("SUBMISSION FILES READY:")
            print(f"  1. Robot JSON:  {robot_json_path}")
            print(f"  2. Controller:  {controller_path}")
            print("="*70)
            
            # Visualize best robot
            print(f"\n Visualizing best robot (mode: {VIEW_MODE})...")
            visualize_best_robot(
                best_individual,
                target_pos=TARGET_POS,
                duration=DURATION,
                mode=VIEW_MODE
            )
        else:
            print("\n No valid robot found!")
    
    elif MODE == "visualize":
        print("\n LOADING AND VISUALIZING SAVED ROBOT")

        genotype_path   = DATA / "best_genotype.npz"
        controller_path = DATA / "best_controller_params.npy"

        if not genotype_path.exists() or not controller_path.exists():
            print(" Missing saved files. Expected:")
            print(f"   {genotype_path}")
            print(f"   {controller_path}")
            print("   Run with MODE='evolve' first!")
            return

        # --- Rebuild morphology from genotype ---
        g = np.load(genotype_path)
        genotype = [g["type_p"], g["conn_p"], g["rot_p"]]
        body_spec, robot_graph = makeBody(NUM_OF_MODULES, genotype)

        # --- Spawn world/model ---
        mj.set_mjcb_control(None)
        world = OlympicArena()
        world.spawn(body_spec.spec, spawn_position=SPAWN_POS, correct_for_bounding_box=True)
        model = world.spec.compile()
        apply_stability_patch(model)

        # --- Rebuild controller and load params ---
        nn_obj = NNController(nu=model.nu, hidden_size=32)
        best_params = np.load(controller_path)
        unflatten_params(nn_obj, best_params)

        # Package a lightweight dict and reuse your viewer util
        indiv_for_viz = {
            "robot_spec": body_spec,
            "nn": nn_obj,
            "fitness": float(np.load(DATA / "best_fitness.npy")),
            "robot_graph": robot_graph,   # optional, not required by visualize
            "genotype": genotype,         # optional
            "controller_params": best_params,
        }

        # Choose how you want to watch it
        VIEW_MODE: ViewerTypes = "launcher"  # type: ignore # "launcher" | "video" | "simple" | "frame"
        visualize_best_robot(
            indiv_for_viz,
            target_pos=TARGET_POS,
            duration=8.0,
            mode=VIEW_MODE
        )
    
    else:
        print(f" Unknown MODE: {MODE}")
        print("   Available modes: 'evolve', 'baseline', 'visualize'")


#if __name__ == "__main__":
#    main()

def visualize_evolved_robot():
    """Load and visualize saved robot - using the saved JSON directly."""
    
    data_dir = Path("__data__/run_full_ea")
    
    # Load the ROBOT JSON directly (not rebuilding from genotype)
    robot_json = data_dir / "best_robot.json"
    controller_params = data_dir / "best_controller_params.npy"
    
    if not robot_json.exists():
        print(f"Missing {robot_json}")
        return
    
    # Load the robot graph from JSON
    import json
    from networkx import node_link_graph
    
    with open(robot_json, 'r') as f:
        graph_data = json.load(f)
    robot_graph = node_link_graph(graph_data, edges="edges")
    
    # Construct robot from graph
    core = construct_mjspec_from_graph(robot_graph)
    
    # Setup world
    mj.set_mjcb_control(None)
    world = OlympicArena()
    
    # Add target marker
    world.spec.worldbody.add_geom(
        type=mj.mjtGeom.mjGEOM_SPHERE,
        name="target_marker",
        size=[0.15, 0.0, 0.0],
        pos=[5.0, 0.0, 0.5],
        rgba=[1, 0, 0, 1],
        contype=0,
        conaffinity=0
    )
    
    world.spawn(core.spec, spawn_position=[-0.8, 0, 0.1], 
               correct_for_bounding_box=True)
    model = world.spec.compile()
    apply_stability_patch(model)
    data = mj.MjData(model)
    
    print(f"Robot has {model.nu} joints")
    
    # Create matching controller
    nn_obj = NNController(nu=model.nu, hidden_size=32)
    
    # Load parameters
    params = np.load(controller_params)
    print(f"Controller has {len(params)} parameters")
    print(f"Network expects {sum(p.numel() for p in nn_obj.parameters())} parameters")
    
    # Verify match before loading
    expected = sum(p.numel() for p in nn_obj.parameters())
    if len(params) != expected:
        print(f"\n MISMATCH: Saved controller ({len(params)}) doesn't match robot ({expected})")
        print("The saved controller was trained on a different morphology.")
        return
    
    unflatten_params(nn_obj, params)
    
    # Setup control
    target_pos = np.array([5.0, 0.0, 0.5], dtype=np.float32)
    
    bind_name = _bind_geom_name(model)  # usually "core"
    geom_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, bind_name)
    def control_cb(m, d):  # type: ignore
        return nn_obj_control(nn_obj, m, d, target_pos, geom_id)
    
    tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")
    tracker.setup(world.spec, data)
    ctrl = Controller(controller_callback_function=control_cb, tracker=tracker)
    mj.set_mjcb_control(ctrl.set_control)
    
    print(f"\n Loaded robot with fitness: {np.load(data_dir / 'best_fitness.npy'):.3f}")
    print("Launching viewer...")
    
    viewer.launch(model=model, data=data)


if __name__ == "__main__":
    visualize_evolved_robot()