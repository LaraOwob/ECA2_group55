#!/usr/bin/env python3
"""
This script validates and compares different inner-loop optimization algorithms (CMA-ES and PSO) on real robot morphologies.
It fixes a small set of randomly sampled morphologies, then optimizes the neural network controller parameters for each morphology using both algorithms under the same computational budget.
The goal is to determine which optimizer is more efficient and reliable when training controllers for our robots.

Validation harness for inner-loop controller optimization.
- Fix 3-5 morphologies (NDE input vectors)
- For each morphology, compare CMA-ES vs PSO (or DE)
- Equalize function evaluations (FE) across algorithms
- Log trial-level results + print aggregate summary

USAGE:
  python validate_inner_loop.py --algos cmaes pso --morphs 5 --fe 256 --trials 5 --seed 777
"""

from __future__ import annotations
import argparse, csv, json, random, time
import cma
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple, List

from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import HighProbabilityDecoder
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.simulation.environments import OlympicArena
from ariel.utils.tracker import Tracker
from ariel.utils.runners import simple_runner
import mujoco as mj
import numpy as np
import torch
import torch.nn as nn

# ────────────────────────────────────────────────────────────────────────────────
# CONFIG / PATHS
# ────────────────────────────────────────────────────────────────────────────────
OUTDIR = Path("experiments/real_task_validation")
OUTDIR.mkdir(parents=True, exist_ok=True)
MORPH_FILE = OUTDIR / "val_morphologies.json"
RESULTS_CSV = OUTDIR / "results.csv"

# ────────────────────────────────────────────────────────────────────────────────
# ==== globals ====
SPAWN_POS        = [0, 0, 0.1]
NUM_OF_MODULES   = 30
TARGET_POSITION  = [5, 0, 0.5]   # z used only for distance calc / plotting
FINISH_RADIUS    = 0.15          # tweak if your arena uses a different threshold
CONTROLLER_HSIZE = 8             # same as your NNController default
GENOTYPE_SIZE    = 64            # you already use 64 in makeGenotype()
# ────────────────────────────────────────────────────────────────────────────────

def sample_random_nde_vectors(seed: int) -> List[np.ndarray]:
    """
    Return the 3 NDE input vectors that define a morphology.
    """
    rng = np.random.default_rng(seed)
    type_p = rng.random(GENOTYPE_SIZE).astype(np.float32)
    conn_p = rng.random(GENOTYPE_SIZE).astype(np.float32)
    rot_p  = rng.random(GENOTYPE_SIZE).astype(np.float32)
    return [type_p, conn_p, rot_p]


def build_robot_from_nde(vectors: List[np.ndarray], seed: int) -> Tuple[Any, Any, Any, Dict[str, Any]]:
    """
    Build the robot (mujoco model+data) and return a NN architecture spec for the controller.
    Replace this with your existing NDE->graph->json->mujoco pipeline and controller spec.
    Returns:
      world, model, data, nn_arch  (nn_arch can contain layer sizes, shapes, flat parameter size, etc.)
    """
    # genotype -> probability matrices -> graph -> mujoco spec
    nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES)
    p_mats = nde.forward(vectors)
    hpd = HighProbabilityDecoder(NUM_OF_MODULES)
    robot_graph = hpd.probability_matrices_to_graph(p_mats[0], p_mats[1], p_mats[2])
    core = construct_mjspec_from_graph(robot_graph)

    # spawn & compile
    mj.set_mjcb_control(None)
    world = OlympicArena()
    world.spawn(core.spec, spawn_position=SPAWN_POS, correct_for_bounding_box=False)
    model = world.spec.compile()
    data  = mj.MjData(model)

    nn_arch = {"nu": model.nu, "hidden": CONTROLLER_HSIZE}
    return world, model, data, nn_arch


# ---- controller utils ----
class NNController(nn.Module):
    def __init__(self, nu: int, hidden_size: int = CONTROLLER_HSIZE):
        super().__init__()
        self.nu = nu
        self.in_dim = nu + 4
        self.fc1 = nn.Linear(self.in_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, nu)
        self.tanh = nn.Tanh()
    def forward(self, prev_ctrl: torch.Tensor, t: float, target_pos: torch.Tensor, current_pos: torch.Tensor):
        w = 2 * torch.pi
        t_tensor = torch.tensor(t, dtype=torch.float32)
        time_feats = torch.stack([torch.sin(w*t_tensor), torch.cos(w*t_tensor)])
        delta = target_pos[:2] - current_pos[:2]
        obs = torch.cat([prev_ctrl, time_feats, delta], dim=0)
        h = self.tanh(self.fc1(obs)); h = self.tanh(self.fc2(h))
        y = self.tanh(self.fc3(h))
        return y  # scaling handled by actuator clip

def build_controller(nn_arch): return NNController(nu=nn_arch["nu"], hidden_size=nn_arch["hidden"])

def pack_params(net: nn.Module) -> np.ndarray:
    with torch.no_grad():
        return torch.cat([p.view(-1) for p in net.parameters()]).cpu().numpy().astype(np.float64)

def unpack_params(net: nn.Module, flat: np.ndarray) -> None:
    flat = torch.tensor(flat, dtype=torch.float32)
    i = 0
    with torch.no_grad():
        for p in net.parameters():
            n = p.numel()
            p.copy_(flat[i:i+n].view_as(p)); i += n

# ---- optimizer ----

def optimize_controller(algo: str, world, model, data, nn_arch, fe_budget: int, seed: int):
    rng  = np.random.default_rng(seed)
    ctrl = build_controller(nn_arch)
    x0   = pack_params(ctrl)
    sigma0 = 0.5

    # CMA-ES minimizes; we pass -fitness so it *maximizes* ours
    def obj(x: np.ndarray) -> float:
        unpack_params(ctrl, x)
        res = evaluate_robot(world, model, data, nn_arch, x, duration=20.0)
        return -float(res["fitness"])

    if algo.lower() == "cmaes":
        pop = 16
        iters = max(1, fe_budget // pop)
        es = cma.CMAEvolutionStrategy(x0, sigma0, {"popsize": pop, "seed": seed, "verb_disp": 0})
        fe_used = 0
        for _ in range(iters):
            cand = es.ask()
            vals = [obj(c) for c in cand]
            es.tell(cand, vals)
            fe_used += len(cand)
        return es.result.xbest.copy(), fe_used, {"algo": "cmaes"}

    elif algo.lower() == "pso":
        from pyswarm import pso
        dim = x0.size
        lb = np.full(dim, -3.0); ub = np.full(dim, 3.0)
        swarm = 32; iters = max(1, fe_budget // swarm)
        xbest, fbest = pso(lambda x: obj(np.asarray(x, dtype=np.float64)),
                           lb, ub, swarmsize=swarm, maxiter=iters, debug=False)
        return np.asarray(xbest, dtype=np.float64), swarm*iters, {"algo": "pso"}

    else:
        raise ValueError(f"Unknown algo {algo}")

# ---- evaluation ----

def evaluate_robot(world, model, data, nn_arch, flat_weights: np.ndarray, duration: float = 20.0): 
    """
    Evaluate one rollout with the given controller weights.
    Must return a dict containing at least:
      {
        "fitness": float,                 # finish-aware fitness you’re using
        "final_distance": float,          # distance covered
        "time_first_viable": float|inf,   # sim time when distance >= 1.0 (or inf if never)
      }
    """
    net = build_controller(nn_arch)
    unpack_params(net, flat_weights)
    net.eval()

    # tracker on "core"
    tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")
    tracker.setup(world.spec, data)

    # actuator clip
    ctrlrange = np.array(model.actuator_ctrlrange) if model.nu > 0 else np.zeros((0,2))
    lo = ctrlrange[:,0] if model.nu > 0 else np.array([])
    hi = ctrlrange[:,1] if model.nu > 0 else np.array([])

    target = np.array(TARGET_POSITION, dtype=np.float32)
    start_xy = None
    reached_t = None
    prev_ctrl = torch.zeros(nn_arch["nu"], dtype=torch.float32)

    def cb(m: mj.MjModel, d: mj.MjData):
        nonlocal start_xy, reached_t, prev_ctrl
        # current pos (use COM for robustness)
        com = np.array(d.subtree_com[0], dtype=np.float32)
        if start_xy is None: start_xy = com[:2].copy()

        with torch.no_grad():
            act = net(prev_ctrl=prev_ctrl, t=d.time,
                      target_pos=torch.tensor(target), current_pos=torch.tensor(com))
        prev_ctrl = act.clone()

        u = act.detach().numpy()
        if u.size and lo.size: u = np.clip(u, lo, hi)
        d.ctrl[:] = u

        # viable gait threshold: +1.0 m from start
        if reached_t is None:
            if np.linalg.norm(com[:2] - start_xy) >= 1.0:
                reached_t = d.time

    mj.mj_resetData(model, data)
    mj.set_mjcb_control(cb)
    simple_runner(model, data, duration=duration)

    # progress / finish-aware fitness
    com_final = np.array(data.subtree_com[0], dtype=np.float32)[:2]
    start_xy_safe = start_xy if start_xy is not None else np.array(SPAWN_POS[:2], dtype=np.float32)
    d0 = np.linalg.norm(start_xy_safe - target[:2])
    dT = np.linalg.norm(com_final - target[:2])
    progress = d0 - dT

    bonus = 0.0
    if dT <= FINISH_RADIUS:
        bonus = 1.0 + (duration - float(data.time))  # earlier finish => larger bonus

    forward_speed = max(0.0, (com_final[0] - SPAWN_POS[0]) / max(1e-6, duration))
    fitness = float(progress + 0.2*forward_speed + bonus)
    # The fitness function measures how well the robot moves toward the target.
    # It rewards progress toward the goal, adds a smaller reward for forward speed, and gives a bonus if the robot actually reaches 
    # the target before the time limit.

    return {
        "fitness": fitness,
        "final_distance": float(dT),
        "time_first_viable": float(reached_t if reached_t is not None else np.inf),
    }


# ────────────────────────────────────────────────────────────────────────────────
# Validation harness (no changes needed below this line, once hooks are wired)
# ────────────────────────────────────────────────────────────────────────────────

@dataclass
class TrialConfig:
    morphs: int = 5
    trials: int = 5
    seed: int = 777
    fe: int = 256
    algos: Tuple[str, ...] = ("cmaes", "pso")
    duration: float = 20.0


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    # try:
    #     torch.manual_seed(seed)
    # except Exception:
    #     pass


def ensure_morph_set(n: int, base_seed: int) -> List[List[np.ndarray]]:
    if MORPH_FILE.exists():
        with open(MORPH_FILE, "r") as f:
            arr = json.load(f)
        # back to np arrays
        morphs = [[np.array(v, dtype=float) for v in morph] for morph in arr]
        if len(morphs) >= n:
            return morphs[:n]
    # (re)create
    morphs = [sample_random_nde_vectors(base_seed + i) for i in range(n)]
    # save as lists (json serializable)
    enc = [[v.tolist() for v in m] for m in morphs]
    with open(MORPH_FILE, "w") as f:
        json.dump(enc, f)
    return morphs


def algo_fe_split(algo: str, total_fe: int) -> Tuple[int, int]:
    """
    Translate FE budget into (pop/swarm, iterations).
    Adjust to match your implementations.
    """
    if algo == "cmaes":
        pop = 16
        iters = max(1, total_fe // pop)
    elif algo in ("pso", "de"):
        pop = 32
        iters = max(1, total_fe // pop)
    else:
        raise ValueError(f"Unknown algo {algo}")
    return pop, iters


def main(cfg: TrialConfig):
    set_all_seeds(cfg.seed)

    morphs = ensure_morph_set(cfg.morphs, cfg.seed + 1000)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    for morph_id, nde_vecs in enumerate(morphs):
        # Build robot once per morphology (per trial we can rebuild to reseed physics if you prefer)
        for algo in cfg.algos:
            for r in range(cfg.trials):
                trial_seed = cfg.seed + 10_000 * morph_id + 100 * r
                set_all_seeds(trial_seed)

                # (Re)build model each trial to avoid stale state
                world, model, data, nn_arch = build_robot_from_nde(nde_vecs, seed=trial_seed)

                pop, iters = algo_fe_split(algo, cfg.fe)
                t0 = time.perf_counter()
                weights, fe_used, _hist = optimize_controller(
                    algo=algo,
                    world=world,
                    model=model,
                    data=data,
                    nn_arch=nn_arch,
                    fe_budget=cfg.fe,
                    seed=trial_seed
                )
                t1 = time.perf_counter()

                # evaluate the resulting controller once (after optimization)
                eval_res = evaluate_robot(world, model, data, nn_arch, weights, duration=cfg.duration)

                rows.append(dict(
                    morph_id=morph_id,
                    algo=algo,
                    trial=r,
                    seed=trial_seed,
                    fe_requested=cfg.fe,
                    fe_used=fe_used,
                    pop_or_swarm=pop,
                    iters=iters,
                    fitness=float(eval_res["fitness"]),
                    final_distance=float(eval_res["final_distance"]),
                    time_first_viable=float(eval_res["time_first_viable"]),
                    wall_time_sec=float(t1 - t0),
                ))

                print(f"[m{morph_id} {algo} r{r}] fit={rows[-1]['fitness']:.3f} "
                    f"dist={rows[-1]['final_distance']:.2f} "
                    f"t1st={rows[-1]['time_first_viable']:.2f} "
                    f"time={rows[-1]['wall_time_sec']:.2f}s")
    # write CSV
    with open(RESULTS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # quick summary
    import pandas as pd
    df = pd.DataFrame(rows)
    agg = (df.groupby("algo")
             .agg(best_mean=("fitness", "mean"),
                  best_std=("fitness", "std"),
                  t_viable_med=("time_first_viable", "median"),
                  time_mean=("wall_time_sec", "mean"),
                  n=("fitness", "size"))
             .reset_index()
           )
    print("\n===== Aggregate (all morphs × trials) =====")
    print(agg.sort_values("best_mean").to_string(index=False))

    # per-morph summary (optional, nice for appendix)
    per_m = (df.groupby(["morph_id", "algo"])
               .agg(best_mean=("fitness","mean"),
                    best_std=("fitness","std"),
                    t_viable_med=("time_first_viable","median"),
                    time_mean=("wall_time_sec","mean"),
                    n=("fitness","size"))
               .reset_index())
    per_m.to_csv(OUTDIR / "per_morph_summary.csv", index=False)
    print(f"\nSaved:\n  {RESULTS_CSV}\n  {OUTDIR/'per_morph_summary.csv'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--algos", nargs="+", default=["cmaes", "pso"],
                   help="inner-loop algorithms to compare (cmaes, pso, de)")
    p.add_argument("--morphs", type=int, default=5)
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--fe", type=int, default=256, help="function evaluation budget per trial")
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--duration", type=float, default=20.0)
    args = p.parse_args()
    cfg = TrialConfig(morphs=args.morphs, trials=args.trials, seed=args.seed,
                      fe=args.fe, algos=tuple(args.algos), duration=args.duration)
    main(cfg)