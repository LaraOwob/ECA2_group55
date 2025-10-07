import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==== pick your problem ====
problem_name = "rastrigin"   # "ackley" or "rastrigin"

# Use the right seeds for each folder
SEEDS = {
    "ackley":    [60, 61, 62, 63, 64, 65],
    "rastrigin": [48, 49, 50, 51, 52, 53],
}[problem_name]

BUDGETS = [50, 100, 150, 200, 250, 300]
ALGOS   = ["cmaes", "de", "pso"]

# ---------- locate data folder robustly ----------
def find_problem_dir(problem: str) -> Path:
    here = Path(__file__).resolve().parent
    candidates = [
        here / "testingAlgorithms" / problem,
        here.parent / "testingAlgorithms" / problem,
        here.parent.parent / "testingAlgorithms" / problem,
        Path.cwd() / "testingAlgorithms" / problem,
        Path.cwd().parent / "testingAlgorithms" / problem,
    ]
    for p in candidates:
        if p.exists():
            return p
    # very last resort: search up to 2 levels
    for root in [here, here.parent, Path.cwd(), Path.cwd().parent]:
        for t in root.rglob("testingAlgorithms"):
            q = t / problem
            if q.exists():
                return q
    raise FileNotFoundError(f"Could not find testingAlgorithms/{problem}")

BASE = find_problem_dir(problem_name)
print("Using data dir:", BASE)

# small helpers
def load_csv(budget: int, seed: int, algo: str) -> pd.DataFrame | None:
    f = BASE / f"{budget}_seed_{seed}_MLP_{algo}_results.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    if not {"generation", "best_fitness"} <= set(df.columns):
        return None
    return df

# --------- plotting: 1 fig per budget, colors=seeds, linestyles=algos ----------
seed_colors = plt.cm.tab10(np.linspace(0, 1, len(SEEDS)))  # 10-color palette

for budget in BUDGETS:
    plt.figure(figsize=(8, 5))
    any_plotted = False

    # plot each seed in its own color; algo with different linestyle
    for color, seed in zip(seed_colors, SEEDS):
        for algo, ls in zip(ALGOS, ["-", "--", ":"]):
            df = load_csv(budget, seed, algo)
            if df is None:
                continue
            plt.plot(
                df["generation"],
                df["best_fitness"],
                color=color,
                linestyle=ls,
                linewidth=1.8,
            )
            any_plotted = True

    if not any_plotted:
        plt.close()
        print(f"[warn] no data for budget {budget}")
        continue

    # Build 2 legends: one for seeds (colors), one for algos (linestyles)
    # Seed legend (dummy lines)
    seed_handles = [
        plt.Line2D([0], [0], color=c, lw=2) for c in seed_colors
    ]
    seed_labels = [f"seed {s}" for s in SEEDS]
    leg1 = plt.legend(seed_handles, seed_labels, title="Seeds", bbox_to_anchor=(1.02, 1), loc="upper left")

    # Algo legend (dummy lines)
    algo_handles = [
        plt.Line2D([0], [0], color="k", lw=2, linestyle=ls)
        for ls in ["-", "--", ":"]
    ]
    algo_labels = [a.upper() for a in ALGOS]
    leg2 = plt.legend(algo_handles, algo_labels, title="Algorithms", bbox_to_anchor=(1.02, 0.45), loc="upper left")

    plt.gca().add_artist(leg1)  # keep both legends
    plt.title(f"{problem_name.capitalize()} — Budget {budget}")
    plt.xlabel("generation")
    plt.ylabel("best fitness")
    plt.tight_layout()
    plt.show()