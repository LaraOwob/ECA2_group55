import pandas as pd
import glob
import os
from collections import Counter

def evaluate_methods(base_dir="./examples/testingAlgorithms"):
    """
    Compares DE, PSO, CMAES performance across different solution spaces.

    Returns:
        summary (dict): {solution_space_name: {"winner": method, "details": per_test_results}}
    """
    summary = {}
    print("in methods")
    method_names = ["ackley","rastrigin", "rosenbrock","schwefel", "sphere"]
    # --- Loop over each solution space folder ---
    for space_name in method_names:
        space_dir = os.path.join(base_dir, space_name)
        if not os.path.isdir(space_dir):
            print(f"⚠️  Skipping {space_name}: folder not found.")
            continue

        space_name = os.path.basename(space_dir)
        print(f"\n🔍 Evaluating solution space: {space_name}")

        # --- Group files by generation test ---
        csv_files = glob.glob(os.path.join(space_dir, "*_results.csv"))
        gen_tests = {}

        for path in csv_files:
            filename = os.path.basename(path)
            # Extract parts, e.g.  "50_seed_60_MLP_DE_results.csv"
            parts = filename.split("_")
            generation = parts[0]        # e.g. "50"
            method = parts[4]            # e.g. "DE", "PSO", "CMAES"
            gen_tests.setdefault(generation, {}).setdefault(method, path)

        # --- Compare methods per generation test ---
        generation_winners = {}

        for gen, methods in gen_tests.items():
            method_avgs = {}

            for method, csv_path in methods.items():
                df = pd.read_csv(csv_path)

                # Use average of best_fitness (lower = better)
                if "best_fitness" not in df.columns:
                    continue
                avg_fitness = df["best_fitness"].mean()
                method_avgs[method] = avg_fitness

            if method_avgs:
                winner = min(method_avgs, key=method_avgs.get)
                generation_winners[gen] = {
                    "winner": winner,
                    "method_avgs": method_avgs
                }
                print(f"Generation {gen}: {winner} wins with avg {method_avgs[winner]:.4f}")

        # --- Count wins across all generation tests ---
        all_winners = [v["winner"] for v in generation_winners.values()]
        most_common = Counter(all_winners).most_common(1)[0][0] if all_winners else None

        summary[space_name] = {
            "winner": most_common,
            "details": generation_winners
        }

        print(f"🏆 Overall winner for {space_name}: {most_common}")

    return summary

def main():
    print("lets start")
    results_summary = evaluate_methods()

if __name__ == "__main__":
    main()