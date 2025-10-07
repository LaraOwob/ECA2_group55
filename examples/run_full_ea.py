# run_full_ea.py
from pathlib import Path
import numpy as np

# ⬇️ Change this to your module filename (without .py)
# e.g. from A3_template_no_stagnation import train_body_de, save_graph_as_json, NUM_OF_MODULES, TARGET_POSITION
from A3_template_no_stagnation_v2 import (
    train_body_de,
    save_graph_as_json,
    NUM_OF_MODULES,
    TARGET_POSITION,
)

# ---- Script-local config (uses your globals where needed) ----
GENERATIONS = 30
POP_SIZE = 30
F = 0.7
CR = 0.9

CMAES_ITERATIONS = 12
CMAES_POPSIZE   = 16

DURATION   = 8.0
TARGET_POS = np.array(TARGET_POSITION, dtype=np.float32)  # use your global target

# ---- Output dir based on this script’s name ----
SCRIPT_NAME = __file__.split("/")[-1][:-3]
DATA = Path.cwd() / "__data__" / SCRIPT_NAME
DATA.mkdir(parents=True, exist_ok=True)

def main():
    print("\n== RUNNING EVOLUTIONARY EXPERIMENT ==")
    best_individual, best_fitness = train_body_de(
        out_csv=str(DATA / "evolution_results.csv"),
        generations=GENERATIONS,
        pop_size=POP_SIZE,
        num_modules=NUM_OF_MODULES,
        F=F,
        CR=CR,
        target_pos=TARGET_POS,
        duration=DURATION,
        cmaes_iterations=CMAES_ITERATIONS,
        cmaes_popsize=CMAES_POPSIZE,
    )

    print(f"\n== Evolution complete! Best fitness: {best_fitness:.3f} ==")

    if best_individual is None or best_individual.get("robot_graph") is None:
        print("No valid robot found — nothing to save.")
        return

    # --- Save morphology (JSON) ---
    robot_json_path = DATA / "best_robot.json"
    save_graph_as_json(best_individual["robot_graph"], str(robot_json_path))
    print(f"Saved robot morphology: {robot_json_path}")

    # --- Save controller parameters (flat vector) ---
    controller_path = DATA / "best_controller_params.npy"
    np.save(controller_path, best_individual["controller_params"])
    print(f"Saved controller params: {controller_path}")

    # --- Save genotype (lightweight, no MuJoCo objects) ---
    genotype_path = DATA / "best_genotype.npz"
    g0, g1, g2 = best_individual["genotype"]
    np.savez(genotype_path, type_p=g0, conn_p=g1, rot_p=g2)
    print(f"Saved genotype: {genotype_path}")

    # --- Save fitness for reference ---
    fitness_path = DATA / "best_fitness.npy"
    np.save(fitness_path, best_fitness)
    print(f"Saved fitness: {fitness_path}")

    print("\n== SUBMISSION FILES READY ==")
    print(f"1) {robot_json_path}")
    print(f"2) {controller_path}")
    print(f"3) {genotype_path}")
    print(f"4) {fitness_path}")

if __name__ == "__main__":
    main()