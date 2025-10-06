from pyswarm import pso
from scipy.optimize import differential_evolution
import numpy as np
import time




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