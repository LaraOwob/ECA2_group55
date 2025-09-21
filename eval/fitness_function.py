import numpy as np
import mujoco
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko

def fitness_distance(start_xy, end_xy, target_xy=(1.0, 0.0), success_radius=0.1):
    """
    Fitness function for targeted locomotion: the goal is for the robot to move from its starting point A towards the fixed target B.
    The fitness is computed as:

        fitness = (distance from start to target) - (distance from end to target)

    and rewards the robot for reducing its distance to the target during its path. If the robot actually reaches the target (within a small radius), an
    additional bonus of 1 is added.

    Parameters:
    start_xy: the robot's initial position in the XY plane.
    end_xy: the robot's final position in the XY plane.
    target_xy: the target location (default is (1.0, 0.0) but ofc we can change it)
    success_radius : the radius around the target that counts as a successful reach (default is 0.1 that i think it's close enough)

    It returns the fitness value that is positive if the robot moved closer to the target and larger if it reached the target.
    """
    d0 = np.linalg.norm(np.array(start_xy) - np.array(target_xy))
    dT = np.linalg.norm(np.array(end_xy)   - np.array(target_xy))
    fit = d0 - dT          # reward gained by getting closer
    if dT <= success_radius:
        fit += 1.0         # bonus if the target is reached
    return fit