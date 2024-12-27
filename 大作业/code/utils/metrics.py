import numpy as np

def compute_distances(x_history, x_true, x_opt):
    distances_true = [np.linalg.norm(x - x_true) for x in x_history]
    distances_opt = [np.linalg.norm(x - x_opt) for x in x_history]
    return distances_true, distances_opt