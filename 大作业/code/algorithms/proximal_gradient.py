import numpy as np
from utils.soft_threshold import soft_threshold

def proximal_gradient(A, b, lambd, x0, max_iter=1000, tol=1e-6):
    L = np.linalg.norm(A, 2)**2
    alpha = 1 / L
    x = x0.copy()
    x_history = [x.copy()]

    for _ in range(max_iter):
        grad = A.T @ (A @ x - b)
        x = soft_threshold(x - alpha * grad, lambd * alpha)
        x_history.append(x.copy())
        if np.linalg.norm(x - x_history[-2]) < tol:
            break
    return x_history