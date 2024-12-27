import numpy as np
from utils.soft_threshold import soft_threshold
def admm(A, b, lambd, x0, rho=1.0, max_iter=1000, tol=1e-6):
    m, n = A.shape
    x = x0.copy()
    z = x.copy()
    u = np.zeros(n)
    x_history = [x.copy()]

    A_T_A = A.T @ A  # 预计算A^T A
    A_T_b = A.T @ b  # 预计算A^T b
    eye = np.eye(n)  # 预计算单位矩阵

    for _ in range(max_iter):
        # 更新 x
        x = np.linalg.solve(A_T_A + rho * eye, A_T_b + rho * (z - u))
        # 更新 z
        z = soft_threshold(x + u, lambd / rho)
        # 更新 u
        u += x - z
        x_history.append(z.copy())
        if np.linalg.norm(x - z) < tol:
            break
    return x_history