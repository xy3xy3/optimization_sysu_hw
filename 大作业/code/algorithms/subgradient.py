import numpy as np

def subgradient_method(A, b, lambd, x0, max_iter=1000, tol=1e-6):
    x = x0.copy()
    x_history = [x.copy()]
    m, n = A.shape

    for k in range(1, max_iter+1):
        # 计算梯度 ∇f(x)
        grad_f = A.T @ (A @ x - b)

        # 计算次梯度 ∂g(x)
        subgrad_g = np.zeros(n)
        for i in range(n):
            if x[i] > 0:
                subgrad_g[i] = lambd
            elif x[i] < 0:
                subgrad_g[i] = -lambd
            else:
                subgrad_g[i] = 0  # 改为0以减少随机性

        # 次梯度下降
        grad = grad_f + subgrad_g

        # 梯度裁剪
        grad_norm = np.linalg.norm(grad)
        max_grad = 10.0  # 根据需要调整
        if grad_norm > max_grad:
            grad = grad / grad_norm * max_grad

        # 步长调整
        alpha = 1 / k

        x = x - alpha * grad
        x_history.append(x.copy())

        # 检查收敛
        if np.linalg.norm(x - x_history[-2]) < tol:
            break
    return x_history
