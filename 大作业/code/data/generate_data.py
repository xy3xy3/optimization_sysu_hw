import numpy as np

def generate_data():
    np.random.seed(42)

    # 生成稀疏向量 x_true
    x_true = np.zeros(200)
    nonzero_indices = np.random.choice(200, 5, replace=False)
    x_true[nonzero_indices] = np.random.normal(0, 1, 5)

    # 生成测量矩阵 A_i 和噪声 e_i
    A = []
    b = []
    for _ in range(10):
        A_i = np.random.normal(0, 1, (5, 200))
        e_i = np.random.normal(0, 0.1, 5)
        b_i = A_i @ x_true + e_i
        A.append(A_i)
        b.append(b_i)

    # 拼接 A 和 b
    A = np.vstack(A)
    b = np.hstack(b)

    return A, b, x_true