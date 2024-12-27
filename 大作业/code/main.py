import os
import csv
import numpy as np
import cvxpy as cp
from data.generate_data import generate_data
from algorithms.proximal_gradient import proximal_gradient
from algorithms.admm import admm
from algorithms.subgradient import subgradient_method
from utils.metrics import compute_distances
from utils.plotting import plot_convergence

# 结果文件夹
img_dir = "img"
csv_dir = "results"
os.makedirs(img_dir, exist_ok=True)
os.makedirs(csv_dir, exist_ok=True)

# 生成数据
A, b, x_true = generate_data()

# 求解最优解 x_opt
x = cp.Variable(200)
objective = cp.Minimize(0.5 * cp.sum_squares(A @ x - b) + cp.norm(x, 1))
prob = cp.Problem(objective)
prob.solve()
x_opt = x.value

# 初始点 x0
x0 = np.zeros(200)

# 定义正则化参数的范围
lambda_values = [0.1, 1.0, 10.0]

def save_to_csv(filename, distances_true, distances_opt):
    """保存迭代历史到CSV文件"""
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Iteration', 'Distance_to_True_x', 'Distance_to_Optimal_x'])
        for i, (d_true, d_opt) in enumerate(zip(distances_true, distances_opt)):
            writer.writerow([i, d_true, d_opt])

for lambd in lambda_values:
    print(f"\n正则化参数 λ = {lambd}")

    # Proximal Gradient
    print("Proximal Gradient Method:")
    x_history_pg = proximal_gradient(A, b, lambd, x0)
    distances_true_pg, distances_opt_pg = compute_distances(x_history_pg, x_true, x_opt)
    print(f"最终迭代: 距离真值={distances_true_pg[-1]:.6f}, 距离最优解={distances_opt_pg[-1]:.6f}")
    plot_convergence(distances_true_pg, distances_opt_pg, f'Proximal Gradient Method (λ={lambd})', f'{img_dir}/pg_lambda_{lambd}.png')
    save_to_csv(f'{csv_dir}/pg_lambda_{lambd}.csv', distances_true_pg, distances_opt_pg)

    # ADMM
    print("ADMM:")
    x_history_admm = admm(A, b, lambd, x0)
    distances_true_admm, distances_opt_admm = compute_distances(x_history_admm, x_true, x_opt)
    print(f"最终迭代: 距离真值={distances_true_admm[-1]:.6f}, 距离最优解={distances_opt_admm[-1]:.6f}")
    plot_convergence(distances_true_admm, distances_opt_admm, f'ADMM (λ={lambd})', f'{img_dir}/admm_lambda_{lambd}.png')
    save_to_csv(f'{csv_dir}/admm_lambda_{lambd}.csv', distances_true_admm, distances_opt_admm)

    # Subgradient Method
    print("Subgradient Method:")
    x_history_sub = subgradient_method(A, b, lambd, x0)
    distances_true_sub, distances_opt_sub = compute_distances(x_history_sub, x_true, x_opt)
    print(f"最终迭代: 距离真值={distances_true_sub[-1]:.6f}, 距离最优解={distances_opt_sub[-1]:.6f}")
    plot_convergence(distances_true_sub, distances_opt_sub, f'Subgradient Method (λ={lambd})', f'{img_dir}/sub_lambda_{lambd}.png')
    save_to_csv(f'{csv_dir}/sub_lambda_{lambd}.csv', distances_true_sub, distances_opt_sub)
