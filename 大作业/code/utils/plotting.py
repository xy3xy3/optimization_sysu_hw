import matplotlib.pyplot as plt

def plot_convergence(distances_true, distances_opt, algorithm_name, save_path):
    plt.figure()
    plt.plot(distances_true, label='Distance to True x')
    plt.plot(distances_opt, label='Distance to Optimal x')
    plt.xlabel('Iterations')
    plt.ylabel('Distance')
    plt.title(f'Convergence of {algorithm_name}')
    plt.legend()
    plt.savefig(save_path)
    plt.close()