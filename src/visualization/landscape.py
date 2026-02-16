import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

def plot_optimization_path(func, paths, save_path="docs/path_comparison.png"):
    x = np.linspace(-2.0, 2.0, 250)
    y = np.linspace(-1.0, 3.0, 250)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[func(jnp.array([xi, yi])) for xi in x] for yi in y])

    plt.figure(figsize=(10, 8))
    cp = plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='viridis', alpha=0.6)
    plt.clabel(cp, inline=1, fontsize=8)

    colors = ['#FF5733', '#33FF57']
    for (name, path), color in zip(paths.items(), colors):
        path = np.array(path)
        plt.plot(path[:, 0], path[:, 1], marker='o', color=color, label=name, linewidth=2, markersize=5)
        plt.annotate(f'Start {name}', (path[0, 0], path[0, 1]), textcoords="offset points", xytext=(0,10), ha='center')

    plt.title("Optimization Trajectory: First-Order vs. Second-Order", fontsize=14)
    plt.xlabel("x", fontsize=12)
    plt.ylabel("y", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.savefig(save_path, dpi=300)
    plt.show()
    