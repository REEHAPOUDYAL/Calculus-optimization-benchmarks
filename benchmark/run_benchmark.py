import jax.numpy as jnp
from src.core.functional import rosenbrock
from src.core.derivatives import get_calculus_engine
from src.optimizers.first_order import VanillaGradientDescent
from src.optimizers.second_order import NewtonOptimizer
from src.visualization.landscape import plot_optimization_path

def run_comparison():
    grad_fn, hess_fn = get_calculus_engine(rosenbrock)
    x_init = jnp.array([-1.2, 1.0])
    
    optimizers = {
        "Gradient Descent": VanillaGradientDescent(learning_rate=0.002),
        "Newton's Method": NewtonOptimizer(learning_rate=1.0)
    }
    
    paths = {}
    for name, opt in optimizers.items():
        print(f"\nRunning {name}")
        x = x_init
        history = [x]
        
        for i in range(50):
            x = opt.step(x, grad_fn, hess_fn)
            history.append(x)
            
            if jnp.linalg.norm(grad_fn(x)) < 1e-5:
                print(f"Converged in {i} steps!")
                break
        
        paths[name] = history
    print("\nGenerating visualization")
    plot_optimization_path(rosenbrock, paths)

if __name__ == "__main__":
    run_comparison()