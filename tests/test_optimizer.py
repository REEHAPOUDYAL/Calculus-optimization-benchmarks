import jax.numpy as jnp
from src.core.functional import rosenbrock
from src.core.derivatives import get_calculus_engine
from src.optimizers.first_order import VanillaGradientDescent
from src.optimizers.second_order import NewtonOptimizer

def test_optimizer_direction():
    grad_fn, hess_fn = get_calculus_engine(rosenbrock)
    x_start = jnp.array([-1.2, 1.0])
    initial_loss = rosenbrock(x_start)
    
    # Test First Order
    gd = VanillaGradientDescent(learning_rate=0.001)
    x_next_gd = gd.step(x_start, grad_fn)
    assert rosenbrock(x_next_gd) < initial_loss, "GD step should decrease loss"
    
    # Test Second Order (Newton)
    newton = NewtonOptimizer()
    x_next_newton = newton.step(x_start, grad_fn, hess_fn)
    assert rosenbrock(x_next_newton) < initial_loss, "Newton step should decrease loss"
    
    print("Optimization Step Checks: PASSED")

if __name__ == "__main__":
    test_optimizer_direction()