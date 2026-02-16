import jax.numpy as jnp
import pytest
from src.core.functional import rosenbrock
from src.core.derivatives import get_calculus_engine

def test_rosenbrock_optimum():
    x_min = jnp.array([1.0, 1.0])
    grad_fn, hess_fn = get_calculus_engine(rosenbrock)
    assert jnp.isclose(rosenbrock(x_min), 0.0), "Value at min should be 0"    
    grad = grad_fn(x_min)
    assert jnp.allclose(grad, 0.0, atol=1e-5), f"Gradient at min is too high: {grad}"
    H = hess_fn(x_min)
    eigenvalues = jnp.linalg.eigvalsh(H)
    assert jnp.all(eigenvalues > 0), "Hessian must be positive definite at the minimum"

if __name__ == "__main__":
    test_rosenbrock_optimum()
    print("Phase 1 Math Check: PASSED")