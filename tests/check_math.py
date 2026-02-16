import jax.numpy as jnp
from src.core.functional import rosenbrock
from src.core.derivatives import get_calculus_engine

x_test = jnp.array([1.0, 1.0])
grad_fn, hess_fn = get_calculus_engine(rosenbrock)

print(f"Value at Min: {rosenbrock(x_test)}")
print(f"Gradient at Min (Should be 0): {grad_fn(x_test)}")
print(f"Hessian at Min:\n{hess_fn(x_test)}")