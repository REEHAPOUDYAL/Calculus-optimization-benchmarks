import jax.numpy as jnp
from typing import Union

def rosenbrock(x: jnp.ndarray, a: float = 1.0, b: float = 100.0) -> float:
    x0, x1 = x[0], x[1]
    term1 = (a - x0) ** 2
    term2 = b * (x1 - x0**2) ** 2
    return term1 + term2

def rastrigin(x: jnp.ndarray, A: float = 10.0) -> float:
    n = x.shape[0]
    return A * n + jnp.sum(x**2 - A * jnp.cos(2 * jnp.pi * x))