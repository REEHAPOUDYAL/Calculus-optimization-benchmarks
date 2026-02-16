import jax.numpy as jnp
from typing import Callable
from .base import Optimizer

class VanillaGradientDescent(Optimizer):
    def __init__(self, learning_rate: float = 0.001):
        self.lr = learning_rate

    def step(self, x: jnp.ndarray, grad_fn: Callable, hess_fn: Callable = None) -> jnp.ndarray:
        g = grad_fn(x)
        return x - self.lr * g