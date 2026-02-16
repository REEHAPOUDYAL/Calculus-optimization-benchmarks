import jax.numpy as jnp
from typing import Callable
from .base import Optimizer

class NewtonOptimizer(Optimizer):
    def __init__(self, learning_rate: float = 1.0, damping: float = 1e-4):
        self.lr = learning_rate
        self.damping = damping

    def step(self, x: jnp.ndarray, grad_fn: Callable, hess_fn: Callable) -> jnp.ndarray:
        g = grad_fn(x)
        H = hess_fn(x)        
        H_reg = H + self.damping * jnp.eye(H.shape[0])
        try:
            delta = jnp.linalg.solve(H_reg, g)
        except jnp.linalg.LinAlgError:
            print("Warning: Singular Hessian encountered. Falling back to gradient step.")
            delta = g
            
        return x - self.lr * delta