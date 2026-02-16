from abc import ABC, abstractmethod
import jax.numpy as jnp
from typing import Callable

class Optimizer(ABC):
    @abstractmethod
    def step(self, x: jnp.ndarray, grad_fn: Callable, hess_fn: Callable = None) -> jnp.ndarray:
        pass