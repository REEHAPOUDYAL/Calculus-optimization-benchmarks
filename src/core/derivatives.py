import jax
from typing import Callable, Tuple

def get_calculus_engine(func: Callable) -> Tuple[Callable, Callable]:    
    grad_fn = jax.jit(jax.grad(func))
    hess_fn = jax.jit(jax.hessian(func))
    return grad_fn, hess_fn