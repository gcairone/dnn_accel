"""fastmlp: a minimal, single-thread, SIMD-tuned MLP core. The native
extension (`_fastmlp`, built from bindings/python/) does all the real work;
this package re-exports its public API plus the pure-Python loss helpers.
"""

from ._fastmlp import MLP, SGD, Adam, ShapeError, StateError
from .losses import bce_grad, cross_entropy_grad, mse_grad

__all__ = [
    "MLP",
    "SGD",
    "Adam",
    "ShapeError",
    "StateError",
    "mse_grad",
    "bce_grad",
    "cross_entropy_grad",
]
