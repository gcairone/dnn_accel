"""Loss gradients. Pure NumPy, not C++: elementwise O(n) work, not a
performance bottleneck (the GEMM microkernel is). Still dtype/contiguity
checked here, since the output feeds straight back into MLP.backward(),
which enforces the same float32/C-contiguous NumPy contract as MLP.__call__.
"""

import numpy as np


def _check(arr: np.ndarray, name: str) -> np.ndarray:
    if arr.dtype != np.float32:
        raise ValueError(f"fastmlp: {name} must be float32, got {arr.dtype}")
    if not arr.flags["C_CONTIGUOUS"]:
        raise ValueError(f"fastmlp: {name} must be C-contiguous")
    return arr


def mse_grad(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Gradient of mean squared error, averaged over the batch dimension:
    L = (1/B) * sum_b ||pred_b - target_b||^2.
    """
    _check(pred, "pred")
    _check(target, "target")
    if pred.shape != target.shape:
        raise ValueError(f"fastmlp: pred.shape {pred.shape} != target.shape {target.shape}")
    grad = (2.0 / pred.shape[0]) * (pred - target)
    return np.ascontiguousarray(grad, dtype=np.float32)


def bce_grad(pred: np.ndarray, target: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Gradient of binary cross-entropy, averaged over the batch dimension.
    `pred` is clipped to [eps, 1-eps] before dividing, to avoid a div-by-zero
    at the 0/1 saturation extremes.
    """
    _check(pred, "pred")
    _check(target, "target")
    if pred.shape != target.shape:
        raise ValueError(f"fastmlp: pred.shape {pred.shape} != target.shape {target.shape}")
    p = np.clip(pred, eps, 1.0 - eps)
    grad = (p - target) / (p * (1.0 - p)) / pred.shape[0]
    return np.ascontiguousarray(grad, dtype=np.float32)
