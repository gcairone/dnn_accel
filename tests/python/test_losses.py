"""cross_entropy_grad against a finite-difference check of the loss it
claims to differentiate (mean softmax cross-entropy over the batch), plus its
input contract."""

import numpy as np
import pytest

import fastmlp as fm


def _ce_loss(logits, labels):
    z = logits.astype(np.float64)
    z = z - z.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return -logp[np.arange(len(labels)), labels].mean()


def test_cross_entropy_grad_matches_finite_differences():
    rng = np.random.default_rng(0)
    logits = (3 * rng.standard_normal((5, 4))).astype(np.float32)
    labels = rng.integers(0, 4, size=5)
    grad = fm.cross_entropy_grad(logits, labels)
    assert grad.dtype == np.float32 and grad.flags["C_CONTIGUOUS"]

    eps = 1e-3
    for i in range(5):
        for j in range(4):
            lp, lm = logits.astype(np.float64), logits.astype(np.float64)
            lp[i, j] += eps
            lm[i, j] -= eps
            fd = (_ce_loss(lp, labels) - _ce_loss(lm, labels)) / (2 * eps)
            assert abs(fd - grad[i, j]) < 1e-5


def test_cross_entropy_grad_one_hot_equals_labels_and_is_stable():
    logits = np.array([[1000.0, 0.0, -1000.0], [0.5, 0.5, 0.5]], dtype=np.float32)
    labels = np.array([2, 0])
    one_hot = np.eye(3, dtype=np.float32)[labels]
    g_labels = fm.cross_entropy_grad(logits, labels)
    g_onehot = fm.cross_entropy_grad(logits, one_hot)
    assert np.all(np.isfinite(g_labels))
    assert np.allclose(g_labels, g_onehot)
    assert np.allclose(g_labels[0], [0.5, 0.0, -0.5])  # (softmax - y) / B, B = 2


def test_cross_entropy_grad_rejects_bad_input():
    logits = np.zeros((2, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="float32"):
        fm.cross_entropy_grad(logits.astype(np.float64), np.array([0, 1]))
    with pytest.raises(ValueError, match="out of range"):
        fm.cross_entropy_grad(logits, np.array([0, 3]))
    with pytest.raises(ValueError, match="shape"):
        fm.cross_entropy_grad(logits, np.array([0, 1, 2]))
    with pytest.raises(ValueError, match="shape"):
        fm.cross_entropy_grad(logits, np.zeros((2, 4), dtype=np.float32))
