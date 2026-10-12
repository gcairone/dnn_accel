"""NumPy-contract enforcement at the binding layer: wrong dtype/layout
raises, never silently cast or copied. Every check here is about the boundary itself, not
training correctness -- see test_train_correctness.py for that.
"""

import numpy as np
import pytest

import fastmlp as fm


def make_leaf(max_batch=8, in_features=4, out_features=2, seed=1):
    return fm.MLP([in_features, 6, out_features], activation="relu", max_batch=max_batch, seed=seed)


def test_call_rejects_wrong_dtype():
    model = make_leaf()
    x = np.random.rand(4, 4).astype(np.float64)
    with pytest.raises(ValueError):
        model(x)


def test_call_rejects_non_contiguous():
    model = make_leaf(in_features=8)
    x = np.random.rand(4, 16).astype(np.float32)[:, ::2]  # (4, 8), not C-contiguous
    with pytest.raises(ValueError):
        model(x)


def test_call_rejects_batch_over_max():
    model = make_leaf(max_batch=8)
    x = np.random.rand(9, 4).astype(np.float32)
    with pytest.raises(ValueError):
        model(x)


def test_call_rejects_wrong_feature_count():
    model = make_leaf(in_features=4)
    x = np.random.rand(4, 5).astype(np.float32)
    with pytest.raises(ValueError):
        model(x)


def test_call_rejects_mismatched_batch_across_leaves():
    a = fm.MLP([4, 6, 4], activation="relu", max_batch=8, seed=1)
    b = fm.MLP([4, 6, 4], activation="relu", max_batch=8, seed=2)
    model = fm.MLP([8, 6, 1], activation="relu", inputs=[a, b], max_batch=8, seed=3)
    with pytest.raises(ValueError):
        model(np.random.rand(4, 4).astype(np.float32), np.random.rand(5, 4).astype(np.float32))


def test_call_wrong_leaf_count_raises():
    model = make_leaf()
    with pytest.raises(ValueError):
        model(np.random.rand(4, 4).astype(np.float32), np.random.rand(4, 4).astype(np.float32))


def test_backward_rejects_wrong_dtype():
    model = make_leaf()
    model(np.random.rand(4, 4).astype(np.float32))
    with pytest.raises(TypeError):
        model.backward(np.zeros((4, 2), dtype=np.float64))


def test_backward_rejects_non_contiguous():
    model = make_leaf(out_features=8)
    model(np.random.rand(4, 4).astype(np.float32))
    grad = np.zeros((4, 16), dtype=np.float32)[:, ::2]
    with pytest.raises(TypeError):
        model.backward(grad)


def test_backward_rejects_wrong_feature_count():
    model = make_leaf(out_features=2)
    model(np.random.rand(4, 4).astype(np.float32))
    with pytest.raises(ValueError):
        model.backward(np.zeros((4, 3), dtype=np.float32))


def test_output_is_read_only():
    # Preallocated-buffer aliasing means the returned array must
    # never be mutated in place -- it's a live view, not a fresh copy.
    model = make_leaf()
    out = model(np.random.rand(4, 4).astype(np.float32))
    assert out.flags["WRITEABLE"] is False


def test_output_aliases_internal_buffer_across_calls():
    # Explicit, documented (not hidden) consequence of zero-allocation
    # buffer reuse: a previous forward()'s return value mutates on the next
    # forward() call on the SAME module. A caller who needs to keep an old
    # prediction around must .copy() it -- this test makes that visible
    # rather than leaving it as a surprise.
    model = make_leaf()
    x1 = np.random.rand(4, 4).astype(np.float32)
    x2 = np.random.rand(4, 4).astype(np.float32)
    out1 = model(x1)
    snapshot = out1.copy()
    out2 = model(x2)
    assert np.array_equal(out1, out2)  # out1 now aliases the SAME buffer as out2
    assert not np.array_equal(snapshot, out2)  # proves it actually changed underneath


def test_accepts_read_only_input_array():
    # InArray uses a const Scalar, so a read-only NumPy view (e.g. from
    # another module's read-only output) must still be accepted.
    model = make_leaf()
    x = np.random.rand(4, 4).astype(np.float32)
    x.setflags(write=False)
    out = model(x)
    assert out.shape == (4, 2)
