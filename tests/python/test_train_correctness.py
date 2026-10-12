"""Numeric correctness through the actual Python API, using a small
independent pure-NumPy MLP as a third oracle (tests/cpp already exhaustively
finite-difference-checks the C++ core directly; this file's job is to catch
binding-layer marshalling bugs -- wrong axis, transposed weights, wrong
activation string mapping -- that only show up when going through Python).

Model.backward() returns nothing and raw weights aren't exposed in the
Python API, so weights are read back through the .mlpf format (a JSON
header line + raw float32 blob) --
the header line is plain JSON, parseable with the stdlib, no need to
reimplement the C++ side's hand-rolled restricted parser.
"""

import json
import math
import os
import tempfile

import numpy as np

import fastmlp as fm


def _read_mlpf(path):
    with open(path, "rb") as f:
        line = f.readline()
        assert line.startswith(b"MLPF1")
        header = json.loads(line[len(b"MLPF1"):])
        blob = f.read()
    return header, blob


def _extract_layers(header, blob):
    assert header["inputs"] == [], "helper assumes a single-file, childless (leaf) .mlpf"
    sizes = header["layers"]
    offset = 0
    layers = []
    for i in range(len(sizes) - 1):
        in_f, out_f = sizes[i], sizes[i + 1]
        w_count = out_f * in_f
        W = np.frombuffer(blob, dtype=np.float32, count=w_count, offset=offset).reshape(out_f, in_f)
        offset += w_count * 4
        b = np.frombuffer(blob, dtype=np.float32, count=out_f, offset=offset)
        offset += out_f * 4
        layers.append((W.copy(), b.copy()))
    return layers


_erf = np.vectorize(math.erf)
_GELU_K0 = math.sqrt(2.0 / math.pi)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _act(name, z):
    if name == "relu":
        return np.maximum(z, 0.0)
    if name == "identity":
        return z
    if name == "sigmoid":
        return _sigmoid(z)
    if name == "tanh":
        return np.tanh(z)
    if name == "leaky_relu":
        return np.where(z > 0, z, 0.01 * z)
    if name == "elu":
        return np.where(z > 0, z, np.expm1(np.minimum(z, 0.0)))
    if name == "silu":
        return z * _sigmoid(z)
    if name == "gelu":
        return 0.5 * z * (1.0 + _erf(z / math.sqrt(2.0)))
    if name == "gelu_tanh":
        return 0.5 * z * (1.0 + np.tanh(_GELU_K0 * (z + 0.044715 * z**3)))
    if name == "softplus":
        return np.logaddexp(0.0, z)
    raise ValueError(name)


def _act_grad(name, z, a):
    if name == "relu":
        return (z > 0).astype(np.float64)
    if name == "identity":
        return np.ones_like(z)
    if name == "sigmoid":
        return a * (1.0 - a)
    if name == "tanh":
        return 1.0 - a * a
    if name == "leaky_relu":
        return np.where(z > 0, 1.0, 0.01)
    if name == "elu":
        return np.where(z > 0, 1.0, np.exp(np.minimum(z, 0.0)))
    if name == "silu":
        s = _sigmoid(z)
        return s * (1.0 + z * (1.0 - s))
    if name == "gelu":
        return 0.5 * (1.0 + _erf(z / math.sqrt(2.0))) + z * np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    if name == "gelu_tanh":
        t = np.tanh(_GELU_K0 * (z + 0.044715 * z**3))
        return 0.5 * (1.0 + t) + 0.5 * z * (1.0 - t * t) * _GELU_K0 * (1.0 + 3 * 0.044715 * z * z)
    if name == "softplus":
        return _sigmoid(z)
    raise ValueError(name)


ALL_ACTIVATIONS = ["identity", "relu", "sigmoid", "tanh", "leaky_relu", "elu", "silu", "gelu",
                   "gelu_tanh", "softplus"]


def _per_layer(activation, n_layers):
    """`activation` as the .mlpf header stores it: one name, or one per layer."""
    return list(activation) if isinstance(activation, list) else [activation] * n_layers


def ref_forward(layers, activation, x):
    acts = _per_layer(activation, len(layers))
    a = x.astype(np.float64)
    for (W, b), act in zip(layers, acts):
        z = a @ W.T.astype(np.float64) + b.astype(np.float64)
        a = _act(act, z)
    return a


def ref_forward_backward(layers, activation, x, dOut):
    """Standard textbook backprop, float64 throughout for a clean oracle."""
    acts = _per_layer(activation, len(layers))
    zs, activations = [], [x.astype(np.float64)]
    a = x.astype(np.float64)
    for (W, b), act in zip(layers, acts):
        z = a @ W.T.astype(np.float64) + b.astype(np.float64)
        zs.append(z)
        a = _act(act, z)
        activations.append(a)

    grads = [None] * len(layers)
    dA = dOut.astype(np.float64)
    for i in reversed(range(len(layers))):
        W, _ = layers[i]
        dZ = dA * _act_grad(acts[i], zs[i], activations[i + 1])
        a_in = activations[i]
        dW = dZ.T @ a_in
        db = dZ.sum(axis=0)
        grads[i] = (dW, db)
        dA = dZ @ W.astype(np.float64)
    return grads, activations[-1]


def test_reference_backward_matches_finite_differences():
    # Sanity-check the oracle itself before trusting it against fastmlp.
    rng = np.random.default_rng(0)
    layers = [
        (rng.standard_normal((6, 5)).astype(np.float32) * 0.3, rng.standard_normal(6).astype(np.float32) * 0.1),
        (rng.standard_normal((3, 6)).astype(np.float32) * 0.3, rng.standard_normal(3).astype(np.float32) * 0.1),
    ]
    x = rng.random((4, 5)).astype(np.float32)
    dOut = rng.standard_normal((4, 3))

    grads, _ = ref_forward_backward(layers, "tanh", x, dOut)

    eps = 1e-5
    W, b = layers[0]
    i, j = 2, 1
    W_plus = [(W.copy(), b.copy()), layers[1]]
    W_plus[0][0][i, j] += eps
    W_minus = [(W.copy(), b.copy()), layers[1]]
    W_minus[0][0][i, j] -= eps

    loss_plus = float(np.sum(ref_forward(W_plus, "tanh", x) * dOut))
    loss_minus = float(np.sum(ref_forward(W_minus, "tanh", x) * dOut))
    numeric = (loss_plus - loss_minus) / (2 * eps)
    analytic = grads[0][0][i, j]
    assert abs(numeric - analytic) < 1e-3


def test_forward_matches_independent_numpy_reference(tmp_path):
    model = fm.MLP([10, 16, 4], activation="relu", max_batch=8, seed=123)
    path = str(tmp_path / "m.mlpf")
    model.save(path)
    header, blob = _read_mlpf(path)
    layers = _extract_layers(header, blob)

    x = np.random.rand(5, 10).astype(np.float32)
    ref = ref_forward(layers, header["activation"], x)
    got = model(x)
    assert np.allclose(got, ref, atol=1e-4)


def test_forward_matches_reference_for_every_activation():
    for activation in ALL_ACTIVATIONS:
        model = fm.MLP([8, 12, 3], activation=activation, max_batch=8, seed=1)
        x = np.random.rand(4, 8).astype(np.float32)
        got = model(x)

        # Round-trip through .mlpf just to read back the weights -- reuses
        # the same path as the other tests instead of a separate mechanism.
        fd, path = tempfile.mkstemp(suffix=".mlpf")
        os.close(fd)
        try:
            model.save(path)
            header, blob = _read_mlpf(path)
            layers = _extract_layers(header, blob)
            ref = ref_forward(layers, activation, x)
            assert np.allclose(got, ref, atol=1e-4), activation
        finally:
            os.remove(path)


def test_sgd_step_matches_independent_numpy_reference(tmp_path):
    model = fm.MLP([10, 16, 4], activation="relu", max_batch=8, seed=321)
    path_before = str(tmp_path / "before.mlpf")
    model.save(path_before)
    layers = _extract_layers(*_read_mlpf(path_before))

    rng = np.random.default_rng(9)
    x = rng.random((5, 10)).astype(np.float32)
    dOut = rng.random((5, 4)).astype(np.float32)

    grads, _ = ref_forward_backward(layers, "relu", x, dOut)
    lr = 0.01
    expected = [(W - lr * dW, b - lr * db) for (W, b), (dW, db) in zip(layers, grads)]

    opt = fm.SGD(model, lr=lr)
    model(x)  # populate forward caches backward() needs
    model.backward(dOut)
    opt.step()
    opt.zero_grad()

    path_after = str(tmp_path / "after.mlpf")
    model.save(path_after)
    got = _extract_layers(*_read_mlpf(path_after))

    for (W_got, b_got), (W_exp, b_exp) in zip(got, expected):
        assert np.allclose(W_got, W_exp, atol=1e-3)
        assert np.allclose(b_got, b_exp, atol=1e-3)


def test_frozen_module_weights_unchanged_after_step(tmp_path):
    model = fm.MLP([6, 10, 2], activation="relu", max_batch=8, seed=1)
    model.freeze()

    path_before = str(tmp_path / "before.mlpf")
    model.save(path_before)
    before = _extract_layers(*_read_mlpf(path_before))

    opt = fm.SGD(model, lr=0.5)
    x = np.random.rand(4, 6).astype(np.float32)
    pred = model(x)
    model.backward(np.random.rand(4, 2).astype(np.float32))
    opt.step()
    opt.zero_grad()

    path_after = str(tmp_path / "after.mlpf")
    model.save(path_after)
    after = _extract_layers(*_read_mlpf(path_after))

    for (W_b, b_b), (W_a, b_a) in zip(before, after):
        assert np.array_equal(W_b, W_a)
        assert np.array_equal(b_b, b_a)


def _sgd_step_matches_reference(tmp_path, model, header_activation, sizes):
    path_before = str(tmp_path / "before.mlpf")
    model.save(path_before)
    header, blob = _read_mlpf(path_before)
    assert header["activation"] == header_activation
    layers = _extract_layers(header, blob)

    rng = np.random.default_rng(17)
    x = rng.standard_normal((5, sizes[0])).astype(np.float32)
    dOut = rng.standard_normal((5, sizes[-1])).astype(np.float32)

    grads, out_ref = ref_forward_backward(layers, header["activation"], x, dOut)
    lr = 0.01
    expected = [(W - lr * dW, b - lr * db) for (W, b), (dW, db) in zip(layers, grads)]

    opt = fm.SGD(model, lr=lr)
    assert np.allclose(model(x), out_ref, atol=1e-4)
    model.backward(dOut)
    opt.step()

    path_after = str(tmp_path / "after.mlpf")
    model.save(path_after)
    got = _extract_layers(*_read_mlpf(path_after))
    for (W_got, b_got), (W_exp, b_exp) in zip(got, expected):
        assert np.allclose(W_got, W_exp, atol=1e-4)
        assert np.allclose(b_got, b_exp, atol=1e-4)


def test_per_layer_activation_list_matches_reference(tmp_path):
    acts = ["gelu", "silu", "elu", "identity"]
    sizes = [7, 12, 10, 6, 3]
    model = fm.MLP(sizes, activation=acts, max_batch=8, seed=5)
    assert model.activations == acts
    _sgd_step_matches_reference(tmp_path, model, acts, sizes)


def test_output_activation_matches_reference(tmp_path):
    sizes = [6, 16, 8, 2]
    model = fm.MLP(sizes, activation="relu", output_activation="identity", max_batch=8, seed=6)
    assert model.activations == ["relu", "relu", "identity"]
    _sgd_step_matches_reference(tmp_path, model, ["relu", "relu", "identity"], sizes)
