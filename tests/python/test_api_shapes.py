"""The tree scenarios (single leaf, two-branch, three-level with both call
forms, frozen sub-modules, inference-only), through the Python API (not the
C++ core directly -- tests/cpp already covers that). Each test is a
structural/API check, not a numeric-correctness check (see
test_train_correctness.py for gradient parity).
"""

import numpy as np
import pytest

import fastmlp as fm


def test_scenario1_single_mlp_binary_classification():
    model = fm.MLP([20, 64, 32, 1], activation="relu", max_batch=128, seed=1)
    opt = fm.SGD(model, lr=1e-2)

    rng = np.random.default_rng(0)
    for _ in range(20):
        xb = rng.random((16, 20), dtype=np.float32)
        yb = (rng.random((16, 1), dtype=np.float32) > 0.5).astype(np.float32)
        pred = model(xb)
        grad_out = fm.bce_grad(pred, yb)
        model.backward(grad_out)
        opt.step()
        opt.zero_grad()

    preds = model(rng.random((4, 20), dtype=np.float32)) > 0.5
    assert preds.shape == (4, 1)


def test_scenario2_two_branches_adam():
    enc_img = fm.MLP([512, 128, 64], activation="relu", max_batch=64, seed=1)
    enc_meta = fm.MLP([8, 32, 64], activation="relu", max_batch=64, seed=2)
    model = fm.MLP([64 + 64, 32, 1], activation="relu", inputs=[enc_img, enc_meta], max_batch=64, seed=3)

    opt = fm.Adam(model, lr=1e-3, beta1=0.9, beta2=0.999)

    rng = np.random.default_rng(1)
    img_feat = rng.random((8, 512), dtype=np.float32)
    meta_feat = rng.random((8, 8), dtype=np.float32)
    target = rng.random((8, 1), dtype=np.float32)

    pred = model(img_feat, meta_feat)
    assert pred.shape == (8, 1)
    grad_out = fm.mse_grad(pred, target)
    model.backward(grad_out)
    opt.step()
    opt.zero_grad()

    # opt discovers enc_img/enc_meta by walking model.inputs recursively --
    # nothing was registered manually.
    assert len(model.inputs) == 2


def test_scenario3_three_level_tree_both_call_forms():
    enc_a = fm.MLP([16, 32, 16], activation="relu", max_batch=256, name="a", seed=1)
    enc_b = fm.MLP([16, 32, 16], activation="relu", max_batch=256, name="b", seed=2)
    enc_c = fm.MLP([24, 32, 16], activation="relu", max_batch=256, name="c", seed=3)
    mid_ab = fm.MLP([16 + 16, 32, 16], activation="relu", inputs=[enc_a, enc_b], max_batch=256, seed=4)
    model = fm.MLP([16 + 16, 16, 1], activation="relu", inputs=[mid_ab, enc_c], max_batch=256, seed=5)

    opt = fm.SGD(model, lr=5e-3)

    rng = np.random.default_rng(2)
    a = rng.random((8, 16), dtype=np.float32)
    b = rng.random((8, 16), dtype=np.float32)
    c = rng.random((8, 24), dtype=np.float32)
    target = rng.random((8, 1), dtype=np.float32)

    # positional: flattened, depth-first order over leaves
    pred_positional = model(a, b, c)
    # dict: keyed by leaf name, unambiguous regardless of tree depth
    pred_dict = model({"a": a, "b": b, "c": c})
    assert np.array_equal(pred_positional, pred_dict)

    grad_out = fm.mse_grad(pred_positional, target)
    model.backward(grad_out)
    opt.step()
    opt.zero_grad()


def test_scenario4_freeze_pretrained_submodules():
    enc_x = fm.MLP([64, 128, 64], activation="relu", max_batch=256, seed=1)
    enc_y = fm.MLP([32, 128, 64], activation="relu", max_batch=256, seed=2)
    model = fm.MLP([128, 64, 1], activation="relu", inputs=[enc_x, enc_y], max_batch=256, seed=3)

    enc_x.freeze()
    enc_y.freeze()

    opt = fm.SGD(model, lr=1e-2)

    rng = np.random.default_rng(3)
    xb = rng.random((8, 64), dtype=np.float32)
    yb = rng.random((8, 32), dtype=np.float32)
    target = rng.random((8, 1), dtype=np.float32)

    pred = model(xb, yb)
    model.backward(fm.mse_grad(pred, target))
    opt.step()
    opt.zero_grad()  # if this reaches here without error, frozen modules were skipped cleanly


def test_scenario5_inference_only_deployment():
    model = fm.MLP(
        [64 + 64, 64, 1],
        activation="relu",
        inputs=[
            fm.MLP([64, 128, 64], activation="relu", max_batch=1, seed=1),
            fm.MLP([32, 128, 64], activation="relu", max_batch=1, seed=2),
        ],
        max_batch=1,
        inference_only=True,
        seed=3,
    )
    assert model.inference_only

    rng = np.random.default_rng(4)
    x = rng.random((1, 64), dtype=np.float32)
    y = rng.random((1, 32), dtype=np.float32)
    prediction = model(x, y)[0, 0]
    assert isinstance(float(prediction), float)


def test_training_loss_actually_decreases():
    # A real end-to-end training loop, not just "doesn't crash": loss on a
    # fixed batch must go down over enough SGD steps on a simple regression
    # target. Confirms the wired-together forward/backward/optimizer chain
    # behaves like training, not just like a compiling API.
    model = fm.MLP([4, 16, 1], activation="relu", max_batch=32, seed=7)
    opt = fm.SGD(model, lr=1e-1)

    rng = np.random.default_rng(5)
    x = rng.random((32, 4), dtype=np.float32)
    true_w = rng.random((4, 1), dtype=np.float32)
    target = (x @ true_w).astype(np.float32)

    def mse(pred):
        return float(np.mean((pred - target) ** 2))

    losses = []
    for _ in range(200):
        pred = model(x)
        losses.append(mse(pred))
        model.backward(fm.mse_grad(pred, target))
        opt.step()
        opt.zero_grad()

    assert losses[-1] < losses[0] * 0.5


def test_activation_forms_and_errors():
    assert fm.MLP([4, 8, 2], max_batch=4).activations == ["relu", "relu"]
    assert fm.MLP([4, 8, 2], activation="tanh", max_batch=4).activations == ["tanh", "tanh"]
    assert fm.MLP([4, 8, 2], activation="tanh", output_activation="sigmoid",
                  max_batch=4).activations == ["tanh", "sigmoid"]
    assert fm.MLP([4, 8, 2], activation=["gelu_tanh", "softplus"],
                  max_batch=4).activations == ["gelu_tanh", "softplus"]
    # a one-layer module: output_activation is its only layer
    assert fm.MLP([4, 2], activation="relu", output_activation="identity",
                  max_batch=4).activations == ["identity"]

    with pytest.raises(ValueError, match="one per layer"):
        fm.MLP([4, 8, 2], activation=["relu"], max_batch=4)
    with pytest.raises(ValueError, match="not both"):
        fm.MLP([4, 8, 2], activation=["relu", "relu"], output_activation="identity", max_batch=4)
    with pytest.raises(ValueError, match="unknown activation 'swish'"):
        fm.MLP([4, 8, 2], activation="swish", max_batch=4)
    with pytest.raises(ValueError, match="unknown activation"):
        fm.MLP([4, 8, 2], activation="relu", output_activation="softmax", max_batch=4)


def test_tree_with_mixed_activations_trains():
    rng = np.random.default_rng(3)
    enc_a = fm.MLP([6, 16, 8], activation="gelu", max_batch=32, seed=1)
    enc_b = fm.MLP([4, 16, 8], activation=["silu", "tanh"], max_batch=32, seed=2)
    model = fm.MLP([16, 16, 3], activation="relu", output_activation="identity",
                   inputs=[enc_a, enc_b], max_batch=32, seed=3)
    opt = fm.Adam(model, lr=1e-2)

    a = rng.standard_normal((32, 6)).astype(np.float32)
    b = rng.standard_normal((32, 4)).astype(np.float32)
    labels = (a[:, 0] > 0).astype(np.int64) + (b[:, 0] > 0).astype(np.int64)

    def loss():
        z = model(a, b).astype(np.float64)
        z -= z.max(axis=1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        return -logp[np.arange(32), labels].mean()

    first = loss()
    for _ in range(100):
        logits = model(a, b)
        model.backward(fm.cross_entropy_grad(logits, labels))
        opt.step()
        opt.zero_grad()
    assert loss() < 0.5 * first
