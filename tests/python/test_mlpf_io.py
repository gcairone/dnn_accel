""".mlpf save/load through the actual Python API (usages.md ex6). Core-level
format edge cases (malformed headers, truncated blobs) are already covered
by tests/cpp/test_mlpf_roundtrip.cpp -- this file only checks what's
specific to the binding layer: MLP.save/.load/.load_weights and the
inline_inputs=False multi-file path.
"""

import numpy as np
import pytest

import fastmlp as fm


def test_single_file_roundtrip(tmp_path):
    model = fm.MLP([20, 64, 32, 1], activation="relu", max_batch=128, seed=1)
    x = np.random.rand(4, 20).astype(np.float32)
    pred_before = model(x).copy()

    path = str(tmp_path / "model.mlpf")
    model.save(path)
    loaded = fm.MLP.load(path)

    assert np.array_equal(pred_before, loaded(x))


def test_load_weights_into_matching_shape(tmp_path):
    src = fm.MLP([8, 16, 4], activation="relu", max_batch=32, seed=1)
    dst = fm.MLP([8, 16, 4], activation="relu", max_batch=32, seed=2)  # different init

    x = np.random.rand(4, 8).astype(np.float32)
    assert not np.array_equal(src(x), dst(x))  # different seeds -> different weights

    path = str(tmp_path / "src.mlpf")
    src.save(path)
    dst.load_weights(path)

    assert np.array_equal(src(x), dst(x))


def test_load_weights_shape_mismatch_raises(tmp_path):
    src = fm.MLP([8, 16, 4], activation="relu", max_batch=32, seed=1)
    path = str(tmp_path / "src.mlpf")
    src.save(path)

    wrong_shape = fm.MLP([8, 16, 5], activation="relu", max_batch=32, seed=2)
    with pytest.raises(ValueError):
        wrong_shape.load_weights(path)


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(ValueError):
        fm.MLP.load(str(tmp_path / "does_not_exist.mlpf"))


def test_multi_file_save_load_roundtrip(tmp_path):
    enc_x = fm.MLP([64, 128, 64], activation="relu", max_batch=256, seed=1)
    enc_y = fm.MLP([32, 128, 64], activation="relu", max_batch=256, seed=2)
    model = fm.MLP([128, 64, 1], activation="relu", inputs=[enc_x, enc_y], max_batch=256, seed=3)

    x = np.random.rand(4, 64).astype(np.float32)
    y = np.random.rand(4, 32).astype(np.float32)
    pred_before = model(x, y).copy()

    enc_x.save(str(tmp_path / "enc_x.mlpf"))
    enc_y.save(str(tmp_path / "enc_y.mlpf"))
    model.save(str(tmp_path / "model_full.mlpf"), inline_inputs=False)

    loaded = fm.MLP.load(str(tmp_path / "model_full.mlpf"))
    assert np.array_equal(pred_before, loaded(x, y))


def test_save_inline_inputs_false_without_child_save_raises(tmp_path):
    # A child that was never itself .save()'d has no path for the parent to
    # reference -- this must raise, not silently write a broken header.
    enc_x = fm.MLP([8, 16, 4], activation="relu", max_batch=32, seed=1)
    enc_y = fm.MLP([8, 16, 4], activation="relu", max_batch=32, seed=2)
    model = fm.MLP([8, 16, 1], activation="relu", inputs=[enc_x, enc_y], max_batch=32, seed=3)

    with pytest.raises(RuntimeError):
        model.save(str(tmp_path / "model.mlpf"), inline_inputs=False)


def test_loaded_model_has_no_reconstructed_children(tmp_path):
    # Documented limitation (bindings/python/mlp_wrapper.hpp): a tree loaded
    # via MLP.load() has no per-child Python wrappers, so .inputs is empty
    # even though the underlying tree has structure. Forward/backward still
    # work correctly across the whole loaded subtree (covered by the
    # roundtrip tests above) -- only the .inputs introspection is limited.
    enc_x = fm.MLP([8, 16, 4], activation="relu", max_batch=32, seed=1)
    enc_y = fm.MLP([8, 16, 4], activation="relu", max_batch=32, seed=2)
    model = fm.MLP([8, 16, 1], activation="relu", inputs=[enc_x, enc_y], max_batch=32, seed=3)
    assert len(model.inputs) == 2

    path = str(tmp_path / "model.mlpf")
    model.save(path)
    loaded = fm.MLP.load(path)
    assert loaded.inputs == []
    assert loaded(np.random.rand(4, 8).astype(np.float32), np.random.rand(4, 8).astype(np.float32)).shape == (4, 1)


def test_mixed_activations_survive_roundtrip(tmp_path):
    enc = fm.MLP([5, 8, 4], activation=["gelu", "elu"], max_batch=8, seed=1)
    model = fm.MLP([4, 6, 2], activation="leaky_relu", output_activation="identity",
                   inputs=[enc], max_batch=8, seed=2)
    x = np.random.rand(3, 5).astype(np.float32)
    pred = model(x).copy()

    path = str(tmp_path / "mixed.mlpf")
    model.save(path)
    loaded = fm.MLP.load(path)
    assert loaded.activations == ["leaky_relu", "identity"]
    assert np.array_equal(pred, loaded(x))
