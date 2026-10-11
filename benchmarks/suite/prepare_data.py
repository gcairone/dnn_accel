"""Writes everything the full-training mode needs into benchmarks/suite/data/
(gitignored), as raw float32/int32 files every implementation reads as-is:

  datasets/<name>/   X_{train,test}_<leaf>.bin, y_{train,test}.bin
  tasks/<task>/      model.mlpf  initial weights (same file for all four)
                     perm.bin    int32 [epochs, n_train] shuffle order per epoch
                     meta.json   for the Python workers
                     spec.txt    the same, as `key value` lines, for suite_cpp

Identical data, identical initial weights and identical batch order are what
make the four learning curves comparable step for step, not just in wall time.

    python3 benchmarks/suite/prepare_data.py   # needs examples/mnist/data/
"""

import json
import os
import sys

import numpy as np

from cases import TASKS, leaves
from common import DATA_DIR, REPO
from models import build_fastmlp

DS_DIR = os.path.join(DATA_DIR, "datasets")
TASK_DIR = os.path.join(DATA_DIR, "tasks")


def f32(a):
    return np.ascontiguousarray(a, dtype=np.float32)


# --- datasets: name -> (leaf_train, leaf_test, y_train, y_test) -------------------------

def mnist():
    src = os.path.join(REPO, "examples", "mnist", "data")
    if not os.path.exists(os.path.join(src, "train_images.bin")):
        sys.exit("MNIST not found: run `python3 examples/mnist/prepare_data.py` first")
    rd = lambda n, rows, cols: np.fromfile(os.path.join(src, n), dtype=np.uint8).reshape(rows, cols)
    xtr = f32(rd("train_images.bin", 60000, 784) / 255.0)
    xte = f32(rd("test_images.bin", 10000, 784) / 255.0)
    ytr = f32(rd("train_labels.bin", 60000, 1))
    yte = f32(rd("test_labels.bin", 10000, 1))
    return [xtr], [xte], ytr, yte


def tabular(n_train=100_000, n_test=20_000, d=32, seed=3):
    """Smooth nonlinear target with feature interactions, plus noise."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_train + n_test, d))
    y = (np.sin(x[:, 0] * x[:, 1]) + 0.5 * x[:, 2] ** 2 - np.tanh(x[:, 3] + x[:, 4])
         + 0.3 * x[:, 5] * x[:, 6] + 0.2 * np.abs(x[:, 7]) * x[:, 8]
         + 0.1 * rng.standard_normal(n_train + n_test))
    y = (y - y.mean()) / y.std()
    x, y = f32(x), f32(y[:, None])
    return [x[:n_train]], [x[n_train:]], y[:n_train], y[n_train:]


def ctr(n_train=200_000, n_test=40_000, d=64, k=8, seed=4):
    """Clicks from a latent user-item affinity: each tower sees only its own
    side's features, so the model has to learn both embeddings."""
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    u, it = rng.standard_normal((n, d)), rng.standard_normal((n, d))
    A, B = rng.standard_normal((d, k)) / np.sqrt(d), rng.standard_normal((d, k)) / np.sqrt(d)
    logit = 2.0 * (np.tanh(u @ A) * np.tanh(it @ B)).sum(axis=1) - 1.0
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(np.float32)[:, None]
    u, it = f32(u), f32(it)
    return [u[:n_train], it[:n_train]], [u[n_train:], it[n_train:]], y[:n_train], y[n_train:]


def spiral(n_train=10_000, n_test=3_000, classes=3, seed=5):
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    c = rng.integers(0, classes, n)
    r = rng.random(n)
    theta = 4.0 * r + c * 2 * np.pi / classes + 0.25 * rng.standard_normal(n)
    x = f32(np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1) * 3)
    y = f32(c[:, None])
    return [x[:n_train]], [x[n_train:]], y[:n_train], y[n_train:]


DATASETS = {"mnist": mnist, "tabular": tabular, "ctr": ctr, "spiral": spiral}
# Reconstruction: MNIST's own files, the input doubling as the target.
ALIASES = {"mnist_ae": "mnist"}


def write_dataset(name):
    out = os.path.join(DS_DIR, name)
    xtr, xte, ytr, yte = DATASETS[name]()
    os.makedirs(out, exist_ok=True)
    paths = {"x_train": [], "x_test": []}
    for i, (a, b) in enumerate(zip(xtr, xte)):
        for split, arr in (("train", a), ("test", b)):
            p = os.path.join(out, f"X_{split}_{i}.bin")
            arr.tofile(p)
            paths[f"x_{split}"].append(p)
    for split, arr in (("train", ytr), ("test", yte)):
        p = os.path.join(out, f"y_{split}.bin")
        arr.tofile(p)
        paths[f"y_{split}"] = p
    paths.update(n_train=len(ytr), n_test=len(yte), y_width=ytr.shape[1],
                 leaf_widths=[a.shape[1] for a in xtr])
    return paths


def main():
    ds_cache = {}
    for task in TASKS:
        base = ALIASES.get(task.dataset, task.dataset)
        if base not in ds_cache:
            print(f"dataset {base} ...", flush=True)
            ds_cache[base] = write_dataset(base)
        ds = dict(ds_cache[base])
        if task.dataset == "mnist_ae":
            ds.update(y_train=ds["x_train"][0], y_test=ds["x_test"][0], y_width=ds["leaf_widths"][0])
        widths = [l.sizes[0] for l in leaves(task.arch.root)]
        assert widths == ds["leaf_widths"], (task.name, widths, ds["leaf_widths"])

        out = os.path.join(TASK_DIR, task.name)
        os.makedirs(out, exist_ok=True)
        model = os.path.join(out, "model.mlpf")
        build_fastmlp(task.arch.root, max_batch=1).save(model)
        rng = np.random.default_rng(42)
        perm = np.stack([rng.permutation(ds["n_train"]) for _ in range(task.epochs)]).astype(np.int32)
        perm_path = os.path.join(out, "perm.bin")
        perm.tofile(perm_path)

        meta = dict(ds, task=task.name, model=model, perm=perm_path, loss=task.arch.loss,
                    batch=task.batch, epochs=task.epochs, optimizer=task.optimizer,
                    lr=task.lr, momentum=task.momentum)
        with open(os.path.join(out, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1)
        # suite_cpp has no JSON parser: the same facts as whitespace tokens.
        with open(os.path.join(out, "spec.txt"), "w") as f:
            for k in ("model", "perm", "loss", "optimizer", "lr", "momentum", "batch", "epochs",
                      "n_train", "n_test", "y_width", "y_train", "y_test"):
                f.write(f"{k} {meta[k]}\n")
            f.write(f"leaves {len(widths)}\n")
            for w, a, b in zip(widths, meta["x_train"], meta["x_test"]):
                f.write(f"leaf {w} {a} {b}\n")
        print(f"task {task.name}: {ds['n_train']} train / {ds['n_test']} test, "
              f"{task.epochs} epochs x {ds['n_train'] // task.batch} steps", flush=True)


if __name__ == "__main__":
    main()
