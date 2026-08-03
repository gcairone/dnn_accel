"""Generates a synthetic regression dataset for the "deep" architecture demo
(6 hidden layers, vs. the 2-3 layers of the other examples) once, writing
raw float32 binary files so all three implementations train on
byte-identical data.

target = tanh(sin(2*pi*x0)*cos(3*x1) + x2^2 - x3*x4 + 0.3*sum(x5:10)), a
nonlinear function mixing several features multiplicatively and
periodically -- enough interaction terms that a single Linear+activation
can't approximate it well, giving the deeper stack something to actually
use (not a rigorous depth-necessity proof, just a task that isn't trivially
linear).

Run once (writes into examples/deep_mlp/data/, gitignored):
    python3 examples/deep_mlp/generate_data.py
"""
import os

import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "data")

N_FEATURES = 20


def make_dataset(n_samples, seed):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n_samples, N_FEATURES)).astype(np.float32)
    raw = (np.sin(2 * np.pi * X[:, 0]) * np.cos(3 * X[:, 1]) + X[:, 2] ** 2 - X[:, 3] * X[:, 4]
           + 0.3 * X[:, 5:10].sum(axis=1))
    y = np.tanh(raw) + rng.normal(scale=0.02, size=n_samples)
    y = np.clip(y, -1.0, 1.0).astype(np.float32).reshape(-1, 1)
    return X, y


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    X_train, y_train = make_dataset(8000, seed=0)
    X_test, y_test = make_dataset(2000, seed=1)

    for name, arr in [
        ("X_train.bin", X_train), ("y_train.bin", y_train),
        ("X_test.bin", X_test), ("y_test.bin", y_test),
    ]:
        np.ascontiguousarray(arr, dtype=np.float32).tofile(os.path.join(OUT_DIR, name))

    print(f"wrote {len(X_train)} train / {len(X_test)} test examples to {OUT_DIR}")
    print(f"shapes: X (N,{N_FEATURES}) float32, y (N,1) float32 in [-1,1], raw row-major")


if __name__ == "__main__":
    main()
