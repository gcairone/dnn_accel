"""Generates the two-moons dataset once and writes it as raw float32 binary
files (row-major, native byte order, no header). All three example
implementations (train_cpp.cpp, train_fastmlp.py, train_torch.py) then read
these exact same files, so their behavior/timing comparison is on
byte-identical data rather than three independently-seeded generators.

Two-moons is a classic non-linearly-separable binary classification
benchmark -- a real (if small) task, not the "loss just needs to decrease"
style of tests/python/test_train_correctness.py.

Run once (writes into examples/two_moons/data/, not checked in -- it's
regenerated on demand, same as any other build artifact):
    python3 examples/two_moons/generate_data.py
"""
import os

import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "data")


def make_two_moons(n_samples, noise, seed):
    rng = np.random.default_rng(seed)
    n0 = n_samples // 2
    n1 = n_samples - n0
    theta0 = rng.uniform(0, np.pi, n0)
    x0 = np.stack([np.cos(theta0), np.sin(theta0)], axis=1)
    theta1 = rng.uniform(0, np.pi, n1)
    x1 = np.stack([1 - np.cos(theta1), 1 - np.sin(theta1) - 0.5], axis=1)
    X = np.concatenate([x0, x1], axis=0)
    y = np.concatenate([np.zeros(n0), np.ones(n1)])
    X = X + rng.normal(scale=noise, size=X.shape)
    perm = rng.permutation(n_samples)
    return X[perm].astype(np.float32), y[perm].astype(np.float32).reshape(-1, 1)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    X, y = make_two_moons(2000, noise=0.15, seed=0)
    n_train = 1600
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]

    np.ascontiguousarray(X_train, dtype=np.float32).tofile(os.path.join(OUT_DIR, "X_train.bin"))
    np.ascontiguousarray(y_train, dtype=np.float32).tofile(os.path.join(OUT_DIR, "y_train.bin"))
    np.ascontiguousarray(X_test, dtype=np.float32).tofile(os.path.join(OUT_DIR, "X_test.bin"))
    np.ascontiguousarray(y_test, dtype=np.float32).tofile(os.path.join(OUT_DIR, "y_test.bin"))
    print(f"wrote {n_train} train / {len(X_test)} test examples to {OUT_DIR}")
    print("shapes: X_*.bin (N,2) float32, y_*.bin (N,1) float32, raw row-major, no header")


if __name__ == "__main__":
    main()
