"""Generates a synthetic two-branch fusion-regression dataset once, writing
raw float32 binary files so all three example implementations (C++ core,
Python fastmlp, PyTorch) train on byte-identical data.

The task genuinely needs BOTH branches: target = (tanh(mean(A)) *
tanh(mean(B)) + 1) / 2 is a multiplicative interaction between the two
feature groups, not something either branch's features alone determine (a
model that only sees A, or only B, cannot predict it -- it needs the fused
representation). That's the point of this example: it's the smallest
non-contrived case that actually requires fm.MLP(inputs=[enc_a, enc_b])'s
tree composition rather than a single concatenated-input MLP.

Run once (writes into examples/tree_regression/data/, gitignored):
    python3 examples/tree_regression/generate_data.py
"""
import os

import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "data")

N_A, N_B = 32, 16


def make_dataset(n_samples, seed):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n_samples, N_A)).astype(np.float32)
    B = rng.standard_normal((n_samples, N_B)).astype(np.float32)
    interaction = np.tanh(A.mean(axis=1)) * np.tanh(B.mean(axis=1))
    y = ((interaction + 1.0) / 2.0 + rng.normal(scale=0.02, size=n_samples)).astype(np.float32)
    y = np.clip(y, 0.0, 1.0).reshape(-1, 1)
    return A, B, y


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    A_train, B_train, y_train = make_dataset(4000, seed=0)
    A_test, B_test, y_test = make_dataset(1000, seed=1)

    for name, arr in [
        ("A_train.bin", A_train), ("B_train.bin", B_train), ("y_train.bin", y_train),
        ("A_test.bin", A_test), ("B_test.bin", B_test), ("y_test.bin", y_test),
    ]:
        np.ascontiguousarray(arr, dtype=np.float32).tofile(os.path.join(OUT_DIR, name))

    print(f"wrote {len(A_train)} train / {len(A_test)} test examples to {OUT_DIR}")
    print(f"shapes: A (N,{N_A}) float32, B (N,{N_B}) float32, y (N,1) float32, raw row-major")


if __name__ == "__main__":
    main()
