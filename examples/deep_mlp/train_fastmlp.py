"""Deep single-branch regression via the fastmlp Python API -- 6 layers
([20,48,48,48,24,12,1]), vs. the 2-3 layer stacks in the other examples,
to cover a different point in the architecture/shape space. tanh (not
sigmoid) throughout: the target is in [-1,1] to match tanh's range, and
tanh's zero-centered output tends to carry gradient better through a
6-layer stack than sigmoid does.

Run:
    python3 examples/deep_mlp/generate_data.py   # once
    python3 examples/deep_mlp/train_fastmlp.py
"""
import os
import sys
import time

os.sched_setaffinity(0, {0})  # pin to core 0 -- reduces scheduling noise in per-iteration timing

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
import numpy as np

import fastmlp as fm

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
BATCH = 128
N_EPOCHS = 60
LR = 1e-3
WEIGHT_DECAY = 1e-4
N_FEATURES = 20
N_TRAIN, N_TEST = 8000, 2000
# Narrow relative to N_TRAIN on purpose: a wider [20,128,128,128,64,32,1] (~46k
# params) on only a few thousand examples memorized the training set outright
# (train_mse 0.004 vs test_mse 0.22, measured) -- this is ~7k params, kept deep
# (6 weight layers, the point of this example) but sized to the data.
LAYERS = [N_FEATURES, 48, 48, 48, 24, 12, 1]
WARMUP_STEPS = N_TRAIN // BATCH


def load(name, n, k):
    return np.fromfile(os.path.join(DATA_DIR, name), dtype=np.float32).reshape(n, k)


def evaluate(model, X, y):
    total_se = 0.0
    for i in range(0, len(X), BATCH):
        xb = np.ascontiguousarray(X[i:i + BATCH])
        pred = model(xb)
        total_se += float(np.sum((pred - y[i:i + len(xb)]) ** 2))
    return total_se / len(X)


def main():
    X_train = load("X_train.bin", N_TRAIN, N_FEATURES)
    y_train = load("y_train.bin", N_TRAIN, 1)
    X_test = load("X_test.bin", N_TEST, N_FEATURES)
    y_test = load("y_test.bin", N_TEST, 1)

    model = fm.MLP(LAYERS, activation="tanh", max_batch=BATCH, seed=1)
    opt = fm.Adam(model, lr=LR, weight_decay=WEIGHT_DECAY)

    rng = np.random.default_rng(7)
    n_train = len(X_train)
    step_times = []
    step = 0
    for epoch in range(N_EPOCHS):
        perm = rng.permutation(n_train)
        Xs, ys = X_train[perm], y_train[perm]
        for i in range(0, n_train, BATCH):
            xb = np.ascontiguousarray(Xs[i:i + BATCH])
            yb = np.ascontiguousarray(ys[i:i + BATCH])

            t0 = time.perf_counter()
            pred = model(xb)
            grad = fm.mse_grad(pred, yb)
            model.backward(grad)
            opt.step()
            opt.zero_grad()
            t1 = time.perf_counter()

            step += 1
            if step > WARMUP_STEPS:
                step_times.append(t1 - t0)

    train_mse = evaluate(model, X_train, y_train)
    test_mse = evaluate(model, X_test, y_test)
    st = np.array(step_times)
    print(f"[fastmlp/python] train_mse={train_mse:.5f} test_mse={test_mse:.5f}")
    print(f"[fastmlp/python] {len(st)} timed steps (after {WARMUP_STEPS} warmup): "
          f"median {np.median(st)*1e6:.1f} us/iter, mean {st.mean()*1e6:.1f} us/iter, "
          f"{1.0/st.mean():.0f} iter/s")


if __name__ == "__main__":
    main()
