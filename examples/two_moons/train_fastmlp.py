"""two-moons training via the fastmlp Python API, reading the fixed dataset
generate_data.py writes -- directly comparable to train_cpp.cpp (same data,
same architecture/hyperparameters) and train_torch.py.

Uses activation="sigmoid" on the whole stack, not "relu" as usages.md ex1
shows literally: with relu on the final layer too, the output is unbounded
above, and bce_grad's clip(pred, eps, 1-eps) then produces a huge gradient
whenever a large positive prediction meets a target of 0 (denominator
p*(1-p) -> 0). Verified empirically -- relu-everywhere collapses to chance
level (~53% test acc) after epoch 0; sigmoid-everywhere reaches 100%. This
is a real constraint of the current API (one activation for the whole
stack, no per-layer override), not a core-library bug -- see the project notes.

Run:
    python3 examples/two_moons/generate_data.py   # once
    python3 examples/two_moons/train_fastmlp.py
"""
import os
import sys
import time

os.sched_setaffinity(0, {0})  # pin to core 0 -- reduces scheduling noise in per-iteration timing

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
import numpy as np

import fastmlp as fm

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
BATCH = 64
N_EPOCHS = 200
LR = 0.5
WARMUP_STEPS = 25  # one epoch's worth of batches (1600 / 64)


def load(name, n, k):
    return np.fromfile(os.path.join(DATA_DIR, name), dtype=np.float32).reshape(n, k)


def accuracy(model, X, y):
    preds = []
    for i in range(0, len(X), BATCH):
        xb = np.ascontiguousarray(X[i:i + BATCH])
        preds.append(model(xb) > 0.5)
    preds = np.concatenate(preds)
    return float((preds == (y > 0.5)).mean())


def main():
    X_train = load("X_train.bin", 1600, 2)
    y_train = load("y_train.bin", 1600, 1)
    X_test = load("X_test.bin", 400, 2)
    y_test = load("y_test.bin", 400, 1)

    model = fm.MLP([2, 32, 16, 1], activation="sigmoid", max_batch=BATCH, seed=1)
    opt = fm.SGD(model, lr=LR)

    rng = np.random.default_rng(42)
    step_times = []
    step = 0
    for epoch in range(N_EPOCHS):
        perm = rng.permutation(len(X_train))
        Xs, ys = X_train[perm], y_train[perm]
        for i in range(0, len(X_train), BATCH):
            xb = np.ascontiguousarray(Xs[i:i + BATCH])
            yb = np.ascontiguousarray(ys[i:i + BATCH])

            t0 = time.perf_counter()
            pred = model(xb)
            grad = fm.bce_grad(pred, yb)
            model.backward(grad)
            opt.step()
            opt.zero_grad()
            t1 = time.perf_counter()

            step += 1
            if step > WARMUP_STEPS:
                step_times.append(t1 - t0)

    train_acc = accuracy(model, X_train, y_train)
    test_acc = accuracy(model, X_test, y_test)
    st = np.array(step_times)
    print(f"[fastmlp/python] train_acc={train_acc:.3f} test_acc={test_acc:.3f}")
    print(f"[fastmlp/python] {len(st)} timed steps (after {WARMUP_STEPS} warmup): "
          f"median {np.median(st)*1e6:.1f} us/iter, mean {st.mean()*1e6:.1f} us/iter, "
          f"{1.0/st.mean():.0f} iter/s")


if __name__ == "__main__":
    main()
