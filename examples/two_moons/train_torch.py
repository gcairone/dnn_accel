"""two-moons training via PyTorch, single-thread, same architecture, data,
and hyperparameters as train_fastmlp.py / train_cpp.cpp -- the comparison
point for this example, not the official acceptance benchmark (that's
benchmarks/python/bench_train_vs_torch.py, with a stricter manual-harness
methodology).

bce_loss below is written out explicitly (sum over the output features,
mean over the batch) rather than torch.nn.BCELoss()'s default
mean-over-everything reduction, so its gradient matches
python/fastmlp/losses.py::bce_grad exactly (same eps clipping too) -- with
a single output feature here the two reductions happen to coincide
numerically, but writing it out keeps the comparison methodology explicit
and reusable for examples/mnist, where they do NOT coincide.

Run:
    python3 examples/two_moons/generate_data.py   # once
    python3 examples/two_moons/train_torch.py
"""
import os
import time

os.sched_setaffinity(0, {0})  # pin to core 0 -- reduces scheduling noise in per-iteration timing

import numpy as np
import torch

torch.set_num_threads(1)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
BATCH = 64
N_EPOCHS = 200
LR = 0.5
WARMUP_STEPS = 25
EPS = 1e-7


def load(name, n, k):
    return np.fromfile(os.path.join(DATA_DIR, name), dtype=np.float32).reshape(n, k)


def bce_loss(pred, target):
    p = torch.clamp(pred, EPS, 1.0 - EPS)
    per_example = -(target * torch.log(p) + (1 - target) * torch.log(1 - p)).sum(dim=1)
    return per_example.mean()


def accuracy(model, X, y):
    with torch.no_grad():
        pred = model(torch.from_numpy(X))
    return float(((pred.numpy() > 0.5) == (y > 0.5)).mean())


def main():
    X_train = load("X_train.bin", 1600, 2)
    y_train = load("y_train.bin", 1600, 1)
    X_test = load("X_test.bin", 400, 2)
    y_test = load("y_test.bin", 400, 1)

    torch.manual_seed(1)
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 32), torch.nn.Sigmoid(),
        torch.nn.Linear(32, 16), torch.nn.Sigmoid(),
        torch.nn.Linear(16, 1), torch.nn.Sigmoid(),
    )
    opt = torch.optim.SGD(model.parameters(), lr=LR)

    rng = np.random.default_rng(42)
    step_times = []
    step = 0
    for epoch in range(N_EPOCHS):
        perm = rng.permutation(len(X_train))
        Xs, ys = X_train[perm], y_train[perm]
        for i in range(0, len(X_train), BATCH):
            xb = torch.from_numpy(np.ascontiguousarray(Xs[i:i + BATCH]))
            yb = torch.from_numpy(np.ascontiguousarray(ys[i:i + BATCH]))

            t0 = time.perf_counter()
            opt.zero_grad()
            pred = model(xb)
            loss = bce_loss(pred, yb)
            loss.backward()
            opt.step()
            t1 = time.perf_counter()

            step += 1
            if step > WARMUP_STEPS:
                step_times.append(t1 - t0)

    train_acc = accuracy(model, X_train, y_train)
    test_acc = accuracy(model, X_test, y_test)
    st = np.array(step_times)
    print(f"[pytorch] train_acc={train_acc:.3f} test_acc={test_acc:.3f}")
    print(f"[pytorch] {len(st)} timed steps (after {WARMUP_STEPS} warmup): "
          f"median {np.median(st)*1e6:.1f} us/iter, mean {st.mean()*1e6:.1f} us/iter, "
          f"{1.0/st.mean():.0f} iter/s")


if __name__ == "__main__":
    main()
