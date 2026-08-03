"""Deep single-branch regression via PyTorch, single-thread, same
architecture/data/hyperparameters as train_fastmlp.py / train_cpp.cpp.

mse_loss is written out explicitly (sum over features, mean over batch) to
match python/fastmlp/losses.py::mse_grad's implied loss -- with a single
output feature here it coincides with torch's default mean reduction, but
kept explicit for consistency with the other examples.

Run:
    python3 examples/deep_mlp/generate_data.py   # once
    python3 examples/deep_mlp/train_torch.py
"""
import os
import time

os.sched_setaffinity(0, {0})  # pin to core 0 -- reduces scheduling noise in per-iteration timing

import numpy as np
import torch

torch.set_num_threads(1)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
BATCH = 128
N_EPOCHS = 60
LR = 1e-3
WEIGHT_DECAY = 1e-4
N_FEATURES = 20
N_TRAIN, N_TEST = 8000, 2000
LAYERS = [N_FEATURES, 48, 48, 48, 24, 12, 1]
WARMUP_STEPS = N_TRAIN // BATCH


def load(name, n, k):
    return np.fromfile(os.path.join(DATA_DIR, name), dtype=np.float32).reshape(n, k)


def build_model():
    modules = []
    for i in range(len(LAYERS) - 1):
        modules.append(torch.nn.Linear(LAYERS[i], LAYERS[i + 1]))
        modules.append(torch.nn.Tanh())
    return torch.nn.Sequential(*modules)


def mse_loss(pred, target):
    return ((pred - target) ** 2).sum(dim=1).mean()


def evaluate(model, X, y):
    total_se = 0.0
    with torch.no_grad():
        for i in range(0, len(X), BATCH):
            xb = torch.from_numpy(np.ascontiguousarray(X[i:i + BATCH]))
            pred = model(xb).numpy()
            total_se += float(np.sum((pred - y[i:i + len(xb)]) ** 2))
    return total_se / len(X)


def main():
    X_train = load("X_train.bin", N_TRAIN, N_FEATURES)
    y_train = load("y_train.bin", N_TRAIN, 1)
    X_test = load("X_test.bin", N_TEST, N_FEATURES)
    y_test = load("y_test.bin", N_TEST, 1)

    torch.manual_seed(1)
    model = build_model()
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    rng = np.random.default_rng(7)
    n_train = len(X_train)
    step_times = []
    step = 0
    for epoch in range(N_EPOCHS):
        perm = rng.permutation(n_train)
        Xs, ys = X_train[perm], y_train[perm]
        for i in range(0, n_train, BATCH):
            xb = torch.from_numpy(np.ascontiguousarray(Xs[i:i + BATCH]))
            yb = torch.from_numpy(np.ascontiguousarray(ys[i:i + BATCH]))

            t0 = time.perf_counter()
            opt.zero_grad()
            pred = model(xb)
            loss = mse_loss(pred, yb)
            loss.backward()
            opt.step()
            t1 = time.perf_counter()

            step += 1
            if step > WARMUP_STEPS:
                step_times.append(t1 - t0)

    train_mse = evaluate(model, X_train, y_train)
    test_mse = evaluate(model, X_test, y_test)
    st = np.array(step_times)
    print(f"[pytorch] train_mse={train_mse:.5f} test_mse={test_mse:.5f}")
    print(f"[pytorch] {len(st)} timed steps (after {WARMUP_STEPS} warmup): "
          f"median {np.median(st)*1e6:.1f} us/iter, mean {st.mean()*1e6:.1f} us/iter, "
          f"{1.0/st.mean():.0f} iter/s")


if __name__ == "__main__":
    main()
