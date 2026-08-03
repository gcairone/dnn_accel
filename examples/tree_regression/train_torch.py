"""Tree-regression training via PyTorch, single-thread, same architecture,
data, and hyperparameters as train_fastmlp.py / train_cpp.cpp -- the two
branches are separate nn.Sequential stacks, concatenated manually in
forward(), mirroring fastmlp's fm.MLP(inputs=[enc_a, enc_b]) exactly.

mse_loss is written out explicitly (sum over features, mean over batch) to
match python/fastmlp/losses.py::mse_grad's implied loss -- see
examples/mnist/train_torch.py's header comment for why this matters whenever
out_features > 1 (here it's 1, so it happens to coincide with torch's
default mean reduction, but the explicit form is kept for consistency).

Run:
    python3 examples/tree_regression/generate_data.py   # once
    python3 examples/tree_regression/train_torch.py
"""
import os
import time

os.sched_setaffinity(0, {0})  # pin to core 0 -- reduces scheduling noise in per-iteration timing

import numpy as np
import torch

torch.set_num_threads(1)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
BATCH = 64
N_EPOCHS = 100
LR = 1e-3
N_A, N_B = 32, 16
WARMUP_STEPS = 4000 // BATCH


def load(name, n, k):
    return np.fromfile(os.path.join(DATA_DIR, name), dtype=np.float32).reshape(n, k)


class TreeNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_a = torch.nn.Sequential(
            torch.nn.Linear(N_A, 64), torch.nn.Sigmoid(),
            torch.nn.Linear(64, 32), torch.nn.Sigmoid(),
        )
        self.enc_b = torch.nn.Sequential(
            torch.nn.Linear(N_B, 32), torch.nn.Sigmoid(),
            torch.nn.Linear(32, 32), torch.nn.Sigmoid(),
        )
        self.head = torch.nn.Sequential(
            torch.nn.Linear(64, 32), torch.nn.Sigmoid(),
            torch.nn.Linear(32, 1), torch.nn.Sigmoid(),
        )

    def forward(self, a, b):
        za = self.enc_a(a)
        zb = self.enc_b(b)
        return self.head(torch.cat([za, zb], dim=1))


def mse_loss(pred, target):
    return ((pred - target) ** 2).sum(dim=1).mean()


def evaluate(model, A, B, y):
    total_se = 0.0
    with torch.no_grad():
        for i in range(0, len(A), BATCH):
            ab = torch.from_numpy(np.ascontiguousarray(A[i:i + BATCH]))
            bb = torch.from_numpy(np.ascontiguousarray(B[i:i + BATCH]))
            pred = model(ab, bb).numpy()
            total_se += float(np.sum((pred - y[i:i + len(ab)]) ** 2))
    return total_se / len(A)


def main():
    A_train = load("A_train.bin", 4000, N_A)
    B_train = load("B_train.bin", 4000, N_B)
    y_train = load("y_train.bin", 4000, 1)
    A_test = load("A_test.bin", 1000, N_A)
    B_test = load("B_test.bin", 1000, N_B)
    y_test = load("y_test.bin", 1000, 1)

    torch.manual_seed(3)
    model = TreeNet()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    rng = np.random.default_rng(42)
    n_train = len(A_train)
    step_times = []
    step = 0
    for epoch in range(N_EPOCHS):
        perm = rng.permutation(n_train)
        As, Bs, ys = A_train[perm], B_train[perm], y_train[perm]
        for i in range(0, n_train, BATCH):
            ab = torch.from_numpy(np.ascontiguousarray(As[i:i + BATCH]))
            bb = torch.from_numpy(np.ascontiguousarray(Bs[i:i + BATCH]))
            yb = torch.from_numpy(np.ascontiguousarray(ys[i:i + BATCH]))

            t0 = time.perf_counter()
            opt.zero_grad()
            pred = model(ab, bb)
            loss = mse_loss(pred, yb)
            loss.backward()
            opt.step()
            t1 = time.perf_counter()

            step += 1
            if step > WARMUP_STEPS:
                step_times.append(t1 - t0)

    train_mse = evaluate(model, A_train, B_train, y_train)
    test_mse = evaluate(model, A_test, B_test, y_test)
    st = np.array(step_times)
    print(f"[pytorch] train_mse={train_mse:.5f} test_mse={test_mse:.5f}")
    print(f"[pytorch] {len(st)} timed steps (after {WARMUP_STEPS} warmup): "
          f"median {np.median(st)*1e6:.1f} us/iter, mean {st.mean()*1e6:.1f} us/iter, "
          f"{1.0/st.mean():.0f} iter/s")


if __name__ == "__main__":
    main()
