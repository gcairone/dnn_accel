"""Tree-regression training via the fastmlp Python API -- the fm.MLP(MLP(),
MLP()) case: two independent leaf branches (enc_a, enc_b), each with their
own layer stack, concatenated feature-wise and fed into a root MLP built on
top. Directly comparable to train_cpp.cpp (same data, same architecture) and
train_torch.py.

Run:
    python3 examples/tree_regression/generate_data.py   # once
    python3 examples/tree_regression/train_fastmlp.py
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
N_EPOCHS = 100
LR = 1e-3
N_A, N_B = 32, 16
WARMUP_STEPS = 4000 // BATCH  # one epoch's worth of batches


def load(name, n, k):
    return np.fromfile(os.path.join(DATA_DIR, name), dtype=np.float32).reshape(n, k)


def evaluate(model, A, B, y):
    total_se = 0.0
    for i in range(0, len(A), BATCH):
        ab = np.ascontiguousarray(A[i:i + BATCH])
        bb = np.ascontiguousarray(B[i:i + BATCH])
        pred = model(ab, bb)
        total_se += float(np.sum((pred - y[i:i + len(ab)]) ** 2))
    return total_se / len(A)


def main():
    A_train = load("A_train.bin", 4000, N_A)
    B_train = load("B_train.bin", 4000, N_B)
    y_train = load("y_train.bin", 4000, 1)
    A_test = load("A_test.bin", 1000, N_A)
    B_test = load("B_test.bin", 1000, N_B)
    y_test = load("y_test.bin", 1000, 1)

    enc_a = fm.MLP([N_A, 64, 32], activation="sigmoid", max_batch=BATCH, seed=1)
    enc_b = fm.MLP([N_B, 32, 32], activation="sigmoid", max_batch=BATCH, seed=2)
    model = fm.MLP([64, 32, 1], activation="sigmoid", inputs=[enc_a, enc_b], max_batch=BATCH,
                    seed=3)
    opt = fm.Adam(model, lr=LR)

    rng = np.random.default_rng(42)
    n_train = len(A_train)
    step_times = []
    step = 0
    for epoch in range(N_EPOCHS):
        perm = rng.permutation(n_train)
        As, Bs, ys = A_train[perm], B_train[perm], y_train[perm]
        for i in range(0, n_train, BATCH):
            ab = np.ascontiguousarray(As[i:i + BATCH])
            bb = np.ascontiguousarray(Bs[i:i + BATCH])
            yb = np.ascontiguousarray(ys[i:i + BATCH])

            t0 = time.perf_counter()
            pred = model(ab, bb)
            grad = fm.mse_grad(pred, yb)
            model.backward(grad)
            opt.step()
            opt.zero_grad()
            t1 = time.perf_counter()

            step += 1
            if step > WARMUP_STEPS:
                step_times.append(t1 - t0)

    train_mse = evaluate(model, A_train, B_train, y_train)
    test_mse = evaluate(model, A_test, B_test, y_test)
    st = np.array(step_times)
    print(f"[fastmlp/python] train_mse={train_mse:.5f} test_mse={test_mse:.5f}")
    print(f"[fastmlp/python] {len(st)} timed steps (after {WARMUP_STEPS} warmup): "
          f"median {np.median(st)*1e6:.1f} us/iter, mean {st.mean()*1e6:.1f} us/iter, "
          f"{1.0/st.mean():.0f} iter/s")


if __name__ == "__main__":
    main()
