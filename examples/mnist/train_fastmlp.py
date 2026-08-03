"""MNIST training via the fastmlp Python API, reading the raw files
prepare_data.py writes -- directly comparable to train_cpp.cpp (same data,
architecture, hyperparameters) and train_torch.py.

fastmlp has no softmax/cross-entropy (idea.md's fixed, small op set), so
this trains as MSE-regression-to-one-hot instead (the same relu+mse_grad
pattern usages.md ex2 already documents, just applied to a classification
target), reading the predicted class off as argmax(pred). Uses
activation="sigmoid" (bounded output, matches [0,1] one-hot targets) rather
than "relu": verified empirically that relu-everywhere plateaus around 69%
test accuracy (dead-ReLU pathology on the output layer under MSE), while
sigmoid-everywhere reaches ~98%.

Run:
    python3 examples/mnist/prepare_data.py   # once
    python3 examples/mnist/train_fastmlp.py
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
N_EPOCHS = 15
LR = 1e-3
WARMUP_STEPS = 469  # skip the whole first epoch (60000 / 128)


def load_images(name, n):
    raw = np.fromfile(os.path.join(DATA_DIR, name), dtype=np.uint8)
    return raw.reshape(n, 784).astype(np.float32) / 255.0


def load_labels(name, n):
    return np.fromfile(os.path.join(DATA_DIR, name), dtype=np.uint8)


def evaluate(model, X, y_int):
    correct = 0
    for i in range(0, len(X), BATCH):
        xb = np.ascontiguousarray(X[i:i + BATCH])
        pred = model(xb)
        correct += int((np.argmax(pred, axis=1) == y_int[i:i + len(xb)]).sum())
    return correct / len(X)


def main():
    X_train = load_images("train_images.bin", 60000)
    y_train_int = load_labels("train_labels.bin", 60000)
    X_test = load_images("test_images.bin", 10000)
    y_test_int = load_labels("test_labels.bin", 10000)
    Y_train = np.eye(10, dtype=np.float32)[y_train_int]

    model = fm.MLP([784, 256, 128, 10], activation="sigmoid", max_batch=BATCH, seed=1)
    opt = fm.Adam(model, lr=LR)

    rng = np.random.default_rng(0)
    step_times = []
    step = 0
    for epoch in range(N_EPOCHS):
        perm = rng.permutation(len(X_train))
        Xs, Ys = X_train[perm], Y_train[perm]
        for i in range(0, len(X_train), BATCH):
            xb = np.ascontiguousarray(Xs[i:i + BATCH])
            yb = np.ascontiguousarray(Ys[i:i + BATCH])

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
        print(f"  epoch {epoch+1}/{N_EPOCHS} done")

    train_acc = evaluate(model, X_train, y_train_int)
    test_acc = evaluate(model, X_test, y_test_int)
    st = np.array(step_times)
    print(f"[fastmlp/python] train_acc={train_acc:.4f} test_acc={test_acc:.4f}")
    print(f"[fastmlp/python] {len(st)} timed steps (after {WARMUP_STEPS} warmup): "
          f"median {np.median(st)*1e6:.1f} us/iter, mean {st.mean()*1e6:.1f} us/iter, "
          f"{1.0/st.mean():.0f} iter/s")


if __name__ == "__main__":
    main()
