"""MNIST training via PyTorch, single-thread, same architecture, data, and
hyperparameters as train_fastmlp.py / train_cpp.cpp.

mse_loss below is written out explicitly as sum-over-features,
mean-over-batch, matching python/fastmlp/losses.py::mse_grad's implied loss
exactly -- torch.nn.MSELoss()'s default reduction='mean' averages over
*all* elements (batch*10 here), which would scale the gradient by 1/10
relative to fastmlp's and make this not a fair comparison.

Run:
    python3 examples/mnist/prepare_data.py   # once
    python3 examples/mnist/train_torch.py
"""
import os
import time

os.sched_setaffinity(0, {0})  # pin to core 0 -- reduces scheduling noise in per-iteration timing

import numpy as np
import torch

torch.set_num_threads(1)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
BATCH = 128
N_EPOCHS = 15
LR = 1e-3
WARMUP_STEPS = 469


def load_images(name, n):
    raw = np.fromfile(os.path.join(DATA_DIR, name), dtype=np.uint8)
    return raw.reshape(n, 784).astype(np.float32) / 255.0


def load_labels(name, n):
    return np.fromfile(os.path.join(DATA_DIR, name), dtype=np.uint8)


def mse_loss(pred, target):
    return ((pred - target) ** 2).sum(dim=1).mean()


def evaluate(model, X, y_int):
    correct = 0
    with torch.no_grad():
        for i in range(0, len(X), BATCH):
            xb = torch.from_numpy(np.ascontiguousarray(X[i:i + BATCH]))
            pred = model(xb).numpy()
            correct += int((np.argmax(pred, axis=1) == y_int[i:i + len(xb)]).sum())
    return correct / len(X)


def main():
    X_train = load_images("train_images.bin", 60000)
    y_train_int = load_labels("train_labels.bin", 60000)
    X_test = load_images("test_images.bin", 10000)
    y_test_int = load_labels("test_labels.bin", 10000)
    Y_train = np.eye(10, dtype=np.float32)[y_train_int]

    torch.manual_seed(1)
    model = torch.nn.Sequential(
        torch.nn.Linear(784, 256), torch.nn.Sigmoid(),
        torch.nn.Linear(256, 128), torch.nn.Sigmoid(),
        torch.nn.Linear(128, 10), torch.nn.Sigmoid(),
    )
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    rng = np.random.default_rng(0)
    step_times = []
    step = 0
    for epoch in range(N_EPOCHS):
        perm = rng.permutation(len(X_train))
        Xs, Ys = X_train[perm], Y_train[perm]
        for i in range(0, len(X_train), BATCH):
            xb = torch.from_numpy(np.ascontiguousarray(Xs[i:i + BATCH]))
            yb = torch.from_numpy(np.ascontiguousarray(Ys[i:i + BATCH]))

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
        print(f"  epoch {epoch+1}/{N_EPOCHS} done")

    train_acc = evaluate(model, X_train, y_train_int)
    test_acc = evaluate(model, X_test, y_test_int)
    st = np.array(step_times)
    print(f"[pytorch] train_acc={train_acc:.4f} test_acc={test_acc:.4f}")
    print(f"[pytorch] {len(st)} timed steps (after {WARMUP_STEPS} warmup): "
          f"median {np.median(st)*1e6:.1f} us/iter, mean {st.mean()*1e6:.1f} us/iter, "
          f"{1.0/st.mean():.0f} iter/s")


if __name__ == "__main__":
    main()
