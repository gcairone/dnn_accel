"""Acceptance benchmark: fastmlp vs PyTorch CPU training step, both pinned to
a single thread/core, on the benchmarks/cpp shapes -- same manual-harness
methodology (warmup, median-of-N wall-clock) as the C++ benchmarks.

Run inside the fastmlp-dev container:
    python3 benchmarks/python/bench_train_vs_torch.py
"""

import os
import sys
import time

os.sched_setaffinity(0, {0})

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

import numpy as np
import torch

import fastmlp as fm

torch.set_num_threads(1)


def median_time_per_call(fn, warmup_blocks, inner_iters, outer_reps):
    for _ in range(warmup_blocks):
        for _ in range(inner_iters):
            fn()
    samples = []
    for _ in range(outer_reps):
        t0 = time.perf_counter()
        for _ in range(inner_iters):
            fn()
        t1 = time.perf_counter()
        samples.append((t1 - t0) / inner_iters)
    samples.sort()
    n = len(samples)
    return samples[n // 2] if n % 2 else 0.5 * (samples[n // 2 - 1] + samples[n // 2])


def bench_fastmlp(layers, batch, lr=1e-2, seed=1):
    model = fm.MLP(layers, activation="relu", max_batch=batch, seed=seed)
    opt = fm.SGD(model, lr=lr)
    rng = np.random.default_rng(seed)
    x = rng.random((batch, layers[0]), dtype=np.float32)
    target = rng.random((batch, layers[-1]), dtype=np.float32)

    def step():
        pred = model(x)
        model.backward(fm.mse_grad(pred, target))
        opt.step()
        opt.zero_grad()

    return step


def bench_torch(layers, batch, lr=1e-2, seed=1):
    torch.manual_seed(seed)
    modules = []
    for i in range(len(layers) - 1):
        modules.append(torch.nn.Linear(layers[i], layers[i + 1]))
        modules.append(torch.nn.ReLU())
    model = torch.nn.Sequential(*modules)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    rng = np.random.default_rng(seed)
    x = torch.from_numpy(rng.random((batch, layers[0]), dtype=np.float32))
    target = torch.from_numpy(rng.random((batch, layers[-1]), dtype=np.float32))

    def step():
        opt.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, target)
        loss.backward()
        opt.step()

    return step


SHAPES = [
    ("small [20,64,32,1] B=128", [20, 64, 32, 1], 128),
    ("medium [512,128,64,1] B=64", [512, 128, 64, 1], 64),
    ("wide [512,256,128,1] B=256", [512, 256, 128, 1], 256),
]

WARMUP_BLOCKS, INNER_ITERS, OUTER_REPS = 3, 200, 7


def main():
    print(f"torch: {torch.__version__}, num_threads={torch.get_num_threads()}")
    print(f"{'shape':45s} {'fastmlp us/step':>16s} {'torch us/step':>14s} {'fastmlp steps/s':>16s} {'torch steps/s':>14s} {'speedup':>8s}")
    for label, layers, batch in SHAPES:
        t_fm = median_time_per_call(bench_fastmlp(layers, batch), WARMUP_BLOCKS, INNER_ITERS, OUTER_REPS)
        t_torch = median_time_per_call(bench_torch(layers, batch), WARMUP_BLOCKS, INNER_ITERS, OUTER_REPS)
        print(f"{label:45s} {t_fm*1e6:16.1f} {t_torch*1e6:14.1f} {1/t_fm:16.0f} {1/t_torch:14.0f} {t_torch/t_fm:7.2f}x")


if __name__ == "__main__":
    main()
