"""Inference-only throughput across the same architectures as
infer_fastmlp.py / infer_cpp.cpp, via PyTorch, single-thread.

model.eval() + torch.no_grad() are used because that's the standard,
idiomatic way anyone runs PyTorch inference -- not an extra optimization
applied to make this comparison favorable to fastmlp. No TorchScript, no
torch.compile, no other inference-specific acceleration: plain eager-mode
PyTorch, same as the rest of this project's PyTorch comparisons.

Run:
    python3 examples/inference/infer_torch.py
"""
import os
import time

os.sched_setaffinity(0, {0})  # pin to core 0 -- reduces scheduling noise in per-call timing

import numpy as np
import torch

torch.set_num_threads(1)

MAX_BATCH = 128
WARMUP_BLOCKS, INNER_ITERS, OUTER_REPS = 3, 200, 7

# Mostly architectures one would actually encounter, plus one deliberately
# unusual shape (deep_narrow) for variety, plus square_WxD (width W in
# {64,128,256,512}, depth 2 or 3 entries in layer_sizes i.e. 1 or 2 hidden
# layers, in_features==out_features==W throughout) -- see infer_cpp.cpp.
ARCHITECTURES = [
    ("tiny_tabular", [8, 16, 8, 1]),
    ("small_classifier", [20, 64, 32, 1]),
    ("mnist_shallow", [784, 128, 10]),
    ("mnist_deep", [784, 256, 128, 10]),
    ("medium_classifier", [128, 256, 128, 64, 10]),
    ("wide_shallow", [100, 1024, 10]),
    ("deep_narrow", [64] * 9 + [1]),
] + [
    (f"square_{x}x{d}", [x] * d)
    for x in (64, 128, 256, 512)
    for d in (2, 3)
]


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


def build_model(layers):
    modules = []
    for i in range(len(layers) - 1):
        modules.append(torch.nn.Linear(layers[i], layers[i + 1]))
        modules.append(torch.nn.ReLU())
    return torch.nn.Sequential(*modules)


def count_params(layers):
    return sum(layers[i] * layers[i + 1] + layers[i + 1] for i in range(len(layers) - 1))


def main():
    print(f"{'arch':18s} {'params':>9s} {'batch':>6s} {'us/call':>10s} {'infer/s':>12s}")
    torch.manual_seed(1)
    rng = np.random.default_rng(0)
    for name, layers in ARCHITECTURES:
        model = build_model(layers)
        model.eval()
        x_full = torch.from_numpy(rng.random((MAX_BATCH, layers[0])).astype(np.float32))
        for batch in (1, MAX_BATCH):
            xb = x_full[:batch]
            with torch.no_grad():
                t = median_time_per_call(lambda: model(xb), WARMUP_BLOCKS, INNER_ITERS,
                                          OUTER_REPS)
            print(f"{name:18s} {count_params(layers):9d} {batch:6d} {t*1e6:10.2f} {batch/t:12.0f}")


if __name__ == "__main__":
    main()
