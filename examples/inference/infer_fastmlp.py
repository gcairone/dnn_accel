"""Inference-only throughput across a handful of common MLP architectures,
via the fastmlp Python API -- see infer_cpp.cpp's header comment for the
architecture choices and the "normal usage, no tricks" rationale (same one
applies here: inference_only=True is the documented normal way to deploy a
trained model, not a speed hack; the benchmark itself is a plain loop of
repeated model(x) calls).

Run:
    python3 examples/inference/infer_fastmlp.py
"""
import os
import sys
import time

os.sched_setaffinity(0, {0})  # pin to core 0 -- reduces scheduling noise in per-call timing

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
import numpy as np

import fastmlp as fm

MAX_BATCH = 128
WARMUP_BLOCKS, INNER_ITERS, OUTER_REPS = 3, 200, 7

# Mostly architectures one would actually encounter, plus one deliberately
# unusual shape (deep_narrow) for variety -- see infer_cpp.cpp.
ARCHITECTURES = [
    ("tiny_tabular", [8, 16, 8, 1]),
    ("small_classifier", [20, 64, 32, 1]),
    ("mnist_shallow", [784, 128, 10]),
    ("mnist_deep", [784, 256, 128, 10]),
    ("medium_classifier", [128, 256, 128, 64, 10]),
    ("wide_shallow", [100, 1024, 10]),
    ("deep_narrow", [64] * 9 + [1]),
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


def count_params(layers):
    return sum(layers[i] * layers[i + 1] + layers[i + 1] for i in range(len(layers) - 1))


def main():
    print(f"{'arch':18s} {'params':>9s} {'batch':>6s} {'us/call':>10s} {'infer/s':>12s}")
    rng = np.random.default_rng(0)
    for name, layers in ARCHITECTURES:
        model = fm.MLP(layers, activation="relu", max_batch=MAX_BATCH, inference_only=True,
                        seed=1)
        x_full = rng.random((MAX_BATCH, layers[0])).astype(np.float32)
        for batch in (1, MAX_BATCH):
            xb = np.ascontiguousarray(x_full[:batch])
            t = median_time_per_call(lambda: model(xb), WARMUP_BLOCKS, INNER_ITERS, OUTER_REPS)
            print(f"{name:18s} {count_params(layers):9d} {batch:6d} {t*1e6:10.2f} {batch/t:12.0f}")


if __name__ == "__main__":
    main()
