"""Shared plumbing for the Python-side workers: paths, core pinning, the
timing harness, and the evaluation metrics. The C++ runner (suite_cpp.cpp)
re-implements the harness and metrics with the same constants -- keep the
two in sync.
"""

import json
import os
import sys
import time

import numpy as np

SUITE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(SUITE_DIR, "..", ".."))
DATA_DIR = os.path.join(SUITE_DIR, "data")          # gitignored: datasets, .mlpf, perms
RESULTS_DIR = os.path.join(SUITE_DIR, "results")    # checked in: raw JSON per mode
MODELS_DIR = os.path.join(DATA_DIR, "models")
CPP_BIN = os.environ.get("FASTMLP_SUITE_CPP", os.path.join(REPO, "build-bench", "benchmarks", "suite", "suite_cpp"))
CORE = int(os.environ.get("FASTMLP_SUITE_CORE", "0"))

sys.path.insert(0, os.path.join(REPO, "python"))

# Timing harness: a few warmup calls, calibrate how many calls fill one
# ~TARGET_SAMPLE_S sample, one warmup sample, then the median of REPS
# samples. Calibrating instead of a fixed inner count keeps a 2 us tiny-model
# call and a 60 ms wide-model call equally well resolved, at bounded cost.
WARMUP_CALLS = 3
TARGET_SAMPLE_S = 0.05
REPS = 7


def pin():
    os.sched_setaffinity(0, {CORE})


def measure(fn):
    for _ in range(WARMUP_CALLS):
        fn()
    t0 = time.perf_counter()
    fn()
    one = time.perf_counter() - t0
    inner = max(1, int(TARGET_SAMPLE_S / max(one, 1e-7)))
    samples = []
    for r in range(REPS + 1):
        t0 = time.perf_counter()
        for _ in range(inner):
            fn()
        dt = (time.perf_counter() - t0) / inner
        if r:  # first sample is warmup
            samples.append(dt)
    samples.sort()
    return samples[len(samples) // 2], inner


def emit(record):
    """One result per line on stdout, prefixed so the driver can tell it from
    library chatter (torch.compile logs to stdout too)."""
    print("RESULT " + json.dumps(record), flush=True)


# --- datasets -----------------------------------------------------------------

def load_task(task_name):
    with open(os.path.join(DATA_DIR, "tasks", task_name, "meta.json")) as f:
        meta = json.load(f)

    def rd(path, rows, cols):
        return np.fromfile(path, dtype=np.float32).reshape(rows, cols)

    d = {"meta": meta}
    d["x_train"] = [rd(p, meta["n_train"], w) for p, w in zip(meta["x_train"], meta["leaf_widths"])]
    d["x_test"] = [rd(p, meta["n_test"], w) for p, w in zip(meta["x_test"], meta["leaf_widths"])]
    d["y_train"] = rd(meta["y_train"], meta["n_train"], meta["y_width"])
    d["y_test"] = rd(meta["y_test"], meta["n_test"], meta["y_width"])
    d["perm"] = np.fromfile(meta["perm"], dtype=np.int32).reshape(meta["epochs"], meta["n_train"])
    return d


# --- metrics (identical definitions in suite_cpp.cpp) ---------------------------

BCE_EPS = 1e-7


def metrics(loss, pred, y):
    """`pred`: model outputs on the whole test set (logits for "ce", the
    sigmoid output for "bce"). `y`: float32 targets as stored on disk (class
    index as a float for "ce")."""
    pred = pred.astype(np.float64)
    if loss == "ce":
        labels = y[:, 0].astype(np.int64)
        z = pred - pred.max(axis=1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        return {"test_loss": float(-logp[np.arange(len(labels)), labels].mean()),
                "test_acc": float((pred.argmax(axis=1) == labels).mean())}
    if loss == "bce":
        p = np.clip(pred[:, 0], BCE_EPS, 1 - BCE_EPS)
        t = y[:, 0]
        return {"test_loss": float(-(t * np.log(p) + (1 - t) * np.log(1 - p)).mean()),
                "test_acc": float(((p > 0.5) == (t > 0.5)).mean())}
    return {"test_loss": float(((pred - y) ** 2).mean())}   # mse: mean over elements
