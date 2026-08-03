# fastmlp

A minimal C++ MLP library (Linear + activation layers, tree composition,
SGD/Adam), bound to Python via `nanobind`, hand-tuned in AVX2 for one fixed
CPU (AMD Ryzen 7 4800H / Zen 2), single-threaded. The point: PyTorch's CPU
path pays for a computational graph, dynamic dispatch, and a general
thread-pool BLAS that a small-to-medium, statically-shaped MLP doesn't need.
This library skips all of that and, on this specific CPU, beats PyTorch's
single-thread throughput on most shapes measured so far — training and
inference alike.

Full design rationale: [`idea.md`](idea.md). Intended API by example:
[`usages.md`](usages.md). Repo layout, file-by-file rationale, and
build/CI conventions: the project notes (not tracked). This file is the fast path
into all three.

## Current numbers

Single-core, pinned, `torch.set_num_threads(1)`. Headline results only —
full tables, methodology, and honest caveats live next to the code that
produced them:
[`benchmarks/cpp/results.md`](benchmarks/cpp/results.md) (kernel-level),
[`benchmarks/python/results.md`](benchmarks/python/results.md) (training,
the acceptance benchmark), [`examples/inference/results.md`](examples/inference/results.md)
(inference-only).

| | fastmlp | PyTorch | speedup |
|---|---|---|---|
| training, `[20,64,32,1]` B=128 | 47 us/step | 408 us/step | **8.65x** |
| training, `[512,256,128,1]` B=256 | 4083 us/step | 3241 us/step | 0.79x (still losing) |
| inference, `[8,16,8,1]` B=1 | 0.22 us/call | 33.6 us/call | **155x** |
| inference, `[784,256,128,10]` B=1 | 22.6 us/call | 33.4 us/call | 1.48x (near parity) |

## What's in scope

`Linear` layers, four activations (ReLU/sigmoid/tanh/identity), tree-only
composition (leaf modules concatenated feature-wise into a parent), SGD/Adam,
`.mlpf` save/load. Static shapes only — architecture and max batch size are
fixed at construction, everything is preallocated. No autograd, no conv/RNN/
attention, no multi-threading, no GPU. See the project notes §Hard scope boundaries
for why these are decisions, not gaps.

## Building and testing in the container

Everything — build, test, benchmark — runs inside the pinned Docker image;
nothing here is expected to build on the bare host.

```bash
# Build the image and start a long-running container (once)
docker build --network host -t fastmlp-dev -f docker/Dockerfile .
docker run -d --name fastmlp-dev --network host -v "$PWD":/workspace -w /workspace fastmlp-dev

# Everything below runs via `docker exec fastmlp-dev ...`, from /workspace
```

`--network host` works around this sandbox's broken bridge-network DNS; see
the project notes §Docker environment if you're not in that environment and it
turns out to be unnecessary.

**C++ core + tests** (two configurations matter: `SCALAR` is the ground
truth the `AVX2` kernel is checked against — always build/test both after
touching anything under `core/`):

```bash
docker exec fastmlp-dev bash -c '
  cmake -S . -B build         -DFASTMLP_SIMD_BACKEND=SCALAR -DFASTMLP_BUILD_PYTHON=OFF
  cmake -S . -B build-avx2    -DFASTMLP_SIMD_BACKEND=AVX2
  cmake --build build      -j$(nproc) && ctest --test-dir build      --output-on-failure
  cmake --build build-avx2 -j$(nproc) && ctest --test-dir build-avx2 --output-on-failure
'
```

**Python bindings + API tests** (needs `build-avx2` built first — that's
what produces `python/fastmlp/_fastmlp*.so`):

```bash
docker exec fastmlp-dev bash -c 'cd /workspace && python3 -m pytest tests/python -q'
```

**Benchmarks** (own build dir, so rebuilding it never clobbers the shipped
extension — see the project notes if you want to know why there are four build
directories in this repo):

```bash
docker exec fastmlp-dev bash -c '
  cmake -S . -B build-bench -DFASTMLP_SIMD_BACKEND=AVX2 -DFASTMLP_BUILD_BENCHMARKS=ON -DFASTMLP_BUILD_TESTS=OFF -DFASTMLP_BUILD_PYTHON=OFF
  cmake --build build-bench -j$(nproc)
  ./build-bench/benchmarks/cpp/bench_breakdown   # phase-by-phase profile of a real training step
  python3 benchmarks/python/bench_train_vs_torch.py
'
```

**Examples** (`examples/`, its own build dir): every subdirectory follows
the same pattern — an optional one-time data-prep script
(`generate_data.py`/`prepare_data.py`), then a C++ program and two Python
scripts running the identical architecture/data through the C++ core, the
Python bindings, and PyTorch, so timings are directly comparable. One build
covers all of them; see each directory (or the project notes' layout section)
for what's currently there.

```bash
docker exec fastmlp-dev bash -c '
  cd /workspace
  for f in examples/*/generate_data.py examples/*/prepare_data.py; do [ -f "$f" ] && python3 "$f"; done
  cmake -S . -B build-examples -DFASTMLP_SIMD_BACKEND=AVX2 -DFASTMLP_BUILD_EXAMPLES=ON -DFASTMLP_BUILD_TESTS=OFF -DFASTMLP_BUILD_PYTHON=OFF
  cmake --build build-examples -j$(nproc)
'
# then run whichever one you want, e.g.:
docker exec fastmlp-dev bash -c 'cd /workspace && ./build-examples/examples/mnist_cpp'
docker exec fastmlp-dev bash -c 'cd /workspace && python3 examples/inference/infer_torch.py'
```

## Using the library (Python)

```python
import fastmlp as fm
import numpy as np

# Binary classifier. activation applies to every layer including the last,
# so for a sigmoid-range output ("prob of class 1") the whole stack must be
# sigmoid -- relu-on-everything is numerically unstable with bce_grad (see
# the project notes' "Performance invariants" section, or just don't do it).
model = fm.MLP([20, 64, 32, 1], activation="sigmoid", max_batch=128, seed=0)
opt = fm.SGD(model, lr=0.5)

for epoch in range(100):
    for xb, yb in dataloader:              # xb: (B,20) float32, yb: (B,1) float32
        pred = model(xb)
        model.backward(fm.bce_grad(pred, yb))
        opt.step()
        opt.zero_grad()

preds = model(x_test) > 0.5
```

A tree, built from independently-reusable branches (Adam, freezing one
branch, `.mlpf` save/load):

```python
enc_img  = fm.MLP([512, 128, 64], activation="relu", max_batch=64)
enc_meta = fm.MLP([8, 32, 64],    activation="relu", max_batch=64)
model = fm.MLP([64 + 64, 32, 1], activation="relu",
               inputs=[enc_img, enc_meta], max_batch=64)

opt = fm.Adam(model, lr=1e-3)
enc_meta.freeze()  # e.g. a pretrained branch: forward/backward still run, no weight update

for img_feat, meta_feat, target in dataloader:
    pred = model(img_feat, meta_feat)
    model.backward(fm.mse_grad(pred, target))
    opt.step()
    opt.zero_grad()

model.save("model.mlpf")
loaded = fm.MLP.load("model.mlpf")
```

NumPy contract: every array crossing into fastmlp must be `float32` and
C-contiguous, or it raises — never a silent cast or copy. Every `model(x)`
call returns a view into a buffer fastmlp owns and reuses; copy it
(`.copy()`) before the next call if you need it to survive one.

More scenarios (deep trees, `inference_only`, dict-style calls for named
branches): [`usages.md`](usages.md). Runnable, three-implementation versions
of these and more architectures: [`examples/`](examples/).

## Using the library (C++ core directly)

The core has zero Python/nanobind dependency and is independently
buildable/testable (`core/`, `tests/cpp/`). See any `examples/*/train_cpp.cpp`
or `examples/*/infer_cpp.cpp` for a complete, working program against the
raw `fastmlp::Module`/`fastmlp::SGD`/`fastmlp::Adam` API — including the
loss gradient, since fastmlp_core has no loss-function concept (that's the
caller's job in both the C++ and Python API, same boundary in both cases).
`examples/tree_regression/train_cpp.cpp` is the one to read for the tree
case: `Module(root_cfg, {&enc_a, &enc_b})`.
