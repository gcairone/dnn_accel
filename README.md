# fastmlp

A minimal C++ MLP library (Linear + activation layers, tree composition,
SGD/Adam), bound to Python via `nanobind`, hand-tuned in AVX2 for one fixed
CPU (AMD Ryzen 7 4800H / Zen 2), single-threaded.

PyTorch's CPU path pays for an autograd tape, dynamic dispatch and a general
thread-pool BLAS that a small-to-medium, statically-shaped MLP does not
need. fastmlp skips all of that. On this CPU it beats single-thread PyTorch
(eager and `torch.compile`) on every training step measured, and on
inference up to batch 128. At batch 1024 on 100k+ parameter models the two
are at parity: MKL's large-batch GEMM catches up there.

## Results

Single core, pinned, `torch.set_num_threads(1)`, 13 architectures from 1k to
2.1M parameters (tabular, RL actor/critic, two-tower recommender, NeRF,
PINN, transformer FFN block, MNIST classifiers and autoencoder, plus tree
models). Speedup of the Python API over PyTorch, geometric mean (range):

| | vs eager torch | vs torch.compile |
|---|--:|--:|
| training step, batch 64 | **4.2x** (1.45–38x) | **3.6x** (1.20–32x) |
| training step, batch 256 | **2.4x** (1.22–16x) | **2.1x** (1.11–14x) |
| inference, batch 1 | **7.5x** (1.5–53x) | **12.7x** (1.9–72x) |
| inference, batch 16 | **3.5x** (1.2–32x) | **5.0x** (1.3–54x) |
| inference, batch 128 | **1.9x** (1.2–7.7x) | **2.2x** (1.2–16x) |
| inference, batch 1024 | 1.36x (0.92–2.4x) | 1.34x (0.84–3.5x) |

Some individual cells (training step, batch 64):

| model | params | fastmlp | torch | torch.compile |
|---|--:|--:|--:|--:|
| small tabular classifier | 1k | 17 us | 585 us | 525 us |
| PPO policy | 8k | 47 us | 663 us | 578 us |
| MNIST MLP `[784,256,128,10]` | 235k | 1.27 ms | 2.68 ms | 2.60 ms |
| NeRF-style 8x256 | 478k | 3.86 ms | 5.97 ms | 5.65 ms |
| transformer FFN `[512,2048,512]` | 2.1M | 15.9 ms | 24.7 ms | 19.5 ms |

The advantage is per-call overhead, so it shrinks as compute per call grows:
the largest ratios are small models at small batch, the smallest are 2M
parameters at batch 256. Full training runs on real and synthetic data (same
initial weights, same batch order, same final loss) are 1.16x (two-tower CTR)
to 10x (a 3-class spiral) faster end to end than eager PyTorch.
`torch.compile` is the slowest of the four on every full run, because the
cold compile takes 8–15 s.

Raw measurements are in `benchmarks/suite/results/*.json`, and
`python3 benchmarks/suite/report.py` turns them into a markdown report.

## Why it is fast

- **No graph, no dispatch.** Each module has a hand-derived forward/backward.
  Every buffer is preallocated at construction, so the hot path never
  allocates.
- **Kernels tuned for one microarchitecture**, from measurements taken on
  it. The tiles keep 12 independent FMA chains in flight (Zen 2 needs ~10).
  The forward reads a transposed copy of W whose row stride is an odd
  number of cache lines. `dX`/`dW` copy each K-slice into an L1 scratch
  buffer.
- **Shape-specific paths.** Batch 1 uses a streaming GEMV over the
  untransposed weights, and the first layer of a stack skips the input
  gradient nobody reads.
- **Few passes over the parameters.** `zero_grad()` is O(1): the next
  backward overwrites the gradient instead of re-reading it. The optimizer
  rebuilds the forward's transposed weights block by block as it updates
  them, while they are still in L1.
- **Vectorized activations** (Cephes-style exp/log/tanh, erfc-based GELU)
  instead of one scalar libm call per element.

## Scope

In scope:
- `Linear` layers;
- ten activations, chosen per layer: identity, ReLU, leaky ReLU, ELU,
  sigmoid, tanh, SiLU, GELU, tanh-GELU, softplus;
- tree-only composition (leaf modules concatenated feature-wise into a
  parent);
- SGD/Adam;
- `.mlpf` save/load (one JSON header line plus a raw float32 blob).

Shapes are static: architecture and max batch size are fixed at
construction.

Out of scope, by design: autograd, conv/RNN/attention/normalization layers,
multi-threading, GPU, runtime CPU dispatch, and general DAGs. The kernels
target one known CPU and nothing else.

## Using the library (Python)

```python
import fastmlp as fm
import numpy as np

# Binary classifier. `activation` applies to every layer, `output_activation`
# overrides the last one (or pass a list, one name per layer).
model = fm.MLP([20, 64, 32, 1], activation="relu", output_activation="sigmoid",
               max_batch=128, seed=0)
opt = fm.SGD(model, lr=0.5)

for epoch in range(100):
    for xb, yb in dataloader:              # xb: (B,20) float32, yb: (B,1) float32
        pred = model(xb)
        model.backward(fm.bce_grad(pred, yb))
        opt.step()
        opt.zero_grad()

preds = model(x_test) > 0.5
```

A tree, built from independently reusable branches (Adam, freezing one
branch, `.mlpf` save/load):

```python
enc_img  = fm.MLP([512, 128, 64], activation="relu", max_batch=64, name="img")
enc_meta = fm.MLP([8, 32, 64],    activation="relu", max_batch=64, name="meta")
model = fm.MLP([64 + 64, 32, 1], activation="relu",
               inputs=[enc_img, enc_meta], max_batch=64)

opt = fm.Adam(model, lr=1e-3)     # finds both branches by walking model.inputs
enc_meta.freeze()  # e.g. a pretrained branch: forward/backward still run, no weight update

for img_feat, meta_feat, target in dataloader:
    pred = model(img_feat, meta_feat)   # or model({"img": ..., "meta": ...})
    model.backward(fm.mse_grad(pred, target))
    opt.step()
    opt.zero_grad()

model.save("model.mlpf")
loaded = fm.MLP.load("model.mlpf", inference_only=True)
```

Multi-class classification uses a linear output plus the fused softmax +
cross-entropy gradient, which takes logits:
`fm.MLP([...], activation="relu", output_activation="identity", ...)`, then
`model.backward(fm.cross_entropy_grad(logits, labels))`.

NumPy contract: every array passed to fastmlp must be `float32` and
C-contiguous, otherwise it raises; nothing is silently cast or copied. Every
`model(x)` call returns a view into a buffer that fastmlp owns and reuses.
`.copy()` it before the next call if you need to keep it.

## Using the library (C++)

The core (`core/`) has no Python dependency and builds, tests and benchmarks
on its own. Every `examples/*/train_cpp.cpp` is a complete program against
the raw `fastmlp::Module` / `fastmlp::SGD` / `fastmlp::Adam` API. Each one
includes the loss gradient, because the core stops at
forward/backward/optimizer. `examples/tree_regression/train_cpp.cpp` shows
the tree case: `Module(root_cfg, {&enc_a, &enc_b})`.

## Building and testing

Everything runs inside the Docker image in `docker/Dockerfile`, capped at 4
cores and 4 GB:

```bash
docker build --network host -t fastmlp-dev -f docker/Dockerfile .
docker run -d --name fastmlp-dev --network host --cpuset-cpus=0-3 --memory=4g --memory-swap=4g \
  -v "$PWD":/workspace -w /workspace fastmlp-dev
```

(`--network host` works around a bridge network without working DNS; drop
it if yours resolves.)

C++ core and tests. Build both backends after touching `core/`, because
`SCALAR` is the ground truth the `AVX2` kernels are checked against. `build`
has the Python extension off, so it never overwrites the AVX2 one:

```bash
docker exec fastmlp-dev bash -c '
  cmake -S . -B build      -DFASTMLP_SIMD_BACKEND=SCALAR -DFASTMLP_BUILD_PYTHON=OFF
  cmake -S . -B build-avx2 -DFASTMLP_SIMD_BACKEND=AVX2
  cmake --build build      -j4 && ctest --test-dir build      --output-on-failure
  cmake --build build-avx2 -j4 && ctest --test-dir build-avx2 --output-on-failure
'
```

Python bindings and API tests. `build-avx2` must be built first; it writes
`python/fastmlp/_fastmlp*.so`:

```bash
docker exec fastmlp-dev python3 -m pytest tests/python -q
```

Benchmarks have their own build directory:

```bash
docker exec fastmlp-dev bash -c '
  cmake -S . -B build-bench -DFASTMLP_SIMD_BACKEND=AVX2 -DFASTMLP_BUILD_BENCHMARKS=ON \
        -DFASTMLP_BUILD_TESTS=OFF -DFASTMLP_BUILD_PYTHON=OFF
  cmake --build build-bench -j4
  ./build-bench/benchmarks/cpp/bench_breakdown   # phase-by-phase profile of a training step
'
```

The full comparison suite takes about 30 minutes; `--quick` runs a subset in
about 3. `prepare_data.py` needs `examples/mnist/prepare_data.py` to have run
first:

```bash
docker exec -u $(id -u):$(id -g) fastmlp-dev bash -c '
  python3 benchmarks/suite/prepare_data.py   # once
  python3 benchmarks/suite/verify.py         # fastmlp and torch compute the same thing
  python3 benchmarks/suite/run.py all        # -> benchmarks/suite/results/
'
```

Examples get their own build too. Each directory runs one architecture and
dataset through the C++ core, the Python API and PyTorch, so the three can be
compared directly:

```bash
docker exec fastmlp-dev bash -c '
  for f in examples/*/generate_data.py examples/*/prepare_data.py; do python3 "$f"; done
  cmake -S . -B build-examples -DFASTMLP_SIMD_BACKEND=AVX2 -DFASTMLP_BUILD_EXAMPLES=ON \
        -DFASTMLP_BUILD_TESTS=OFF -DFASTMLP_BUILD_PYTHON=OFF
  cmake --build build-examples -j4
  ./build-examples/examples/mnist_cpp
'
```

## Layout

```
core/                 C++ core: Linear, Module (tree), SGD/Adam, .mlpf I/O,
                      scalar + AVX2 GEMM and activation kernels. No Python.
bindings/python/      nanobind extension: the NumPy contract lives here only
python/fastmlp/       the Python package (+ pure-NumPy loss gradients)
tests/cpp/            core tests, both backends, BLIS as an extra oracle
tests/python/         binding/API tests, incl. parity with a NumPy reference MLP
benchmarks/
  hardware_analysis/  CPU probe, BLIS/OpenBLAS baseline, roofline ceilings
  cpp/                kernel, layer and training-step benchmarks; bench_breakdown
  python/             training step vs PyTorch on three shapes
  suite/              the 13-architecture comparison (C++, Python, torch, compile)
examples/             two_moons, mnist, tree_regression, deep_mlp, inference
docker/               the dev/benchmark image
```
