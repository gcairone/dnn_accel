"""Runs one (mode, implementation, case) and prints RESULT lines. Invoked by
run.py, one process per call, so a torch.compile cache or allocator state
never leaks from one case into the next.

    python3 worker.py infer  {fastmlp|torch|torch_compile} <arch> <batch>...
    python3 worker.py train  {fastmlp|torch|torch_compile} <arch> <batch>...
    python3 worker.py loop   {fastmlp|torch|torch_compile} <task>

What each implementation is allowed to do is the point of the comparison, so
it is spelled out here once:
  fastmlp        the public Python API, used as documented.
  torch          idiomatic eager PyTorch: inference_mode() for inference,
                 zero_grad/forward/loss/backward/step for training.
  torch_compile  the same model through torch.compile(dynamic=False) --
                 inference compiles the model, training compiles the whole
                 step (forward, loss, backward, optimizer), which measured
                 fastest of the variants tried.
All of them single-threaded and pinned to one core.
"""

import atexit
import os
import shutil
import sys
import tempfile
import time

# A fresh, empty inductor cache per process: compile_s is then always a cold
# compile, as on a first run. With the default persistent cache in /tmp it
# depended on what earlier runs had left behind (5.8 s cold vs 1.1 s warm,
# same model) -- not reproducible.
_cache = tempfile.mkdtemp(prefix="suite_inductor_")
os.environ["TORCHINDUCTOR_CACHE_DIR"] = _cache
atexit.register(shutil.rmtree, _cache, ignore_errors=True)

import numpy as np

import common
from cases import ARCH_BY_NAME, TASK_BY_NAME, leaves
from common import emit, measure
from models import build_torch, fastmlp_loss_grad, model_path, torch_loss

common.pin()


_torch = None


def setup_torch():
    """Import torch single-threaded. Once per process: torch refuses a second
    set_num_interop_threads()."""
    global _torch
    if _torch is None:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        _torch = torch
    return _torch


def warm_inductor():
    """Compile and run a throwaway function once, so inductor's one-time
    import/startup (~5 s) is not billed to whichever model compiles first --
    compile_s should be that model's own compile cost."""
    torch = setup_torch()
    f = torch.compile(lambda x: torch.relu(x) * 2, dynamic=False)
    f(torch.ones(4))


def synthetic(arch, batch, rng):
    """Random leaf inputs + a target of the right kind for the arch's loss."""
    xs = [rng.standard_normal((batch, l.sizes[0]), dtype=np.float32) for l in leaves(arch.root)]
    out = arch.root.sizes[-1]
    if arch.loss == "ce":
        y = rng.integers(0, out, size=batch).astype(np.int64)
    elif arch.loss == "bce":
        y = (rng.random((batch, out)) > 0.5).astype(np.float32)
    else:
        y = rng.standard_normal((batch, out), dtype=np.float32)
    return xs, y


# --- inference ------------------------------------------------------------------------

def run_infer(impl, arch, batches):
    path = model_path(arch.name, arch.root)
    rng = np.random.default_rng(0)
    if impl != "fastmlp":
        torch = setup_torch()
        model = build_torch(arch.root, path).eval()
        if impl == "torch_compile":
            warm_inductor()
    for batch in batches:
        xs, _ = synthetic(arch, batch, rng)
        compile_s = None
        if impl == "fastmlp":
            import fastmlp as fm
            m = fm.MLP.load(path, max_batch=batch, inference_only=True)
            fn = lambda: m(*xs)
        elif impl == "torch":
            txs = [torch.from_numpy(x) for x in xs]

            def fn():
                with torch.inference_mode():
                    return model(*txs)
        else:
            import torch._dynamo
            torch._dynamo.reset()
            txs = [torch.from_numpy(x) for x in xs]
            cm = torch.compile(model, dynamic=False)
            t0 = time.perf_counter()
            with torch.no_grad():
                cm(*txs)
            compile_s = time.perf_counter() - t0

            def fn():
                with torch.no_grad():
                    return cm(*txs)
        sec, inner = measure(fn)
        emit({"mode": "infer", "impl": impl, "arch": arch.name, "batch": batch,
              "sec": sec, "inner": inner, "compile_s": compile_s})


# --- training step ----------------------------------------------------------------------

def run_train(impl, arch, batches):
    path = model_path(arch.name, arch.root)
    rng = np.random.default_rng(0)
    for batch in batches:
        xs, y = synthetic(arch, batch, rng)
        compile_s = None
        if impl == "fastmlp":
            import fastmlp as fm
            m = fm.MLP.load(path, max_batch=batch)
            opt = fm.Adam(m, lr=1e-3)
            grad = fastmlp_loss_grad(arch.loss)

            def fn():
                pred = m(*xs)
                m.backward(grad(pred, y))
                opt.step()
                opt.zero_grad()
        else:
            torch = setup_torch()
            model = build_torch(arch.root, path)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            lossf = torch_loss(arch.loss)
            txs = [torch.from_numpy(x) for x in xs]
            ty = torch.from_numpy(y)

            def step(*args):
                opt.zero_grad(set_to_none=True)
                loss = lossf(model(*args[:-1]), args[-1])
                loss.backward()
                opt.step()

            if impl == "torch_compile":
                import torch._dynamo
                warm_inductor()
                torch._dynamo.reset()
                step = torch.compile(step)
                t0 = time.perf_counter()
                step(*txs, ty)
                compile_s = time.perf_counter() - t0
            fn = lambda: step(*txs, ty)
        sec, inner = measure(fn)
        emit({"mode": "train", "impl": impl, "arch": arch.name, "batch": batch,
              "sec": sec, "inner": inner, "compile_s": compile_s})


# --- full training loop -------------------------------------------------------------------

def run_loop(impl, task):
    d = common.load_task(task.name)
    meta = d["meta"]
    loss, batch = meta["loss"], meta["batch"]
    steps = meta["n_train"] // batch     # drop_last: every step has the same shape
    y_train = d["y_train"][:, 0].astype(np.int64) if loss == "ce" else d["y_train"]

    if impl == "fastmlp":
        import fastmlp as fm
        m = fm.MLP.load(meta["model"], max_batch=batch)
        opt = (fm.Adam(m, lr=meta["lr"]) if meta["optimizer"] == "adam"
               else fm.SGD(m, lr=meta["lr"], momentum=meta["momentum"]))
        grad = fastmlp_loss_grad(loss)
        xtr = d["x_train"]

        def train_step(idx):
            pred = m(*[x[idx] for x in xtr])
            m.backward(grad(pred, y_train[idx]))
            opt.step()
            opt.zero_grad()

        def predict():
            out = []
            for a in range(0, meta["n_test"], batch):
                out.append(np.array(m(*[x[a:a + batch] for x in d["x_test"]])))
            return np.concatenate(out)

        perms = d["perm"]
        slice_idx = lambda e, s: perms[e, s * batch:(s + 1) * batch]
    else:
        torch = setup_torch()
        model = build_torch(task.arch.root, meta["model"])
        opt = (torch.optim.Adam(model.parameters(), lr=meta["lr"]) if meta["optimizer"] == "adam"
               else torch.optim.SGD(model.parameters(), lr=meta["lr"], momentum=meta["momentum"]))
        lossf = torch_loss(loss)
        xtr = [torch.from_numpy(x) for x in d["x_train"]]
        ytr = torch.from_numpy(y_train)
        xte = [torch.from_numpy(x) for x in d["x_test"]]

        def step(*args):
            opt.zero_grad(set_to_none=True)
            l = lossf(model(*args[:-1]), args[-1])
            l.backward()
            opt.step()

        if impl == "torch_compile":
            step = torch.compile(step)

        def train_step(idx):
            step(*[x[idx] for x in xtr], ytr[idx])

        def predict():
            with torch.inference_mode():
                return torch.cat([model(*[x[a:a + batch] for x in xte])
                                  for a in range(0, meta["n_test"], batch)]).numpy()

        perms = torch.from_numpy(d["perm"].astype(np.int64))
        slice_idx = lambda e, s: perms[e, s * batch:(s + 1) * batch]

    total = 0.0
    first_step_s = None
    for e in range(meta["epochs"]):
        t0 = time.perf_counter()
        for s in range(steps):
            if first_step_s is None:
                t1 = time.perf_counter()
                train_step(slice_idx(e, s))
                first_step_s = time.perf_counter() - t1
            else:
                train_step(slice_idx(e, s))
        dt = time.perf_counter() - t0
        total += dt
        final = common.metrics(loss, predict(), d["y_test"])   # untimed
        emit({"mode": "loop_epoch", "impl": impl, "task": task.name,
              "epoch": e + 1, "train_s": dt, **final})
    emit({"mode": "loop", "impl": impl, "task": task.name, "total_s": total,
          "first_step_s": first_step_s, "steps": steps * meta["epochs"], **final})


def main():
    mode, impl, name = sys.argv[1], sys.argv[2], sys.argv[3]
    batches = [int(b) for b in sys.argv[4:]]
    if mode == "infer":
        run_infer(impl, ARCH_BY_NAME[name], batches)
    elif mode == "train":
        run_train(impl, ARCH_BY_NAME[name], batches)
    elif mode == "loop":
        run_loop(impl, TASK_BY_NAME[name])
    else:
        raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    main()
