"""Benchmark suite driver: fastmlp C++ core vs fastmlp Python API vs PyTorch
eager vs torch.compile, on the cases in cases.py.

    python3 benchmarks/suite/run.py infer   # inference throughput, arch x batch
    python3 benchmarks/suite/run.py train   # training step throughput, arch x batch
    python3 benchmarks/suite/run.py loop    # full training runs (needs prepare_data.py)
    python3 benchmarks/suite/run.py all
options:
    --quick              a 4-architecture subset; writes results/quick/ instead
    --impls a,b          subset of cpp,fastmlp,torch,torch_compile
    --archs a,b / --tasks a,b
    --rounds N           repeat every case N times, interleaved, keep the median (default 3)

Run inside the fastmlp-dev container (capped at 4 cores / 4 GB; every
measurement is still pinned to one core) after building build-bench/suite_cpp:
    cmake -S . -B build-bench -DFASTMLP_SIMD_BACKEND=AVX2 -DFASTMLP_BUILD_BENCHMARKS=ON \\
          -DFASTMLP_BUILD_PYTHON=OFF -DFASTMLP_BUILD_TESTS=OFF
    cmake --build build-bench -j4 --target suite_cpp

Cases run case-major, implementation-minor: all four implementations of one
case back to back, so drift in the machine's background load hits them alike
instead of biasing whichever implementation ran during a noisy minute. Each
case is repeated --rounds times in that interleaved order and the median
round kept: a single round measured two cells 20-40% off on this shared host
(re-measurement put both back at parity), which within-run medians cannot
catch when a load burst outlasts the whole ~0.5 s measurement.
"""

import argparse
import datetime
import json
import os
import platform
import subprocess
import sys
import time

import cases
import common
from common import CPP_BIN, RESULTS_DIR, SUITE_DIR

ENV = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
           TORCHINDUCTOR_COMPILE_THREADS="1", PYTHONUNBUFFERED="1")


def run_proc(cmd, timeout=3600):
    p = subprocess.run(cmd, capture_output=True, text=True, env=ENV, timeout=timeout, cwd=SUITE_DIR)
    recs = [json.loads(l[len("RESULT "):]) for l in p.stdout.splitlines() if l.startswith("RESULT ")]
    if p.returncode != 0:
        tail = "\n".join((p.stderr or p.stdout).splitlines()[-15:])
        print(f"  !! {' '.join(cmd[:4])} failed (exit {p.returncode}):\n{tail}", flush=True)
    return recs


def worker_cmd(mode, impl, name, extra):
    return [sys.executable, os.path.join(SUITE_DIR, "worker.py"), mode, impl, name] + [str(e) for e in extra]


def cpp_cmd(mode, arch_or_task, batches=()):
    if mode == "loop":
        spec = os.path.join(common.DATA_DIR, "tasks", arch_or_task.name, "spec.txt")
        return [CPP_BIN, "loop", arch_or_task.name, spec]
    from models import model_path
    path = model_path(arch_or_task.name, arch_or_task.root)
    if mode == "infer":
        return [CPP_BIN, "infer", arch_or_task.name, path] + [str(b) for b in batches]
    return [CPP_BIN, "train", arch_or_task.name, path, arch_or_task.loss] + [str(b) for b in batches]


def fmt_rate(r):
    return f"{r / 1e6:.2f}M" if r >= 1e6 else f"{r / 1e3:.1f}k" if r >= 1e3 else f"{r:.0f}"


def median_round(recs, key):
    """The record whose `key` is the median across rounds, annotated with every
    round's value so the report can show the spread."""
    recs = sorted(recs, key=lambda r: r[key])
    pick = dict(recs[len(recs) // 2])
    pick[key + "_rounds"] = [r[key] for r in recs]
    return pick


def run_bench(mode, archs, impls, batches, rounds):
    from models import model_path
    records = []
    for arch in archs:
        model_path(arch.name, arch.root)
        print(f"[{mode}] {arch.name}  ({cases.num_params(arch.root):,} params)", flush=True)
        got = {}
        for rnd in range(rounds):
            for impl in impls:
                cmd = cpp_cmd(mode, arch, batches) if impl == "cpp" else worker_cmd(mode, impl, arch.name, batches)
                for r in run_proc(cmd):
                    r["params"] = cases.num_params(arch.root)
                    got.setdefault((impl, r["batch"]), []).append(r)
                    rate = r["batch"] / r["sec"] if mode == "infer" else 1 / r["sec"]
                    unit = "samples/s" if mode == "infer" else "steps/s"
                    extra = f"  (compile {r['compile_s']:.1f}s)" if r.get("compile_s") else ""
                    print(f"    r{rnd + 1} {impl:14s} B={r['batch']:<5d} {r['sec'] * 1e6:11.1f} us  "
                          f"{fmt_rate(rate):>8s} {unit}{extra}", flush=True)
        records += [median_round(v, "sec") for v in got.values()]
    return records


def run_loop(tasks, impls, rounds):
    records = []
    for task in tasks:
        meta = os.path.join(common.DATA_DIR, "tasks", task.name, "meta.json")
        if not os.path.exists(meta):
            sys.exit(f"{meta} missing: run `python3 benchmarks/suite/prepare_data.py` first")
        print(f"[loop] {task.name}  ({task.note})", flush=True)
        got = {}
        for rnd in range(rounds):
            for impl in impls:
                cmd = cpp_cmd("loop", task) if impl == "cpp" else worker_cmd("loop", impl, task.name, [])
                recs = run_proc(cmd)
                final = [r for r in recs if r["mode"] == "loop"]
                if not final:
                    continue
                r = final[0]
                r["epochs"] = [e for e in recs if e["mode"] == "loop_epoch"]
                got.setdefault(impl, []).append(r)
                acc = f"  acc {r['test_acc']:.4f}" if "test_acc" in r else ""
                print(f"    r{rnd + 1} {impl:14s} {r['total_s']:8.2f} s total  first step {r['first_step_s']:.3f} s"
                      f"  test_loss {r['test_loss']:.4f}{acc}", flush=True)
        for v in got.values():
            pick = median_round(v, "total_s")
            records += pick.pop("epochs") + [pick]
    return records


def environment():
    cpu = "?"
    try:
        with open("/proc/cpuinfo") as f:
            cpu = next(l.split(":", 1)[1].strip() for l in f if l.startswith("model name"))
    except (OSError, StopIteration):
        pass
    env = {"date": datetime.datetime.now().isoformat(timespec="seconds"), "cpu": cpu,
           "python": platform.python_version(), "pinned_core": common.CORE,
           "allowed_cores": sorted(os.sched_getaffinity(0))}
    try:
        import numpy
        env["numpy"] = numpy.__version__
        out = subprocess.run([sys.executable, "-c", "import torch; print(torch.__version__)"],
                             capture_output=True, text=True, env=ENV)
        env["torch"] = out.stdout.strip()
        env["git"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                    text=True, cwd=SUITE_DIR).stdout.strip()
    except OSError:
        pass
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["infer", "train", "loop", "all"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--impls", default=",".join(cases.IMPLS))
    ap.add_argument("--archs")
    ap.add_argument("--tasks")
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()

    impls = args.impls.split(",")
    unknown = set(impls) - set(cases.IMPLS)
    if unknown:
        sys.exit(f"unknown impls {unknown}; choose from {cases.IMPLS}")
    if "cpp" in impls and not os.path.exists(CPP_BIN):
        sys.exit(f"{CPP_BIN} not built (see this file's docstring), or drop it with --impls")

    arch_names = args.archs.split(",") if args.archs else (cases.QUICK_ARCHS if args.quick else None)
    archs = [cases.ARCH_BY_NAME[n] for n in arch_names] if arch_names else cases.ARCHS
    tasks = [cases.TASK_BY_NAME[n] for n in args.tasks.split(",")] if args.tasks else cases.TASKS

    out_dir = os.path.join(RESULTS_DIR, "quick") if args.quick else RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    modes = ["infer", "train", "loop"] if args.mode == "all" else [args.mode]
    env = environment()
    print(json.dumps(env), flush=True)
    for mode in modes:
        t0 = time.time()
        if mode == "loop":
            recs = run_loop(tasks[:2] if args.quick else tasks, impls, args.rounds)
        else:
            batches = cases.INFER_BATCHES if mode == "infer" else cases.TRAIN_BATCHES
            recs = run_bench(mode, archs, impls, batches, args.rounds)
        with open(os.path.join(out_dir, f"{mode}.json"), "w") as f:
            json.dump({"env": env, "impls": impls, "rounds": args.rounds, "records": recs}, f, indent=1)
        print(f"[{mode}] done in {time.time() - t0:.0f} s -> {out_dir}/{mode}.json", flush=True)

    import report
    path = report.write(out_dir)
    print(f"report: {path}")


if __name__ == "__main__":
    main()
