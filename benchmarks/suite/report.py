"""results/{infer,train,loop}.json -> results/results.md. Re-runnable on its
own (`python3 benchmarks/suite/report.py [results_dir]`) to regenerate the
tables without re-measuring anything."""

import json
import math
import os
import sys

import cases
from common import RESULTS_DIR

NAMES = {"cpp": "C++", "fastmlp": "Python", "torch": "torch", "torch_compile": "compile"}


def load(out_dir, mode):
    p = os.path.join(out_dir, f"{mode}.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def rate(r, mode):
    return r["batch"] / r["sec"] if mode == "infer" else 1.0 / r["sec"]


def fmt(v):
    if v >= 1e6:
        return f"{v / 1e6:.2f}M"
    if v >= 1e4:
        return f"{v / 1e3:.0f}k"
    if v >= 1e3:
        return f"{v / 1e3:.1f}k"
    return f"{v:.1f}" if v < 100 else f"{v:.0f}"


def gmean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def ratio_cell(a, b):
    if a is None or b is None:
        return "–"
    r = a / b
    s = f"{r:.2f}x"
    return f"**{s}**" if r < 1 else s   # bold = fastmlp loses


def bench_section(data, mode):
    """One table per batch size: throughput of each implementation, then the
    ratios that answer the actual question."""
    unit = "samples/s" if mode == "infer" else "steps/s"
    recs = data["records"]
    by = {(r["arch"], r["batch"], r["impl"]): r for r in recs}
    batches = sorted({r["batch"] for r in recs})
    archs = [a for a in cases.ARCHS if any(r["arch"] == a.name for r in recs)]
    lines = []
    ratios = {k: [] for k in ("py_torch", "py_comp", "cpp_torch", "cpp_comp")}
    wins = {k: [0, 0] for k in ratios}
    for b in batches:
        lines.append(f"\n#### batch {b} — {unit} (higher is better)\n")
        lines.append("| arch | params | C++ | Python | torch | compile | Python/torch | Python/compile "
                     "| C++/torch | C++/compile | compile s |")
        lines.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
        for a in archs:
            g = {i: by.get((a.name, b, i)) for i in cases.IMPLS}
            v = {i: rate(g[i], mode) if g[i] else None for i in g}
            for key, (x, y) in {"py_torch": ("fastmlp", "torch"), "py_comp": ("fastmlp", "torch_compile"),
                                "cpp_torch": ("cpp", "torch"), "cpp_comp": ("cpp", "torch_compile")}.items():
                if v[x] and v[y]:
                    ratios[key].append(v[x] / v[y])
                    wins[key][0] += v[x] >= v[y]
                    wins[key][1] += 1
            comp_s = g["torch_compile"]["compile_s"] if g["torch_compile"] else None
            cells = [a.name, f"{cases.num_params(a.root):,}"]
            cells += [fmt(v[i]) if v[i] else "–" for i in cases.IMPLS]
            cells += [ratio_cell(v["fastmlp"], v["torch"]), ratio_cell(v["fastmlp"], v["torch_compile"]),
                      ratio_cell(v["cpp"], v["torch"]), ratio_cell(v["cpp"], v["torch_compile"]),
                      f"{comp_s:.1f}" if comp_s else "–"]
            lines.append("| " + " | ".join(cells) + " |")
    return lines, ratios, wins


def loop_section(data):
    recs = data["records"]
    finals = {(r["task"], r["impl"]): r for r in recs if r["mode"] == "loop"}
    epochs = {}
    for r in recs:
        if r["mode"] == "loop_epoch":
            epochs.setdefault((r["task"], r["impl"]), []).append(r)
    lines = []
    ratios = {"py_torch": [], "py_comp": [], "cpp_torch": [], "cpp_comp": []}
    for task in cases.TASKS:
        rows = {i: finals.get((task.name, i)) for i in cases.IMPLS}
        if not any(rows.values()):
            continue
        meta_steps = next(r["steps"] for r in rows.values() if r)
        lines.append(f"\n#### {task.name} — {task.note}\n")
        lines.append(f"`{task.arch.name}`, {cases.num_params(task.arch.root):,} params, batch {task.batch}, "
                     f"{task.epochs} epochs = {meta_steps:,} steps, {task.optimizer} lr {task.lr}"
                     + (f" momentum {task.momentum}" if task.momentum else "") + "\n")
        has_acc = any(r and "test_acc" in r for r in rows.values())
        lines.append("| impl | total train s | first step s | rest, us/step | test loss "
                     + ("| test acc " if has_acc else "") + "| vs torch | vs compile |")
        lines.append("|---|--:|--:|--:|--:|" + ("--:|" if has_acc else "") + "--:|--:|")
        t = {i: rows[i]["total_s"] if rows[i] else None for i in rows}
        for i in cases.IMPLS:
            r = rows[i]
            if not r:
                continue
            steady = (r["total_s"] - r["first_step_s"]) / max(1, r["steps"] - 1) * 1e6
            cells = [NAMES[i], f"{r['total_s']:.2f}", f"{r['first_step_s']:.3f}", f"{steady:.1f}",
                     f"{r['test_loss']:.4f}"]
            if has_acc:
                cells.append(f"{r['test_acc']:.4f}" if "test_acc" in r else "–")
            if i in ("cpp", "fastmlp"):
                cells.append(ratio_cell(t["torch"], r["total_s"]))
                cells.append(ratio_cell(t["torch_compile"], r["total_s"]))
            else:  # torch rows: plain reference numbers, nothing to flag
                other = "torch_compile" if i == "torch" else "torch"
                rel = f"{t[other] / r['total_s']:.2f}x" if t[other] else "–"
                cells += ["1.00x", rel] if i == "torch" else [rel, "1.00x"]
            lines.append("| " + " | ".join(cells) + " |")
        for key, (x, y) in {"py_torch": ("fastmlp", "torch"), "py_comp": ("fastmlp", "torch_compile"),
                            "cpp_torch": ("cpp", "torch"), "cpp_comp": ("cpp", "torch_compile")}.items():
            if t[x] and t[y]:
                ratios[key].append(t[y] / t[x])
        # Learning curves: same init, same data, same batch order -> these
        # should track each other; a diverging column is a correctness bug.
        metric = "test_acc" if has_acc else "test_loss"
        present = [i for i in cases.IMPLS if (task.name, i) in epochs]
        lines.append(f"\n<details><summary>per-epoch {metric}</summary>\n")
        lines.append("| epoch | " + " | ".join(NAMES[i] for i in present) + " |")
        lines.append("|--:|" + "--:|" * len(present))
        for e in range(task.epochs):
            vals = []
            for i in present:
                ep = epochs[(task.name, i)]
                vals.append(f"{ep[e][metric]:.4f}" if e < len(ep) else "–")
            lines.append(f"| {e + 1} | " + " | ".join(vals) + " |")
        lines.append("\n</details>")
    return lines, ratios


def summary_line(label, ratios, wins=None):
    def cell(k):
        g = gmean(ratios[k])
        w = f" ({wins[k][0]}/{wins[k][1]})" if wins else ""
        return f"{g:.2f}x{w}" if ratios[k] else "–"
    return f"| {label} | {cell('py_torch')} | {cell('py_comp')} | {cell('cpp_torch')} | {cell('cpp_comp')} |"


def write(out_dir=RESULTS_DIR):
    infer, train, loop = load(out_dir, "infer"), load(out_dir, "train"), load(out_dir, "loop")
    env = next(d["env"] for d in (infer, train, loop) if d)
    body, summary = [], []
    if infer:
        lines, r, w = bench_section(infer, "infer")
        summary.append(summary_line("inference throughput", r, w))
        body += ["\n## 1. Inference throughput\n",
                 "Forward pass only. fastmlp `inference_only=True`; torch `eval()` + `inference_mode()`; "
                 "compile = `torch.compile(model, dynamic=False)` under `no_grad()`, compiled once per batch "
                 "size (the `compile s` column: a cold compile, paid once, excluded from the throughput)."] + lines
    if train:
        lines, r, w = bench_section(train, "train")
        summary.append(summary_line("training step throughput", r, w))
        body += ["\n## 2. Training step throughput\n",
                 "One full step on a fixed batch: forward, loss gradient, backward, Adam (lr 1e-3), zero_grad. "
                 "torch.compile compiles the whole step function (forward + loss + backward + optimizer)."] + lines
    if loop:
        lines, r = loop_section(loop)
        summary.append(summary_line("full training run (total time)", r))
        body += ["\n## 3. Full training runs\n",
                 "Real training loops on real/synthetic datasets: every step gathers a shuffled batch "
                 "(`X[idx]`), all four implementations start from the same `.mlpf` weights and see the "
                 "same batch order, so their learning curves should match. `total train s` is the training "
                 "loop only (evaluation after each epoch is untimed) and includes torch.compile's "
                 "compilation, which lands in the first step: a cold compile, with an empty inductor cache "
                 "(a warm on-disk cache from an earlier run brings it to ~1 s)."] + lines

    head = [
        "# fastmlp benchmark suite — results\n",
        "Generated by `benchmarks/suite/report.py` from the JSON next to it; do not edit by hand "
        "(the commentary comes from `notes.md`).\n",
        f"- CPU: {env.get('cpu')}, one core pinned (core {env.get('pinned_core')}), "
        f"container cores {env.get('allowed_cores')}, 4 GB",
        f"- torch {env.get('torch')}, numpy {env.get('numpy')}, Python {env.get('python')}, "
        f"fastmlp @ {env.get('git')}, {env.get('date')}",
        "- every implementation single-threaded; each throughput is the median of 7 samples of ~50 ms, "
        f"and each cell the median of {max(d.get('rounds', 1) for d in (infer, train, loop) if d)} "
        "rounds with the four implementations interleaved",
        "- C++ = fastmlp core called directly; Python = the fastmlp Python API; "
        "torch = eager PyTorch; compile = torch.compile",
        "- ratios are fastmlp ÷ torch throughput (or torch ÷ fastmlp time): >1 means fastmlp is faster; "
        "**bold** marks a loss\n",
        "## Summary (geometric mean of the ratios, wins/cases)\n",
        "| | Python vs torch | Python vs compile | C++ vs torch | C++ vs compile |",
        "|---|--:|--:|--:|--:|",
    ] + summary
    notes = os.path.join(out_dir, "notes.md")
    if os.path.exists(notes):
        with open(notes) as f:
            head += ["", f.read().rstrip()]
    path = os.path.join(out_dir, "results.md")
    with open(path, "w") as f:
        f.write("\n".join(head + body) + "\n")
    return path


if __name__ == "__main__":
    print(write(sys.argv[1] if len(sys.argv) > 1 else RESULTS_DIR))
