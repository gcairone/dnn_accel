"""Checks the suite compares like with like before it times anything: for
every architecture, the torch model built from the .mlpf file produces the
same outputs as fastmlp's (forward), and one SGD step moves every weight the
same way -- i.e. loss gradient + backward agree. (SGD, not Adam: Adam's first
step is ~lr*sign(g), which turns float32 noise on near-zero gradients into
full-size differences; Adam itself is checked against closed form in
tests/cpp/test_optimizer.cpp.) A mismatch here means
the timing tables would be comparing different computations.

    python3 benchmarks/suite/verify.py
"""

import os

import numpy as np
import torch

import common
from cases import ARCHS, leaves
from models import build_torch, fastmlp_loss_grad, model_path, read_mlpf_blob, torch_loss

import fastmlp as fm

torch.set_num_threads(1)
common.pin()


def main():
    rng = np.random.default_rng(7)
    bad = 0
    for arch in ARCHS:
        path = model_path(arch.name, arch.root)
        batch = 16
        xs = [rng.standard_normal((batch, l.sizes[0]), dtype=np.float32) for l in leaves(arch.root)]
        out = arch.root.sizes[-1]
        if arch.loss == "ce":
            y = rng.integers(0, out, batch).astype(np.int64)
        elif arch.loss == "bce":
            y = (rng.random((batch, out)) > 0.5).astype(np.float32)
        else:
            y = rng.standard_normal((batch, out), dtype=np.float32)

        m = fm.MLP.load(path, max_batch=batch)
        t = build_torch(arch.root, path)
        pf = np.array(m(*xs))
        pt = t(*[torch.from_numpy(x) for x in xs]).detach().numpy()
        fwd_err = np.abs(pf - pt).max() / (np.abs(pt).max() + 1e-6)

        # One SGD step each (lr 1: the step is -grad), then compare every weight.
        opt_f = fm.SGD(m, lr=1.0)
        m.backward(fastmlp_loss_grad(arch.loss)(np.array(m(*xs)), y))
        opt_f.step()
        tmp = path + ".verify"
        m.save(tmp)
        wf = read_mlpf_blob(tmp)
        os.remove(tmp)
        opt_t = torch.optim.SGD(t.parameters(), lr=1.0)
        torch_loss(arch.loss)(t(*[torch.from_numpy(x) for x in xs]), torch.from_numpy(y)).backward()
        opt_t.step()
        w0 = read_mlpf_blob(path)
        # Walk torch parameters in the same pre-order as the blob.
        wt = []

        def walk(n):
            for lin in n.linears:
                wt.append(lin.weight.detach().numpy().ravel())
                wt.append(lin.bias.detach().numpy().ravel())
            for k in n.kids:
                walk(k)

        walk(t)
        wt = np.concatenate(wt)
        step_f, step_t = wf - w0, wt - w0
        step_err = np.abs(step_f - step_t).max() / (np.abs(step_t).max() + 1e-12)
        ok = fwd_err < 1e-4 and step_err < 1e-4
        bad += not ok
        print(f"{'ok ' if ok else 'BAD'} {arch.name:12s} forward rel err {fwd_err:.1e}   "
              f"gradient rel err {step_err:.1e}", flush=True)
    if bad:
        raise SystemExit(f"{bad} architecture(s) disagree")


if __name__ == "__main__":
    main()
