"""Turns a cases.Node tree into each implementation's model.

The .mlpf file is the hand-off: fastmlp builds the tree once (seeded) and
saves it, the C++ runner and fastmlp's Python API load it back, and the torch
model copies its weights out of the same file. So every implementation starts
from bit-identical weights -- which is what lets the full-training mode
compare learning curves, not just wall time.
"""

import json
import os

import numpy as np

from cases import nodes_postorder, seed_of
from common import MODELS_DIR


# --- fastmlp ----------------------------------------------------------------------

def build_fastmlp(root, max_batch):
    import fastmlp as fm
    order = {id(n): i for i, n in enumerate(nodes_postorder(root))}

    def build(node):
        kids = [build(c) for c in node.inputs]
        return fm.MLP(node.sizes, activation=list(node.acts), max_batch=max_batch,
                      inputs=kids, seed=seed_of(order[id(node)]))

    return build(root)


def model_path(name, root):
    """The canonical initial weights for an architecture, created on first use."""
    path = os.path.join(MODELS_DIR, f"{name}.mlpf")
    if not os.path.exists(path):
        os.makedirs(MODELS_DIR, exist_ok=True)
        build_fastmlp(root, max_batch=1).save(path)
    return path


def read_mlpf_blob(path):
    """The flat float32 weight blob after the one-line JSON header."""
    with open(path, "rb") as f:
        header = f.readline()
        if not header.startswith(b"MLPF1"):
            raise ValueError(f"{path}: not an MLPF1 file")
        json.loads(header[len(b"MLPF1"):])
        return np.frombuffer(f.read(), dtype=np.float32)


# --- torch --------------------------------------------------------------------------

def build_torch(root, mlpf_path):
    import torch
    import torch.nn.functional as F

    act_fns = {
        "identity": None,
        "relu": F.relu,
        "sigmoid": torch.sigmoid,
        "tanh": torch.tanh,
        "leaky_relu": F.leaky_relu,            # slope 0.01, same as fastmlp
        "elu": F.elu,
        "silu": F.silu,
        "gelu": F.gelu,
        "gelu_tanh": lambda x: F.gelu(x, approximate="tanh"),
        "softplus": F.softplus,
    }

    class TreeNode(torch.nn.Module):
        """Plain idiomatic torch: children's outputs torch.cat'ed feature-wise,
        then a Linear stack -- what someone would write by hand for this tree."""

        def __init__(self, node):
            super().__init__()
            self.kids = torch.nn.ModuleList([TreeNode(c) for c in node.inputs])
            self.kid_leaves = [len(_leaves(c)) for c in node.inputs]
            self.linears = torch.nn.ModuleList(
                [torch.nn.Linear(node.sizes[i], node.sizes[i + 1]) for i in range(len(node.sizes) - 1)])
            self.acts = [act_fns[a] for a in node.acts]

        def forward(self, *xs):
            if self.kids:
                outs, k = [], 0
                for kid, n in zip(self.kids, self.kid_leaves):
                    outs.append(kid(*xs[k:k + n]))
                    k += n
                h = torch.cat(outs, dim=1)
            else:
                h = xs[0]
            for lin, act in zip(self.linears, self.acts):
                h = lin(h)
                if act is not None:
                    h = act(h)
            return h

    model = TreeNode(root)

    # .mlpf blob order: a node's own layers (W row-major [out, in], then b),
    # then each child's blob in order -- i.e. pre-order. torch's Linear.weight
    # is [out, in] too, so it is a straight copy.
    blob = read_mlpf_blob(mlpf_path)
    pos = 0

    def fill(tnode):
        nonlocal pos
        for lin in tnode.linears:
            for p in (lin.weight, lin.bias):
                n = p.numel()
                p.data.copy_(torch.from_numpy(blob[pos:pos + n].copy()).view_as(p))
                pos += n
        for kid in tnode.kids:
            fill(kid)

    fill(model)
    if pos != blob.size:
        raise RuntimeError(f"{mlpf_path}: blob has {blob.size} floats, torch model used {pos}")
    return model


def _leaves(node):
    if not node.inputs:
        return [node]
    return [l for c in node.inputs for l in _leaves(c)]


def torch_loss(loss):
    """Each matches the gradient fastmlp's losses.py (and suite_cpp) uses, so
    the trajectories stay comparable: mse is summed over features and
    averaged over the batch, like fm.mse_grad -- not F.mse_loss's mean over
    every element, which would scale the gradient by 1/C."""
    import torch.nn.functional as F
    if loss == "ce":
        return lambda pred, y: F.cross_entropy(pred, y)
    if loss == "bce":
        return lambda pred, y: F.binary_cross_entropy(pred, y)
    return lambda pred, y: F.mse_loss(pred, y, reduction="sum") / pred.shape[0]


def fastmlp_loss_grad(loss):
    import fastmlp as fm
    return {"ce": fm.cross_entropy_grad, "bce": fm.bce_grad, "mse": fm.mse_grad}[loss]
