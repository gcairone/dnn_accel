"""Single source of truth for the benchmark suite: every architecture, batch
size and training task, shared by all four implementations (C++ core,
fastmlp Python, torch eager, torch.compile). Nothing here runs anything.

An architecture is a tree of nodes, mirroring fm.MLP(inputs=[...]): each node
has its own Linear stack (`sizes`, one activation per layer) and optional
children whose outputs are concatenated feature-wise into its input. Leaves
are fed in depth-first order -- the same order fastmlp, the C++ core and the
torch model (models.py) all use.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Node:
    sizes: List[int]
    acts: List[str]
    inputs: List["Node"] = field(default_factory=list)


def mlp(sizes, act, out=None, inputs=()):
    """One node: `act` on every layer, `out` (if given) on the last one."""
    acts = [act] * (len(sizes) - 1)
    if out is not None:
        acts[-1] = out
    return Node(list(sizes), acts, list(inputs))


@dataclass
class Arch:
    name: str
    root: Node
    loss: str          # "ce" (logits, int labels) | "mse" | "bce" (sigmoid output)
    note: str          # what kind of real model this stands in for


def nodes_postorder(root):
    """Children before parents, left to right; the root is last."""
    out = []

    def visit(n):
        for c in n.inputs:
            visit(c)
        out.append(n)

    visit(root)
    return out


def leaves(root):
    if not root.inputs:
        return [root]
    return [l for c in root.inputs for l in leaves(c)]


def in_features(root):
    """Total raw input width = sum over leaves, in depth-first order."""
    return sum(l.sizes[0] for l in leaves(root))


def out_features(root):
    return root.sizes[-1]


def num_params(root):
    return sum(n.sizes[i] * n.sizes[i + 1] + n.sizes[i + 1]
               for n in nodes_postorder(root) for i in range(len(n.sizes) - 1))


def seed_of(node_index):
    """Per-node init seed (post-order index). The model is built once by
    fastmlp with these seeds and saved as .mlpf; every implementation (torch
    included) loads that file, so all four start from identical weights."""
    return 1000 + 7 * node_index


# --- architectures -----------------------------------------------------------
# A realistic spread: tabular, vision-as-MLP, RL actor/critic, recommender
# towers, a transformer FFN block, coordinate MLPs (NeRF / PINN), an
# autoencoder. ~1k to ~2M parameters, 2 to 9 layers deep, flat and tree.

ARCHS = [
    Arch("tabular_s", mlp([16, 32, 16, 1], "relu", out="sigmoid"), "bce",
         "small tabular classifier (fraud/churn style)"),
    Arch("tabular_m", mlp([64, 256, 256, 128, 1], "relu"), "mse",
         "medium tabular regressor"),
    Arch("rl_policy", mlp([48, 64, 64, 8], "tanh", out="identity"), "mse",
         "PPO policy, small continuous-control task"),
    Arch("sac_actor", mlp([376, 256, 256, 17], "relu", out="tanh"), "mse",
         "SAC actor, Humanoid-sized observation/action"),
    Arch("sac_critic",
         mlp([256 + 64, 256, 256, 1], "relu", out="identity", inputs=[
             mlp([376, 256], "relu"),          # observation encoder
             mlp([17, 64], "relu"),            # action encoder
         ]), "mse", "SAC Q-network: obs and action encoders merged (tree)"),
    Arch("two_tower",
         mlp([128 + 128, 128, 1], "relu", out="sigmoid", inputs=[
             mlp([64, 256, 128], "relu"),      # user tower
             mlp([64, 256, 128], "relu"),      # item tower
         ]), "bce", "two-tower recommender / CTR model (tree)"),
    Arch("deep_tree",
         mlp([16 + 16, 32, 1], "relu", out="identity", inputs=[
             mlp([16 + 16, 32, 16], "relu", inputs=[
                 mlp([16, 32, 16], "relu"),
                 mlp([16, 32, 16], "relu"),
             ]),
             mlp([24, 32, 16], "relu"),
         ]), "mse", "three-level tree (usages.md example 3)"),
    Arch("mnist_mlp", mlp([784, 256, 128, 10], "relu", out="identity"), "ce",
         "classic MNIST MLP"),
    Arch("mnist_wide", mlp([784, 1024, 1024, 10], "relu", out="identity"), "ce",
         "wide MNIST MLP, ~1.9M params"),
    Arch("autoencoder", mlp([784, 512, 128, 32, 128, 512, 784], "relu", out="sigmoid"), "mse",
         "MNIST autoencoder, 6 layers"),
    Arch("ffn_block", mlp([512, 2048, 512], "gelu", out="identity"), "mse",
         "transformer FFN block (d_model 512, 4x expansion, GELU)"),
    Arch("nerf", mlp([63] + [256] * 8 + [4], "relu", out="identity"), "mse",
         "NeRF-style coordinate MLP, 8x256"),
    Arch("pinn", mlp([3] + [64] * 6 + [1], "tanh", out="identity"), "mse",
         "physics-informed net, deep and narrow"),
]

ARCH_BY_NAME = {a.name: a for a in ARCHS}

INFER_BATCHES = [1, 16, 128, 1024]
TRAIN_BATCHES = [64, 256]

# A smaller set for a quick sanity run (--quick).
QUICK_ARCHS = ["tabular_s", "sac_critic", "mnist_mlp", "ffn_block"]


# --- full training loops -----------------------------------------------------

@dataclass
class Task:
    name: str
    arch: Arch
    dataset: str       # key into prepare_data.DATASETS
    batch: int
    epochs: int
    optimizer: str     # "adam" | "sgd"
    lr: float
    momentum: float = 0.0
    note: str = ""


TASKS = [
    Task("mnist_classifier", ARCH_BY_NAME["mnist_mlp"], "mnist", batch=128, epochs=5,
         optimizer="adam", lr=1e-3, note="real MNIST, softmax cross-entropy"),
    Task("mnist_autoencoder", ARCH_BY_NAME["autoencoder"], "mnist_ae", batch=256, epochs=3,
         optimizer="adam", lr=1e-3, note="real MNIST, reconstruction MSE"),
    Task("tabular_regression",
         Arch("tabular_reg", mlp([32, 128, 128, 64, 1], "silu", out="identity"), "mse",
              "SiLU tabular regressor"),
         "tabular", batch=256, epochs=10, optimizer="adam", lr=1e-3,
         note="synthetic nonlinear regression, 100k rows"),
    Task("two_tower_ctr", ARCH_BY_NAME["two_tower"], "ctr", batch=512, epochs=5,
         optimizer="adam", lr=1e-3, note="synthetic click-through data, 200k rows"),
    Task("spiral_small",
         Arch("spiral", mlp([2, 64, 64, 3], "tanh", out="identity"), "ce", "tiny classifier"),
         "spiral", batch=32, epochs=20, optimizer="sgd", lr=5e-2, momentum=0.9,
         note="3-class spiral, 10k points: per-step overhead dominates"),
]

TASK_BY_NAME = {t.name: t for t in TASKS}

IMPLS = ["cpp", "fastmlp", "torch", "torch_compile"]
