import os

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes

from nanoGPT.model import GPTConfig, GPT


def load_model(init_from: str, out_dir: str, seed: int = 1337) -> GPT:
    device = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    # model
    if init_from == "resume":
        # init from a model saved in a specific directory
        ckpt_path = os.path.join(out_dir, "ckpt.pt")
        checkpoint = torch.load(ckpt_path, map_location=device)
        gptconf = GPTConfig(**checkpoint["model_args"])
        model = GPT(gptconf)
        state_dict = checkpoint["model"]
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
    elif init_from.startswith("gpt2"):
        # init from a given GPT-2 model
        model = GPT.from_pretrained(init_from, dict(dropout=0.0))

    model.eval()
    model.to(device)
    return model


def get_attn_weights_from_block(
    model: GPT, block_idx: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    block = model.transformer["h"][block_idx]
    weights = block.attn.c_attn.weight.detach()

    # Assume y = x @ W, instead of nn.linear
    wq, wk, wv = weights.T.split(model.config.n_embd, dim=1)
    return wq, wk, wv


def split_weights_by_head(*args, n_heads: int) -> list[list[torch.Tensor]]:
    results = [[] for _ in range(len(args))]

    for idx, weight in enumerate(args):
        d_in, d_out = weight.shape
        assert (
            d_out % n_heads == 0
        ), "Weight dimensions not consistent with the number of heads"
        results[idx] += weight.split(d_out // n_heads, dim=1)

    return results


def find_closest_to_square(area: int) -> tuple[int, int]:
    square = area**0.5
    rows = int(square)
    while True:
        if area % rows == 0:
            cols = area // rows
            return (rows, cols)
        rows += 1


def plot_head_interactions(
    wq_heads: list[torch.Tensor], wk_heads: list[torch.Tensor]
) -> tuple[Figure, Axes]:
    n_head = len(wq_heads)
    assert n_head == len(wk_heads), "Number of heads do not match"

    rows, cols = find_closest_to_square(n_head)
    f, axs = plt.subplots(rows, cols)

    if max(rows, cols) > 1:
        _axs = axs.ravel()
    else:
        _axs = [axs]

    for idx in range(n_head):
        wq = wq_heads[idx]
        wk = wk_heads[idx]
        scores = wq @ wk.T
        scores = scores.to("cpu").numpy()

        vmin = scores.min()
        vmax = scores.max()
        norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        _axs[idx].set_aspect("equal", "box")
        _axs[idx].imshow(scores, cmap="coolwarm", interpolation="nearest", norm=norm)
        _axs[idx].set_title(f"Head {idx}")

    f.suptitle("(E W_q) @ (E W_k)^T interactions")

    return f, axs


def plot_eigenvalues(wq_heads: list[torch.Tensor], wk_heads: list[torch.Tensor]):
    n_head = len(wq_heads)
    assert n_head == len(wk_heads), "Number of heads do not match"

    rows, cols = find_closest_to_square(n_head)
    f, axs = plt.subplots(rows, cols)

    if max(rows, cols) > 1:
        _axs = axs.ravel()
    else:
        _axs = [axs]

    for idx in range(n_head):
        wq = wq_heads[idx]
        wk = wk_heads[idx]

        norm_q = np.linalg.norm(wq.to("cpu").numpy(), axis=1, ord=2)
        norm_k = np.linalg.norm(wk.to("cpu").numpy(), axis=1, ord=2)
        eigenvalues = norm_q * norm_k

        idxs = np.argsort(-1 * eigenvalues)
        eigenvalues = eigenvalues[idxs]

        feature_percentile = np.linspace(0, 1, num=len(eigenvalues))
        eigenvalue_coverage = np.cumsum(eigenvalues) / sum(eigenvalues)
        index_80 = np.argmax(eigenvalue_coverage > 0.8)
        x_value = feature_percentile[index_80]
        y_value = eigenvalue_coverage[index_80]

        _axs[idx].plot(feature_percentile, eigenvalue_coverage)
        _axs[idx].set_aspect("equal", "box")
        _axs[idx].set_xlabel("% of features")
        _axs[idx].set_ylabel("Eigenvalue Coverage %")
        _axs[idx].grid()
        _axs[idx].set_title(f"Head {idx}")
        _axs[idx].axvline(x_value, color="red", linestyle="--")
        _axs[idx].axhline(y_value, color="red", linestyle="--")

    f.suptitle("Pseudo-Eigenvalue analysis")

    return f, axs


if __name__ == "__main__":
    plt.ion()

    model = load_model(init_from="resume", out_dir="nanoGPT/out-shakespeare-char-no-pe")

    e = model.transformer.wte.weight.detach()

    wq, wk, _ = get_attn_weights_from_block(model, block_idx=0)

    wq = e @ wq
    wk = e @ wk

    wq_heads, wk_heads = split_weights_by_head(wq, wk, n_heads=model.config.n_head)
    f, axs = plot_head_interactions(wq_heads=wq_heads, wk_heads=wk_heads)
    f2, axs2 = plot_eigenvalues(wq_heads=wq_heads, wk_heads=wk_heads)
