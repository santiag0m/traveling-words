import torch
import matplotlib.pyplot as plt

from model_utils import load_model, get_attn_weights_from_block, split_weights_by_head
from plot_utils import find_closest_to_square, plt_subplots_3d
from umap_utils import UMAPWithLaplacianInit


def plot_umap_per_head(
    wq_heads: list[torch.Tensor],
    wk_heads: list[torch.Tensor],
    umap: UMAPWithLaplacianInit,
):
    n_head = len(wq_heads)
    assert n_head == len(wk_heads), "Number of heads do not match"

    rows, cols = find_closest_to_square(n_head)
    f, axs = plt.subplots(rows, cols)
    plt.subplot(rows, cols, subplot_kw={"projection": "3d"})


if __name__ == "__main__":
    plt.ion()

    model = load_model(out_dir="nanoGPT/out-shakespeare-char-no-pe")

    e = model.transformer.wte.weight.detach()

    umap = UMAPWithLaplacianInit(n_components=3)
    e_3d = umap.fit(e.cpu().numpy())

    f, axs = plt_subplots_3d(nrows=1, ncols=2)
    axs[0].scatter(e_3d[:, 0], e_3d[:, 1], e_3d[:, 2])

    # wq, wk, wv = get_attn_weights_from_block(model, block_idx=0)

    # wq = e @ wq
    # wk = e @ wk

    # wq_heads, wk_heads = split_weights_by_head(wq, wk, n_heads=model.config.n_head)
