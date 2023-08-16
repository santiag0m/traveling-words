import torch
import matplotlib.pyplot as plt

from model_utils import (
    load_model,
    get_attn_weights_from_block,
    split_weights_by_head,
    get_ln_from_block,
)
from plot_utils import find_closest_to_square, plt_subplots_3d
from umap_utils import UMAPWithLaplacianInit


def plot_umap_per_head(
    inputs: torch.Tensor,
    wq_heads: list[torch.Tensor],
    wk_heads: list[torch.Tensor],
    wv_heads: list[torch.Tensor],
    wo_heads: list[torch.Tensor],
    umap: UMAPWithLaplacianInit,
    labels: list = [],
) -> tuple[tuple[plt.Figure, plt.Axes]]:
    n_head = len(wq_heads)
    assert n_head == len(wk_heads), "Number of heads do not match (Q vs. K)"
    assert n_head == len(wv_heads), "Number of heads do not match (Q vs. V)"
    assert n_head == len(wo_heads), "Number of heads do not match (Q vs. O)"

    rows, cols = find_closest_to_square(n_head)
    f_qk, axs_qk = plt_subplots_3d(
        nrows=rows, ncols=cols, sharex=True, sharey=True, sharez=True
    )
    f_vo, axs_vo = plt_subplots_3d(
        nrows=rows, ncols=cols, sharex=True, sharey=True, sharez=True
    )
    f_qk.suptitle("QK interaction")
    f_vo.suptitle("VO gradients")
    axs_qk = axs_qk.ravel()
    axs_vo = axs_vo.ravel()

    with torch.no_grad():
        e_norm = ln_1(inputs)  # Vocab x Dims
        e_3d_norm = umap.transform(e_norm.cpu().numpy())  # Vocab x 3
        q = e_3d_norm  # Vocab x 3

        min_z = min(q[:, 2])
        max_z = max(q[:, 2])

        for i in range(model.config.n_head):
            wq = wq_heads[i]
            wk = wk_heads[i]
            wv = wv_heads[i]
            wo = wo_heads[i]

            k = (
                wq @ wk.T @ e_norm.T
            )  # (Dims x Dims_h) @ (Dims_h x Dims) @ (Dims x Vocab)
            k = k.T  # Vocab x Dims
            k = umap.transform(k.cpu().numpy())  # Vocab x 3

            v = e_norm @ (
                wv @ wo.T
            )  # (Vocab x Dims) @ (Dims x Dims_h) @ (Dims_h x Dims)
            v = umap.transform(v.cpu().numpy())  # Vocab x 3

            v = v - v.mean(axis=-1, keepdims=True)

            axs_qk[i].scatter(q[:, 0], q[:, 1], q[:, 2], c="red", label="query")
            axs_qk[i].scatter(k[:, 0], k[:, 1], k[:, 2], c="blue", label="key")

            for j, label in enumerate(labels):
                axs_qk[i].text(q[j, 0], q[j, 1], q[j, 2], label, color="red")
                axs_qk[i].text(k[j, 0], k[j, 1], k[j, 2], label, color="blue")

            axs_vo[i].scatter(q[:, 0], q[:, 1], q[:, 2], c="red", label="query")
            axs_vo[i].quiver(
                q[:, 0],
                q[:, 1],
                q[:, 2],
                v[:, 0],
                v[:, 1],
                v[:, 2],
                color="green",
                label="value",
                normalize=True,
            )

            for j, label in enumerate(labels):
                axs_vo[i].text(q[j, 0], q[j, 1], q[j, 2], label, color="red")

            axs_qk[i].legend()
            axs_vo[i].legend()

            axs_qk[i].set_title(f"Head {i}")
            axs_vo[i].set_title(f"Head {i}")

            min_z = min(min_z, min(k[:, 2]))
            max_z = max(max_z, max(k[:, 2]))

        for i in range(model.config.n_head):
            axs_qk[i].set_zlim(min_z, max_z)
            axs_vo[i].set_zlim(min_z, max_z)

        return (f_qk, axs_qk), (f_vo, axs_vo)


if __name__ == "__main__":
    plt.ion()

    model, encode, decode = load_model(out_dir="nanoGPT/out-shakespeare-char-no-pe")

    e = model.transformer.wte.weight.detach()

    umap = UMAPWithLaplacianInit(n_components=3)
    e_3d = umap.fit(e.cpu().numpy())

    ln_1, ln_2 = get_ln_from_block(model, block_idx=0)
    wq, wk, wv, wo = get_attn_weights_from_block(model, block_idx=0)

    wq_heads, wk_heads, wv_heads, wo_heads = split_weights_by_head(
        wq, wk, wv, wo, n_heads=model.config.n_head
    )

    with torch.no_grad():
        e_norm = ln_1(e)

    (f_qk, axs_qk), (f_vo, axs_vo) = plot_umap_per_head(
        inputs=e_norm,
        wq_heads=wq_heads,
        wk_heads=wq_heads,
        wv_heads=wq_heads,
        wo_heads=wq_heads,
        umap=umap,
        labels=[decode([i])[0] for i in range(model.config.vocab_size)],
    )
