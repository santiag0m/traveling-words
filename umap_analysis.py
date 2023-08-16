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
    ln_1: torch.nn.LayerNorm,
    ln_2: torch.nn.LayerNorm,
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
        x_norm = ln_1(inputs)  # Vocab x Dims

        x_3d = umap.transform(inputs.cpu().numpy())
        x_norm_3d = umap.transform(x_norm.cpu().numpy())  # Vocab x 3

        min_qk_z = min(x_norm_3d[:, 2])
        max_qk_z = max(x_norm_3d[:, 2])

        min_vo_z = min(x_3d[:, 2])
        max_vo_z = min(x_3d[:, 2])

        for i in range(model.config.n_head):
            wq = wq_heads[i]
            wk = wk_heads[i]
            wv = wv_heads[i]
            wo = wo_heads[i]

            qk = (
                wq @ wk.T @ x_norm.T
            )  # (Dims x Dims_h) @ (Dims_h x Dims) @ (Dims x Vocab)
            qk = qk.T  # Vocab x Dims
            qk = umap.transform(qk.cpu().numpy())  # Vocab x 3

            gradient = x_norm @ (
                wv @ wo.T
            )  # (Vocab x Dims) @ (Dims x Dims_h) @ (Dims_h x Dims)

            # Add & Norm
            y = inputs + gradient
            y_norm = ln_2(y)

            y_norm_3d = umap.transform(y_norm.cpu().numpy())
            delta = (
                y_norm_3d - x_3d
            )  # compare to un-normalized inputs (bc weight tying at the end)

            axs_qk[i].scatter(
                x_norm_3d[:, 0],
                x_norm_3d[:, 1],
                x_norm_3d[:, 2],
                c="red",
                label="normalized_inputs",
            )
            axs_qk[i].scatter(
                qk[:, 0], qk[:, 1], qk[:, 2], c="blue", label="query_key_transform"
            )

            for j, label in enumerate(labels):
                axs_qk[i].text(
                    x_norm_3d[j, 0],
                    x_norm_3d[j, 1],
                    x_norm_3d[j, 2],
                    label,
                    color="red",
                )
                axs_qk[i].text(qk[j, 0], qk[j, 1], qk[j, 2], label, color="blue")

            axs_vo[i].scatter(
                x_3d[:, 0],
                x_3d[:, 1],
                x_3d[:, 2],
                c="red",
                label="original_inputs",
            )
            axs_vo[i].quiver(
                x_3d[:, 0],
                x_3d[:, 1],
                x_3d[:, 2],
                delta[:, 0],
                delta[:, 1],
                delta[:, 2],
                color="green",
                label="normalized_gradient",
                alpha=0.3,
            )
            axs_vo[i].scatter(
                y_norm_3d[:, 0],
                y_norm_3d[:, 1],
                y_norm_3d[:, 2],
                c="purple",
                label="normalized_outputs",
            )

            for j, label in enumerate(labels):
                axs_vo[i].text(
                    x_3d[j, 0],
                    x_3d[j, 1],
                    x_3d[j, 2],
                    label,
                    color="red",
                )
                axs_vo[i].text(
                    y_norm_3d[j, 0],
                    y_norm_3d[j, 1],
                    y_norm_3d[j, 2],
                    label,
                    color="purple",
                )

            axs_qk[i].legend()
            axs_vo[i].legend()

            axs_qk[i].set_title(f"Head {i}")
            axs_vo[i].set_title(f"Head {i}")

            min_qk_z = min(min_qk_z, min(qk[:, 2]))
            max_qk_z = max(max_qk_z, max(qk[:, 2]))

            min_vo_z = min(min_vo_z, min(y_norm_3d[:, 2]))
            max_vo_z = max(max_vo_z, max(y_norm_3d[:, 2]))

        for i in range(model.config.n_head):
            axs_qk[i].set_zlim(min_qk_z, max_qk_z)
            axs_vo[i].set_zlim(min_qk_z, max_qk_z)

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

    (f_qk, axs_qk), (f_vo, axs_vo) = plot_umap_per_head(
        inputs=e,
        ln_1=ln_1,
        ln_2=ln_2,
        wq_heads=wq_heads,
        wk_heads=wq_heads,
        wv_heads=wq_heads,
        wo_heads=wq_heads,
        umap=umap,
        labels=[decode([i])[0] for i in range(model.config.vocab_size)],
    )
