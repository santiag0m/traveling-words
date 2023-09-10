import torch
import numpy as np
import matplotlib.pyplot as plt

from nanoGPT.model import GPT
from utils.model_utils import load_model
from utils.plot_utils import find_closest_to_square, plt_subplots_3d
from utils.umap_utils import UMAPWithLaplacianInit


def add_labels(ax: plt.Axes, x: np.ndarray, labels: list[str], **kwargs):
    t, d = x.shape
    assert t == len(labels), "Labels do not match the number of characters"
    assert d == 3, "Features should be 3 dimensional"

    for i, token in enumerate(labels):
        ax.text(x[i, 0], x[i, 1], x[i, 2], token, **kwargs)


def plot_umap_per_layer(
    inputs: torch.Tensor,
    model: GPT,
    umap: UMAPWithLaplacianInit,
    labels: list[str],
    num_layers_to_plot: int = 6,
):
    b, t = inputs.size()
    assert b == 1, "Batch has more than one example"
    assert t == len(labels), "Labels do not match the number of characters"

    # labels = [label + f"_{pos}" for pos, label in enumerate(labels)]

    # All layers + (emb + pe) + norm(emb + pe)

    n_plots = num_layers_to_plot + 2

    n_layers = len(model.transformer.h)
    first_half = int(num_layers_to_plot / 2)
    second_half = num_layers_to_plot - first_half
    layers_to_plot = list(range(first_half)) + list(
        range(n_layers - second_half, n_layers)
    )

    rows, cols = find_closest_to_square(n_plots)
    f, axs = plt_subplots_3d(
        nrows=rows, ncols=cols, sharex=True, sharey=True, sharez=True
    )
    axs = axs.ravel()
    f.suptitle("Transformer Layers")

    final_f, final_axs = plt_subplots_3d(nrows=1, ncols=1)
    final_axs = final_axs[0][0]

    with torch.no_grad():
        # Prepare inputs
        emb = model.transformer.wte(inputs)
        emb_3d = umap.transform(emb.clone().cpu().numpy()[0, ...])

        pos = torch.arange(0, t, dtype=torch.long, device=inputs.device)  # shape (t)
        pos_emb = model.transformer.wpe(pos)
        x = emb + pos_emb

        final_ln = model.transformer.ln_f

        # Plot original embeddings and pos embedding
        x_in = x.clone().cpu().numpy()[0, ...]
        x_in = umap.transform(x_in)

        axs[0].scatter(
            emb_3d[:, 0], emb_3d[:, 1], emb_3d[:, 2], c="blue", label="Word Embeddings"
        )
        axs[0].scatter(
            x_in[:, 0], x_in[:, 1], x_in[:, 2], c="purple", label="+ Pos. Embeddings"
        )
        add_labels(axs[0], emb_3d, labels=labels, color="blue")
        add_labels(axs[0], x_in, labels=labels, color="purple")
        axs[0].set_title("Input Layer")
        axs[0].legend()

        # Plot input with Layer Norm
        x_in = torch.clone(final_ln(x)).cpu().numpy()[0, ...]
        x_in = umap.transform(x_in)

        axs[1].scatter(
            emb_3d[:, 0], emb_3d[:, 1], emb_3d[:, 2], c="blue", label="Word Embeddings"
        )
        axs[1].scatter(
            x_in[:, 0],
            x_in[:, 1],
            x_in[:, 2],
            c="purple",
            label="+ Pos. Embeddings + LayerNorm",
        )
        add_labels(axs[1], emb_3d, labels=labels, color="blue")
        add_labels(axs[1], x_in, labels=labels, color="purple")
        axs[1].set_title("Input Layer (Normalized)")
        axs[1].legend()

        delta_history = []
        pos_history = []

        plot_idx = 2  # Start at third plot
        for i, block in enumerate(model.transformer.h):
            x = block(x)

            # Map every layer to the norm space used at the end
            x_out = umap.transform(final_ln(x).cpu().numpy()[0, ...])

            delta = x_out - x_in

            # Add trajectory for the second to last character
            pos_history.append(x_in[[-2], :])
            delta_history.append(delta[[-2], :])

            pos_ = np.concatenate(pos_history, axis=0)
            delta_ = np.concatenate(delta_history, axis=0)

            if i in layers_to_plot:
                axs[plot_idx].scatter(
                    emb_3d[:, 0],
                    emb_3d[:, 1],
                    emb_3d[:, 2],
                    c="blue",
                    label="Word Embeddings",
                )
                axs[plot_idx].scatter(
                    x_out[:, 0],
                    x_out[:, 1],
                    x_out[:, 2],
                    c="purple",
                    label="Latent Features",
                )

                axs[plot_idx].quiver(
                    pos_[:, 0],
                    pos_[:, 1],
                    pos_[:, 2],
                    delta_[:, 0],
                    delta_[:, 1],
                    delta_[:, 2],
                    color="red",
                    arrow_length_ratio=0,
                )

                add_labels(axs[plot_idx], emb_3d, labels=labels, color="blue")
                add_labels(axs[plot_idx], x_out, labels=labels, color="purple")
                axs[plot_idx].set_title(f"Layer {i}")
                axs[plot_idx].legend()

                plot_idx += 1

            x_in = x_out

        # Plot final result in separate figure
        final_axs.scatter(
            emb_3d[:, 0],
            emb_3d[:, 1],
            emb_3d[:, 2],
            c="blue",
            label="Word Embeddings",
        )
        final_axs.scatter(
            x_out[-2, 0],
            x_out[-2, 1],
            x_out[-2, 2],
            c="purple",
            label="Latent Features",
        )

        final_axs.quiver(
            pos_[:, 0],
            pos_[:, 1],
            pos_[:, 2],
            delta_[:, 0],
            delta_[:, 1],
            delta_[:, 2],
            color="red",
            arrow_length_ratio=0,
        )

        add_labels(final_axs, emb_3d, labels=labels, color="blue", fontsize=12)
        add_labels(
            final_axs, x_out[[-2], :], labels=[labels[-2]], color="purple", fontsize=12
        )
        final_axs.set_title("Word trajectory")
        final_axs.legend()

    for ax in [*axs, final_axs]:
        ax.xaxis.set_ticklabels([])
        ax.xaxis.set_ticks_position("none")
        ax.yaxis.set_ticklabels([])
        ax.yaxis.set_ticks_position("none")
        ax.zaxis.set_ticklabels([])
        ax.zaxis.set_ticks_position("none")

    return (f, axs), (final_f, final_axs)


if __name__ == "__main__":
    plt.ion()

    TEST_INPUT = "To kill two birds with one stone"

    model, encode, decode = load_model(init_from="gpt2")

    # Initialize UMAP
    umap = UMAPWithLaplacianInit(n_components=3)
    e = model.transformer.wte.weight.detach()

    # If e is too big, select first 10_000 points
    # (super important to have lots of points)
    e = e[:10000, :]
    _ = umap.fit(e.cpu().numpy())

    # Encode input
    inputs = encode(TEST_INPUT)
    inputs = torch.tensor([inputs]).to("cuda")

    labels = [decode([token]) for token in encode(TEST_INPUT)]
    print(labels)

    (f, axs), (final_f, final_axs) = plot_umap_per_layer(
        inputs=inputs,
        model=model,
        umap=umap,
        labels=labels,
    )
