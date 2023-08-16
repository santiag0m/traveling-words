import torch
import numpy as np
import matplotlib.pyplot as plt

from nanoGPT.model import GPT
from model_utils import load_model
from plot_utils import find_closest_to_square, plt_subplots_3d
from umap_utils import UMAPWithLaplacianInit


def add_labels(ax: plt.Axes, x: np.ndarray, labels: list[str], **kwargs):
    t, d = x.shape
    assert t == len(labels), "Labels do not match the number of characters"
    assert d == 3, "Features should be 3 dimensional"

    for i, letter in enumerate(labels):
        ax.text(x[i, 0], x[i, 1], x[i, 2], letter, **kwargs)


def plot_umap_per_layer(
    inputs: torch.Tensor, model: GPT, umap: UMAPWithLaplacianInit, labels: list[str]
):
    b, t = inputs.size()
    assert b == 1, "Batch has more than one example"
    assert t == len(labels), "Labels do not match the number of characters"

    labels = [label + f"_{pos}" for pos, label in enumerate(labels)]

    # All layers + (emb + pe) + norm(emb + pe)
    n_plots = len(model.transformer.h) + 2

    rows, cols = find_closest_to_square(n_plots)
    f, axs = plt_subplots_3d(
        nrows=rows, ncols=cols, sharex=True, sharey=True, sharez=True
    )
    axs = axs.ravel()
    f.suptitle("Transformer Layers")

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

        for i, block in enumerate(model.transformer.h):
            x = block(x)

            # Map every layer to the norm space used at the end
            x_out = umap.transform(final_ln(x).cpu().numpy()[0, ...])

            delta = x_out - x_in

            axs[i + 2].scatter(
                emb_3d[:, 0],
                emb_3d[:, 1],
                emb_3d[:, 2],
                c="blue",
                label="Word Embeddings",
            )
            axs[i + 2].scatter(
                x_out[:, 0],
                x_out[:, 1],
                x_out[:, 2],
                c="purple",
                label="Latent Features",
            )

            add_labels(axs[i + 2], emb_3d, labels=labels, color="blue")
            add_labels(axs[i + 2], x_out, labels=labels, color="purple")
            axs[i + 2].set_title(f"Layer {i + 1}")
            axs[i + 2].legend()

            # Add trajectory for the second to last character
            pos_history.append(x_in[[-2], :])
            delta_history.append(delta[[-2], :])

            pos_ = np.concatenate(pos_history, axis=0)
            delta_ = np.concatenate(delta_history, axis=0)

            axs[i + 2].quiver(
                pos_[:, 0],
                pos_[:, 1],
                pos_[:, 2],
                delta_[:, 0],
                delta_[:, 1],
                delta_[:, 2],
                color="red",
                arrow_length_ratio=0,
            )

            x_in = x_out

    return f, axs


if __name__ == "__main__":
    plt.ion()

    TEST_INPUT = "Clown:"

    model, encode, decode = load_model(out_dir="nanoGPT/out-shakespeare-char")

    # Initialize UMAP
    umap = UMAPWithLaplacianInit(n_components=3)
    e = model.transformer.wte.weight.detach()
    _ = umap.fit(e.cpu().numpy())

    # Encode input
    inputs = encode(TEST_INPUT)
    inputs = torch.tensor([inputs]).to("cuda")

    f, axs = plot_umap_per_layer(
        inputs=inputs, model=model, umap=umap, labels=[letter for letter in TEST_INPUT]
    )
