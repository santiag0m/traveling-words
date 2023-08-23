import cmath
from typing import Optional

import torch
import numpy as np
from scipy.linalg import eig
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes

from nanoGPT.model import GPT

from model_utils import load_model, get_attn_weights_from_block, split_weights_by_head
from plot_utils import find_closest_to_square


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


def check_left_eigenvectors(A, left_eigenvectors, eigenvalues):
    for i, w in enumerate(left_eigenvectors):
        # Use np.conj to take the complex conjugate for complex vectors
        lhs = np.dot(np.conj(w), A)
        rhs = eigenvalues[i] * np.conj(w)

        if not np.allclose(lhs, rhs):
            return False
    return True


def check_left_eigenvectors(A, left_eigenvectors, eigenvalues):
    # Calculate the LHS for all left eigenvectors at once
    lhs = np.dot(
        left_eigenvectors, A.T
    ).T  # Using A.T here for correct matrix multiplication

    # Calculate the RHS using broadcasting
    rhs = eigenvalues * left_eigenvectors

    # Check if all are close
    return np.allclose(lhs, rhs)


import cmath


def compute_dilation_angle(eigenvalue):
    # Dilation (scale) factor is the magnitude of the eigenvalue
    dilation = abs(eigenvalue)

    # Rotation angle in radians
    angle_radians = cmath.phase(eigenvalue)

    # Convert the angle to degrees
    angle_degrees = angle_radians * (180 / cmath.pi)

    return dilation, angle_degrees


def plot_eigenvalues(
    wq_heads: list[torch.Tensor],
    wk_heads: list[torch.Tensor],
    use_pseudo: bool = False,
) -> tuple[plt.Figure, plt.Axes, np.ndarray, np.ndarray, np.ndarray]:
    n_head = len(wq_heads)
    assert n_head == len(wk_heads), "Number of heads do not match"

    rows, cols = find_closest_to_square(n_head)
    f, axs = plt.subplots(rows, cols)

    if max(rows, cols) > 1:
        _axs = axs.ravel()
    else:
        _axs = [axs]

    eigenvalues_per_head = []
    eigen_queries_per_head = []
    eigen_keys_per_head = []

    for idx in range(n_head):
        wq = wq_heads[idx]
        wk = wk_heads[idx]

        if use_pseudo:
            # Consider the factorized matrices as "true" singular vectors
            norm_q = np.linalg.norm(wq.to("cpu").numpy(), axis=1, ord=2)
            norm_k = np.linalg.norm(wk.to("cpu").numpy(), axis=1, ord=2)
            eigenvalues = norm_q * norm_k
        else:
            A = (wq @ wk.T).cpu().numpy()

            eigenvalues, eigen_query, eigen_key = eig(A, left=True)

            abs_eigenvalues = np.abs(eigenvalues)

        idxs = np.argsort(-1 * abs_eigenvalues)
        eigenvalues = eigenvalues[idxs]

        eigen_query = eigen_query[:, idxs]
        eigen_key = eigen_key[idxs, :]

        feature_percentile = np.linspace(0, 1, num=len(abs_eigenvalues))
        eigenvalue_coverage = np.cumsum(abs_eigenvalues) / sum(abs_eigenvalues)
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

        eigenvalues_per_head.append(eigenvalues)
        eigen_queries_per_head.append(eigen_query)
        eigen_keys_per_head.append(eigen_key)

    return (f, axs, eigenvalues_per_head, eigen_queries_per_head, eigen_keys_per_head)


def attention_eigenvalues(
    model: GPT, block_idx: int, print_top_eigen: int = 20, print_top_k: int = 3
) -> tuple[plt.Figure, plt.Axes]:
    wq, wk, wv, wo = get_attn_weights_from_block(model, block_idx=block_idx)

    with torch.no_grad():
        e = model.transformer.wte.weight.detach()
        e = model.transformer.ln_f(e)
        e = e.cpu().numpy()

    n_heads = model.config.n_head
    wq_heads, wk_heads, wv_heads, wo_heads = split_weights_by_head(
        wq, wk, wv, wo, n_heads=n_heads
    )
    (
        f,
        axs,
        eigenvalues_per_head,
        eigen_queries_per_head,
        eigen_keys_per_head,
    ) = plot_eigenvalues(wq_heads=wq_heads, wk_heads=wk_heads)

    f.suptitle(f"Eigenvalue analysis - Layer {block_idx}")

    if print_top_k:
        with open("eigenvalue_logs.txt", "a+") as file_obj:
            for i in range(n_heads):
                eigenvalues = eigenvalues_per_head[i][:print_top_eigen]
                eigen_queries = eigen_queries_per_head[i][:, :print_top_eigen]
                eigen_keys = eigen_keys_per_head[i][:, :print_top_eigen]

                eigen_queries_real_idxs = knn_dot_product(
                    eigen_queries.real.T, e, k=print_top_k
                )
                eigen_queries_imag_idxs = knn_dot_product(
                    eigen_queries.imag.T, e, k=print_top_k
                )
                # eigen_keys_idx = knn_dot_product(eigen_keys.T, e, k=print_top_k)

                file_obj.write(f"Top eigen queries for head {i} - block {block_idx}\n")
                for q in range(eigen_queries_real_idxs.shape[0]):
                    a, b = eigenvalues[q].real, eigenvalues[q].imag

                    difference = (
                        a * eigen_queries[:, [q]].real - b * eigen_queries[:, [q]].imag
                    )
                    similar = (
                        b * eigen_queries[:, [q]].real + a * eigen_queries[:, [q]].imag
                    )

                    middle = difference + similar

                    difference_idxs = knn_dot_product(difference.T, e, k=print_top_k)
                    similar_idxs = knn_dot_product(similar.T, e, k=print_top_k)
                    middle_idxs = knn_dot_product(middle.T, e, k=print_top_k)

                    file_obj.write(
                        f"\nEigenvalue: {q} - [{a:.2f}+{b:.2f}i]\n"
                        + f"\tReal part:  "
                        + f"\t{[decode([idx]) for idx in eigen_queries_real_idxs[q, :]]}\n"
                        + f"\tImag part:  "
                        + f"\t{[decode([idx]) for idx in eigen_queries_imag_idxs[q, :]]}\n"
                        + f"\t[Real -> Diff.] a * Real - b * Imag:  "
                        + f"\t{[decode([idx]) for idx in difference_idxs[0, :]]}\n"
                        + f"\t[Imag -> Similar] b * Real + a * Imag:  "
                        f"\t{[decode([idx]) for idx in similar_idxs[0, :]]}\n"
                        + f"\t[Middle]:  "
                        f"\t{[decode([idx]) for idx in middle_idxs[0, :]]}\n"
                    )

                    # eigen_query_target = eigenvalues[q] * eigen_queries[:, [q]].T
                    # eigen_query_target_real_idxs = knn_dot_product(
                    #     eigen_query_target.real, e, k=print_top_k
                    # )
                    # eigen_query_target_imag_idxs = knn_dot_product(
                    #     eigen_query_target.imag, e, k=print_top_k
                    # )
                    # dilation, angle = compute_dilation_angle(eigenvalues[q])
                    # angle = f" {angle:.2f}" if angle > 0 else f"{angle:.2f}"
                    # file_obj.write(
                    #     f"\nEigenvalue: {q} - [{dilation:.2f}x, {angle} deg]\n"
                    #     + f"\tReal part:  "
                    #     + f"\t{[decode([idx]) for idx in eigen_queries_real_idxs[q, :]]}  -->  {[decode([idx]) for idx in eigen_query_target_real_idxs[0, :]]}\n"
                    #     + f"\tImag part:  "
                    #     f"\t{[decode([idx]) for idx in eigen_queries_imag_idxs[q, :]]}  -->  {[decode([idx]) for idx in eigen_query_target_imag_idxs[0, :]]}",
                    # )

    return f, axs


def knn_dot_product(A, B, k):
    dot_products = np.matmul(A, B.T)  # Compute the dot products
    return np.argsort(dot_products, axis=1)[:, -k:][:, ::-1]


if __name__ == "__main__":
    plt.ion()

    model, encode, decode = load_model(init_from="gpt2")

    attention_eigenvalues(model=model, block_idx=0, print_top_k=5)
    attention_eigenvalues(model=model, block_idx=5, print_top_k=5)
    attention_eigenvalues(model=model, block_idx=11, print_top_k=5)
