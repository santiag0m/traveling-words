import torch
import numpy as np
from torch.linalg import svd
import matplotlib.pyplot as plt

from nanoGPT.model import GPT

from utils.model_utils import (
    load_model,
    get_attn_weights_from_block,
    get_attn_bias_from_block,
    split_weights_by_head,
)
from utils.plot_utils import plot_singular_values_per_head
from utils.linalg_utils import knn_dot_product


def svd_per_head(
    left_heads: list[torch.Tensor],
    right_heads: list[torch.Tensor],
    scale: float = 1,
) -> tuple[plt.Figure, plt.Axes, np.ndarray, np.ndarray, np.ndarray]:
    n_head = len(left_heads)
    assert n_head == len(right_heads), "Number of heads do not match"

    singular_values = []
    left_singular_vectors = []
    right_singular_vectors = []

    with torch.no_grad():
        for idx in range(n_head):
            w_left = left_heads[idx]
            w_right = right_heads[idx]

            A = scale * (w_left @ w_right.T)

            U, S, Vt = svd(A)

            singular_values.append(S)
            left_singular_vectors.append(U)
            right_singular_vectors.append(Vt)

    return (
        left_singular_vectors,
        singular_values,
        right_singular_vectors,
    )


def adjust_matrix_sign(
    U: torch.Tensor, E: torch.Tensor
) -> (torch.Tensor, torch.Tensor):
    """
    Given matrix U and reference matrix E, determine for each row of U,
    which is closer in terms of dot product to any of the rows in E: U or -U.
    Returns a matrix U_p and a vector S such that elementwise multiplication U_p * S = U.

    Args:
    - U (Tensor): Input matrix.
    - E (Tensor): Reference matrix.

    Returns:
    - torch.Tensor: Adjusted matrix U_p.
    - torch.Tensor: Vector S.
    """
    # Compute dot products
    U_dot_E = torch.mm(U, E.t())
    neg_U_dot_E = torch.mm(-U, E.t())

    # For each row in U, determine whether U or -U has the larger dot product with any row in E
    S = torch.where(
        torch.max(U_dot_E, dim=1).values >= torch.max(neg_U_dot_E, dim=1).values, 1, -1
    )

    # Construct U_p
    U_p = U * S[:, None]

    return U_p, S


def knn_svd_per_head(
    *,
    left_singular_vectors: list[torch.Tensor],
    right_singular_vectors: list[torch.Tensor],
    left_embed_matrix: torch.Tensor,
    right_embed_matrix: torch.Tensor,
    top_k_nearest: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert len(left_singular_vectors) == len(
        right_singular_vectors
    ), "Number of heads do not match"

    # Align signs
    U_aligned = []
    V_aligned = []
    signs = []

    for Vt in right_singular_vectors:
        Vp, S = adjust_matrix_sign(Vt, right_embed_matrix)
        V_aligned.append(Vp)
        signs.append(S)

    for U, S in zip(left_singular_vectors, signs):
        Up = U.T * S[:, None]
        U_aligned.append(Up)

    left_idxs = knn_per_head(
        U_aligned,
        left_embed_matrix,
        top_k_nearest=top_k_nearest,
    )
    right_idxs = knn_per_head(
        V_aligned, right_embed_matrix, top_k_nearest=top_k_nearest
    )

    return left_idxs, right_idxs


def knn_per_head(
    vectors_per_head: list[torch.Tensor], embed_matrix: torch.Tensor, top_k_nearest: int
) -> torch.Tensor:
    n_head = len(vectors_per_head)
    device = embed_matrix.device
    dims = vectors_per_head[0].shape[0]

    idxs = torch.zeros(n_head, dims, top_k_nearest, dtype=torch.int64, device=device)
    for i in range(n_head):
        with torch.no_grad():
            idxs[i, :, :] = knn_dot_product(
                vectors_per_head[i], embed_matrix, k=top_k_nearest
            )
    return idxs


def generate_latex_table(
    block_idx: int,
    head_idx: int,
    left_list: list[list[str]],
    right_list: list[list[str]],
) -> str:
    """Generates a LaTeX table from the given left and right lists."""

    # Start the table
    table = "\\begin{table}\n"
    table += "\\centering\n"
    table += (
        "\\caption{Left and Right Singular Vectors at "
        + f"Layer {block_idx} - Head {head_idx}"
        + "}\n"
    )
    table += "\\label{tab:" + f"svd_head_{head_idx}" + "}\n"
    table += "\\resizebox{\linewidth}{!}{\n"
    table += "\\begin{tabular}{c|p{0.4\linewidth}|p{0.4\linewidth}}\n"
    table += "\\toprule\n"
    table += "\\textbf{Rank} & \\textbf{Top-3 Left Words} & \\textbf{Top-3 Right Words} \\\\\n"
    table += "\\midrule\n"

    assert len(left_list) == len(
        right_list
    ), "Left and Right lists are not the same length"

    # Add each row
    for idx in range(len(left_list)):
        left_words = ", ".join(left_list[idx])
        right_words = ", ".join(right_list[idx])

        table += f"{idx} & {left_words} & {right_words} \\\\\n"

    # End the table
    table += "\\hline\n"
    table += "\\end{tabular}\n"
    table += "}\n"
    table += "\\end{table}\n\n"

    return table


def to_latex_escape(s):
    # Mapping of special characters to their LaTeX escape sequences
    escape_map = {
        "\\": "\\textbackslash",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde",
        "^": "\\textasciicircum",
        "&": "\\&",
        "\n": "\\textbackslash n",
    }

    # Convert special characters using the escape_map
    escaped = "".join(escape_map.get(c, c) for c in s)

    # Replace non-ASCII characters with their Unicode code point
    return "".join(
        c if ord(c) < 128 else "\\textbackslash u{:04x}".format(ord(c)) for c in escaped
    )


def attention_svd(
    model: GPT,
    block_idx: int,
    top_singular_values: int = 20,
    top_k_nearest: int = 5,
    plot: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    with torch.no_grad():
        e = model.transformer.wte.weight.detach()
        layer_norm = model.transformer["h"][block_idx].ln_1
        e_norm = layer_norm(e)

    n_heads = model.config.n_head
    wq, wk, wv, wo = get_attn_weights_from_block(model, block_idx=block_idx)
    wq_heads, wk_heads, wv_heads, wo_heads = split_weights_by_head(
        wq, wk, wv, wo, n_heads=n_heads
    )
    bias_q, bias_k, bias_v, bias_o = get_attn_bias_from_block(
        model, block_idx=block_idx
    )
    bias_q_heads, bias_k_heads, bias_v_heads = split_weights_by_head(
        bias_q,
        bias_k,
        bias_v,
        n_heads=n_heads,
    )

    dk = wq_heads[1].shape[1]

    U_qk, S_qk, Vt_qk = svd_per_head(
        left_heads=wq_heads, right_heads=wk_heads, scale=(1 / (dk**0.5))
    )
    U_qk = [U @ torch.diag(S) for U, S in zip(U_qk, S_qk)]

    U_vo, S_vo, Vt_vo = svd_per_head(left_heads=wv_heads, right_heads=wo_heads, scale=1)
    U_vo = [U @ torch.diag(S) for U, S in zip(U_vo, S_vo)]

    if plot:
        f1, _ = plot_singular_values_per_head([S[:dk].cpu().numpy() for S in S_qk])
        f1.suptitle(r"$W_{QK}$ SVD analysis - Layer " + f"{block_idx}")

        f2, _ = plot_singular_values_per_head([S[:dk].cpu().numpy() for S in S_vo])
        f2.suptitle(r"$W_{VO}$ SVD analysis - Layer " + f"{block_idx}")

    # W_qk
    idxs_qk = knn_svd_per_head(
        left_singular_vectors=U_qk,
        right_singular_vectors=Vt_qk,
        left_embed_matrix=e_norm,
        right_embed_matrix=e_norm,
        top_k_nearest=top_k_nearest,
    )

    # W_vo
    idxs_vo = knn_svd_per_head(
        left_singular_vectors=U_vo,
        right_singular_vectors=Vt_vo,
        left_embed_matrix=e_norm,
        right_embed_matrix=e,
        top_k_nearest=top_k_nearest,
    )

    filenames = ["svd_qk", "svd_vo"]
    S_list = [S_qk, S_vo]
    idxs_list = [idxs_qk, idxs_vo]

    for filename, S, pos_idxs in zip(filenames, S_list, idxs_list):
        left_idxs, right_idxs = pos_idxs

        with open(f"{filename}_logs.txt", "a+") as file_obj:

            def write_logs_section(
                section_title: str, idxs: torch.Tensor, head_idx: int
            ):
                singular_values = S[head_idx]
                singular_values = singular_values[:top_singular_values]
                file_obj.write(f"{section_title}:\n")
                for s in range(top_singular_values):
                    file_obj.write(
                        f"[{s}] {[decode([idx]) for idx in idxs[head_idx, s, :]]} [{singular_values[s]:.2f}]\n"
                    )

            for i in range(n_heads):
                file_obj.write(
                    f"\n=== Top singular values for block {block_idx} - head {i} ===\n"
                )
                write_logs_section("Left", idxs=left_idxs, head_idx=i)
                write_logs_section("Right", idxs=right_idxs, head_idx=i)

        with open(f"{filename}_latex_table_layer_{block_idx}.txt", "a+") as f:
            for i in range(n_heads):
                left_list = []
                right_list = []
                for s in range(top_singular_values):
                    left_list.append(
                        [to_latex_escape(decode([idx])) for idx in left_idxs[i, s, :3]]
                    )
                    right_list.append(
                        [to_latex_escape(decode([idx])) for idx in right_idxs[i, s, :3]]
                    )

                f.write(
                    generate_latex_table(
                        block_idx=block_idx,
                        head_idx=i,
                        left_list=left_list,
                        right_list=right_list,
                    )
                )

    # Query bias
    query_biases = [(q @ bk.T).T for q, bk in zip(wq_heads, bias_k_heads)]
    pos_idxs_query_bias = knn_per_head(
        vectors_per_head=query_biases, embed_matrix=e_norm, top_k_nearest=top_k_nearest
    )
    neg_idxs_query_bias = knn_per_head(
        vectors_per_head=[-1 * bq for bq in query_biases],
        embed_matrix=e_norm,
        top_k_nearest=top_k_nearest,
    )

    # Key bias
    key_biases = [bq @ k.T for k, bq in zip(wk_heads, bias_q_heads)]
    pos_idxs_key_bias = knn_per_head(
        vectors_per_head=key_biases, embed_matrix=e_norm, top_k_nearest=top_k_nearest
    )
    neg_idxs_key_bias = knn_per_head(
        vectors_per_head=[-1 * bk for bk in key_biases],
        embed_matrix=e_norm,
        top_k_nearest=top_k_nearest,
    )

    with open("key_query_bias_logs.txt", "a+") as f:
        for i in range(n_heads):
            f.write(f"Block {block_idx} - Head: {i}\n")
            f.write(
                f"[+] Query: {[decode([idx]) for idx in pos_idxs_query_bias[i, 0, :]]}\n"
            )
            f.write(
                f"[+] Key: {[decode([idx]) for idx in pos_idxs_key_bias[i, 0, :]]}\n"
            )
            f.write(
                f"[-] Query: {[decode([idx]) for idx in neg_idxs_query_bias[i, 0, :]]}\n"
            )
            f.write(
                f"[-] Key: {[decode([idx]) for idx in neg_idxs_key_bias[i, 0, :]]}\n\n"
            )


if __name__ == "__main__":
    plt.ion()

    model, encode, decode = load_model(init_from="gpt2")

    attention_svd(model=model, block_idx=0, top_k_nearest=5, plot=False)
    attention_svd(model=model, block_idx=5, top_k_nearest=5, plot=False)
    attention_svd(model=model, block_idx=11, top_k_nearest=5, plot=False)
