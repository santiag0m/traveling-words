from typing import Callable

import torch

from nanoGPT.model import GPT

from utils.model_utils import (
    load_model,
    get_attn_weights_from_block,
    get_attn_bias_from_block,
    split_weights_by_head,
    get_ln_from_block,
)
from utils.linalg_utils import knn_dot_product


def load_nouns(
    encode: Callable,
    txt_file: str = "COMMON_NOUNS.txt",
) -> tuple[list[int], list[str]]:
    with open(txt_file, "r") as f:
        nouns = f.readlines()
        nouns = [noun.strip() for noun in nouns]

    idxs = []
    labels = []
    for noun in nouns:
        noun_idx = encode(noun)

        if len(noun_idx) == 1:
            [noun_idx] = noun_idx
            idxs.append(noun_idx)
            labels.append(noun)

    return idxs, labels


def attention_ellipse(
    model: GPT,
    encode: Callable,
    decode: Callable,
    block_idx: int,
    print_top_k_nearest: int = 3,
    txt_file: str = "COMMON_NOUNS.txt",
):
    n_heads = model.config.n_head

    layer_norm, _ = get_ln_from_block(model, block_idx=block_idx)
    wq, wk, wv, wo = get_attn_weights_from_block(model, block_idx=block_idx)
    wq_heads, wk_heads, wv_heads, wo_heads = split_weights_by_head(
        wq, wk, wv, wo, n_heads=n_heads
    )
    _, _, bias_v, bias_o = get_attn_bias_from_block(model, block_idx=block_idx)
    bias_v_heads = split_weights_by_head(
        bias_v,
        n_heads=n_heads,
    )
    model_dim = wq.shape[0]

    query_idxs, query_labels = load_nouns(encode=encode, txt_file=txt_file)
    n_queries = len(query_labels)

    # Project to hypersphere and transform
    with torch.no_grad():
        e = model.transformer.wte.weight.detach()
        e_norm = layer_norm(e)

        queries = e_norm[query_idxs, :]

    keys = torch.zeros(n_heads, n_queries, print_top_k_nearest, dtype=torch.int64)
    values = torch.zeros(n_heads + 1, n_queries, print_top_k_nearest, dtype=torch.int64)
    value_norms = torch.zeros(n_heads + 1, n_queries, dtype=torch.int64)

    with torch.no_grad():
        layer_update = 0
        for i in range(n_heads):
            w_qk = wq_heads[i] @ wk_heads[i].T / ((model_dim / n_heads) ** 0.5)
            w_vo = wv_heads[i] @ wo_heads[i].T

            keys_head = queries @ w_qk
            word_idxs_keys = knn_dot_product(keys_head, e_norm, k=print_top_k_nearest)

            head_update = queries @ w_vo
            head_update += bias_v_heads[i] @ wo_heads[i].T  # Add value bias

            layer_update += head_update  # Accummulate for later

            values_head = e[query_idxs] + head_update + bias_o
            values_head = model.transformer.ln_f(values_head)

            word_idxs_values = knn_dot_product(values_head, e, k=print_top_k_nearest)
            keys[i, :, :] = word_idxs_keys
            values[i, :, :] = word_idxs_values
            value_norms[i, :] = torch.linalg.norm(head_update, dim=1)

        values_layer = e[query_idxs] + layer_update + bias_o
        values_layer = model.transformer.ln_f(values_layer)

        word_idxs_values = knn_dot_product(values_layer, e, k=print_top_k_nearest)
        values[-1, :, :] = word_idxs_values
        value_norms[-1, :] = torch.linalg.norm(layer_update, dim=1)

    with open("common_noun_logs_qk.txt", "a+") as file_obj:
        file_obj.write(f"=== block {block_idx} ===\n")
        for i in range(n_queries):
            file_obj.write(f"Query: {query_labels[i]}\n" + f"Keys:\n")

            for j in range(n_heads):
                file_obj.write(
                    f"\tHead {j}: {[decode([idx]) for idx in keys[j, i, :]]}" + "\n"
                )
            file_obj.write("\n")

    with open("common_noun_logs_vo.txt", "a+") as file_obj:
        file_obj.write(f"=== block {block_idx} ===\n")
        for i in range(n_queries):
            file_obj.write(f"Key: {query_labels[i]}\n" + f"Values:\n")

            for j in range(n_heads):
                norm = value_norms[j, i].item()
                file_obj.write(
                    f"\tHead {j} [{norm:.4f}]: {[decode([idx]) for idx in values[j, i, :]]}"
                    + "\n"
                )

            norm = value_norms[-1, i].item()
            file_obj.write(
                f"\tLayer Sum [{norm:.4f}]: {[decode([idx]) for idx in values[-1, i, :]]}"
                + "\n"
            )
            file_obj.write("\n")


if __name__ == "__main__":
    model, encode, decode = load_model(init_from="gpt2")

    attention_ellipse(
        model=model, encode=encode, decode=decode, block_idx=0, print_top_k_nearest=5
    )
    attention_ellipse(
        model=model, encode=encode, decode=decode, block_idx=5, print_top_k_nearest=5
    )
    attention_ellipse(
        model=model, encode=encode, decode=decode, block_idx=11, print_top_k_nearest=5
    )
