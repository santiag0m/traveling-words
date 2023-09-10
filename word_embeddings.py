from typing import Callable

import torch
import tiktoken
import pandas as pd

from nanoGPT.model import GPT

from utils.model_utils import load_model


def measure_l2_distance(A: torch.Tensor, B: torch.Tensor) -> tuple[float, float]:
    diff = A - B
    norm = torch.linalg.norm(diff, dim=1)
    return norm.mean().item(), norm.std().item()


def measure_cosine_distance(A: torch.Tensor, B: torch.Tensor) -> tuple[float, float]:
    cosine_similarity = torch.nn.functional.cosine_similarity(A, B)
    cosine_distance = 1 - cosine_similarity
    return cosine_distance.mean().item(), cosine_distance.std().item()


def format_bytes(byte_seq):
    return "b'" + "".join(f"\\x{byte:02x}" for byte in byte_seq) + "'"


def inspect_top_and_bottom_k(
    model: GPT,
    decode_bytes: Callable,
    k: int = 5,
    scale: bool = True,
    bias: bool = True,
):
    with torch.no_grad():
        e = model.transformer.wte.weight.detach()
        gamma = model.transformer.ln_f.weight
        beta = model.transformer.ln_f.bias
        bias_score = ((beta.view(1, -1) @ e.T).T)[:, 0]

        if scale:
            e = (torch.diag(gamma) @ e.T).T

    norms = torch.linalg.norm(e, dim=1)
    if bias:
        norms += bias_score
    top_k = torch.argsort(norms, descending=True)[:k]
    bottom_k = torch.argsort(norms, descending=False)[:k]

    if bias and scale:
        setting = "bias_and_scale"
    elif scale:
        setting = "scale"
    elif bias:
        setting = "bias"
    else:
        setting = "original"

    records = []
    for i, idx in enumerate(top_k):
        byte_seq = decode_bytes([idx])
        str_repr = byte_seq.decode("utf-8", "replace")
        records.append(
            {
                "Position": f"Top {i+1}",
                "Norm": norms[idx].item(),
                "String": str_repr,
                "Bytes": format_bytes(byte_seq),
                "Setting": setting,
            }
        )

    for i, idx in enumerate(bottom_k.cpu().numpy()[::-1]):
        byte_seq = decode_bytes([idx])
        str_repr = byte_seq.decode("utf-8", "replace")
        records.append(
            {
                "Position": f"Bottom {k-i}",
                "Norm": norms[idx].item(),
                "String": str_repr,
                "Bytes": format_bytes(byte_seq),
                "Setting": setting,
            }
        )

    records = pd.DataFrame.from_records(records)

    return records


def normalize_word_embeddings(model: GPT) -> pd.DataFrame:
    with torch.no_grad():
        e = model.transformer.wte.weight.detach()

    _, dim_size = e.shape

    # Calculate avg norm in e and scale
    avg_norm = torch.linalg.norm(e, dim=1).mean().item()
    scale_factor = (dim_size**0.5) / avg_norm

    print(f"Avg. Norm: {avg_norm:.3f}")

    # Calculate center
    center = e.mean(dim=0, keepdim=True)

    # Calculate different matrices and their distance
    e_center = e - center
    e_scaled = e * scale_factor
    e_center_and_scaled = e_center * scale_factor
    e_norm = torch.nn.functional.layer_norm(e, normalized_shape=[dim_size])

    records = []
    for setting, mat in zip(
        ["Original", "Centered", "Scaled", "Centered & Scaled"],
        [e, e_center, e_scaled, e_center_and_scaled],
    ):
        mean_l2_distance, l2_std_dev = measure_l2_distance(mat, e_norm)
        mean_cosine_distance, cosine_std_dev = measure_cosine_distance(mat, e_norm)

        records.append(
            {
                "setting": setting,
                "mean_l2_distance": mean_l2_distance,
                "l2_std_dev": l2_std_dev,
                "mean_cosine_distance": mean_cosine_distance,
                "cosine_std_dev": cosine_std_dev,
            }
        )

    return pd.DataFrame.from_records(records)


if __name__ == "__main__":
    model, encode, decode = load_model(init_from="gpt2")

    stats = normalize_word_embeddings(model)
    pd.options.display.float_format = "{:,.3f}".format
    print(stats)

    original = inspect_top_and_bottom_k(
        model,
        decode_bytes=tiktoken.get_encoding("gpt2").decode_bytes,
        k=5,
        scale=False,
        bias=False,
    )
    scaled = inspect_top_and_bottom_k(
        model,
        decode_bytes=tiktoken.get_encoding("gpt2").decode_bytes,
        k=5,
        scale=True,
        bias=False,
    )
    bias = inspect_top_and_bottom_k(
        model,
        decode_bytes=tiktoken.get_encoding("gpt2").decode_bytes,
        k=5,
        scale=False,
        bias=True,
    )
    scaled_and_bias = inspect_top_and_bottom_k(
        model,
        decode_bytes=tiktoken.get_encoding("gpt2").decode_bytes,
        k=5,
        scale=True,
        bias=True,
    )

    all_df = pd.concat([original, scaled, bias, scaled_and_bias], axis=0)
    print(all_df)
