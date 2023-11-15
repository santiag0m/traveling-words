import torch
import tiktoken
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets import load_dataset

from nanoGPT.model import GPT
from utils.model_utils import load_model


Batch = list[list[str]]


def prepare_batches(*, batch_size: int, block_size: int) -> list[Batch]:
    print("Preparing batches ...")

    dataset_name = "stas/openwebtext-10k"
    dataset = load_dataset(dataset_name, split="train")

    enc = tiktoken.get_encoding("gpt2")

    all_lines = []

    for document in dataset:
        lines: list[str] = document["text"].split(".")
        lines = [line.strip() for line in lines]
        lines = [line for line in lines if len(line) > 0]
        all_lines += lines

    all_lines = list(sorted(all_lines, key=lambda x: len(x)))

    batches = [
        all_lines[i : i + batch_size] for i in range(0, len(all_lines), batch_size)
    ]
    batches = [
        [idxs[:block_size] for idxs in enc.encode_ordinary_batch(batch)]
        for batch in batches
    ]

    print("Batches ready!")

    return batches


def pad_batch(batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = len(batch)
    max_len = max([len(line) for line in batch])

    idxs = torch.zeros((batch_size, max_len), dtype=torch.int64)
    padding_mask = torch.ones_like(idxs, dtype=torch.bool)

    for i, line_idxs in enumerate(batch):
        n = len(line_idxs)
        idxs[i, :n] = torch.tensor(line_idxs, dtype=torch.int64)
        padding_mask[i, n:] = 0

    return idxs, padding_mask


def calculate_similarity(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    diff = (x - y) ** 2
    diff = diff.sum(dim=-1) ** 0.5
    dot = (x * y).sum(dim=-1)
    return dot


def smooth_lines(line, kernel_size=10):
    kernel = np.ones(kernel_size) / kernel_size
    return np.convolve(line, kernel, mode="same")


def plot_lines(x: torch.Tensor):
    x = x.numpy()

    num_layers, pos = x.shape

    cm = plt.get_cmap("viridis")
    colors = [cm(1.0 * i / num_layers) for i in range(num_layers)]

    f, ax = plt.subplots(figsize=(12, 8))
    for i in range(num_layers):
        ax.plot(smooth_lines(x[i]), color=colors[i], label=f"Layer {i+1}")

    ax.set_xlabel("Sequence Position (S)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    return f, ax


def ratio_to_next_token(
    batches: list[list[int]], model: GPT, block_size: int = 1024
) -> torch.Tensor:
    with torch.no_grad():
        num_layers = len(model.transformer.h)

        total_distances = torch.zeros(num_layers, block_size, dtype=torch.float64)
        sample_counts = torch.zeros_like(total_distances)

        for batch in tqdm(batches):
            idxs, padding_mask = pad_batch(batch)
            b, seq_len = idxs.shape

            if seq_len < 2:
                continue

            seq_len = seq_len - 1  # Account for shifted token
            idxs = idxs.to("cuda")

            pos = torch.arange(0, seq_len, dtype=torch.long, device=idxs.device)
            pos_emb = model.transformer.wpe(pos)

            emb = model.transformer.wte(idxs)
            x = emb[:, :-1, :] + pos_emb
            y_diff = emb[:, 1:, :] - emb[:, :-1, :]
            padding_mask = padding_mask[:, :-1]

            final_ln = model.transformer.ln_f

            for i, block in enumerate(model.transformer.h):
                x = block(x)
                distance = (
                    calculate_similarity(final_ln(x), y_diff).cpu().type(torch.float64)
                )  # B, S
                distance = padding_mask * distance  # B, S
                distance = distance.sum(dim=0)  # S

                total_distances[i, :seq_len] += distance
                sample_counts[i, :seq_len] += padding_mask.sum(dim=0)

        mean_distance = total_distances / (sample_counts + 1e-5)

    return mean_distance


if __name__ == "__main__":
    batch_size = 256
    block_size = 1024

    model, encode, decode = load_model(init_from="gpt2")

    batches = prepare_batches(batch_size=batch_size, block_size=block_size)

    batches = batches[::-1]  # Bigger batches go first

    mean_distance = ratio_to_next_token(
        batches=batches, model=model, block_size=block_size
    )

    plt.ion()

    f, ax = plot_lines(mean_distance)
    ax.set_title(r"$log(p(w_{t+1})/p(w_{t}))$")
    ax.grid()
    f.tight_layout()
