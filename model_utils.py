import os

import torch

from nanoGPT.model import GPTConfig, GPT


def load_model(out_dir: str, init_from: str = "resume", seed: int = 1337) -> GPT:
    device = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    # model
    if init_from == "resume":
        # init from a model saved in a specific directory
        ckpt_path = os.path.join(out_dir, "ckpt.pt")
        checkpoint = torch.load(ckpt_path, map_location=device)
        gptconf = GPTConfig(**checkpoint["model_args"])
        model = GPT(gptconf)
        state_dict = checkpoint["model"]
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
    elif init_from.startswith("gpt2"):
        # init from a given GPT-2 model
        model = GPT.from_pretrained(init_from, dict(dropout=0.0))

    model.eval()
    model.to(device)
    return model


def get_attn_weights_from_block(
    model: GPT, block_idx: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    block = model.transformer["h"][block_idx]
    weights = block.attn.c_attn.weight.detach()

    # Assume y = x @ W, instead of nn.linear
    wq, wk, wv = weights.T.split(model.config.n_embd, dim=1)
    return wq, wk, wv


def split_weights_by_head(*args, n_heads: int) -> list[list[torch.Tensor]]:
    results = [[] for _ in range(len(args))]

    for idx, weight in enumerate(args):
        d_in, d_out = weight.shape
        assert (
            d_out % n_heads == 0
        ), "Weight dimensions not consistent with the number of heads"
        results[idx] += weight.split(d_out // n_heads, dim=1)

    return results