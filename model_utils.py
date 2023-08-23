import os
import pickle
from typing import Callable, Optional

import torch
import tiktoken

from nanoGPT.model import GPTConfig, GPT, Block


def load_model(
    init_from: str = "gpt2", out_dir: Optional[str] = None, seed: int = 1337
) -> tuple[GPT, Callable, Callable]:
    device = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    # model
    if init_from == "resume":
        if out_dir is None:
            raise ValueError("No 'out_dir' to resume from was provided")

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

        meta_path = os.path.join(
            "nanoGPT", "data", checkpoint["config"]["dataset"], "meta.pkl"
        )
        load_meta = os.path.exists(meta_path)
        if load_meta:
            print(f"Loading meta from {meta_path}...")
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            # TODO want to make this more general to arbitrary encoder/decoder schemes
            stoi, itos = meta["stoi"], meta["itos"]
            encode = lambda s: [stoi[c] for c in s]
            decode = lambda l: "".join([itos[i] for i in l])
        else:
            enc = tiktoken.get_encoding("gpt2")
            encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
            decode = lambda l: enc.decode(l)
    elif init_from.startswith("gpt2"):
        # init from a given GPT-2 model
        model = GPT.from_pretrained(init_from, dict(dropout=0.0))
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)

    model.eval()
    model.to(device)
    return model, encode, decode


def get_attn_weights_from_block(
    model: GPT, block_idx: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    block: Block = model.transformer["h"][block_idx]
    weights = block.attn.c_attn.weight.detach()

    # Assume y = x @ W, instead of nn.linear
    wq, wk, wv = weights.T.split(model.config.n_embd, dim=1)

    # Output matrix should NOT be transposed
    wo = block.attn.c_proj.weight
    return wq, wk, wv, wo


def get_ln_from_block(
    model: GPT, block_idx: int
) -> tuple[torch.nn.LayerNorm, torch.nn.LayerNorm]:
    block: Block = model.transformer["h"][block_idx]
    return block.ln_1, block.ln_2


def split_weights_by_head(*args, n_heads: int) -> list[list[torch.Tensor]]:
    results = [[] for _ in range(len(args))]

    for idx, weight in enumerate(args):
        d_in, d_out = weight.shape
        assert (
            d_out % n_heads == 0
        ), "Weight dimensions not consistent with the number of heads"
        results[idx] += weight.split(d_out // n_heads, dim=1)

    return results
