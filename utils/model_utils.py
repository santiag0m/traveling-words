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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    block: Block = model.transformer["h"][block_idx]
    weights = block.attn.c_attn.weight.detach()

    # Assume y = x @ W, instead of nn.linear
    wq, wk, wv = weights.T.split(model.config.n_embd, dim=1)

    # Output matrix should NOT be transposed
    wo = block.attn.c_proj.weight
    return wq, wk, wv, wo


def get_attn_bias_from_block(
    model: GPT, block_idx: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    block: Block = model.transformer["h"][block_idx]
    if model.config.bias:
        bias_q, bias_k, bias_v = (
            block.attn.c_attn.bias.detach()
            .view(1, -1)
            .split(model.config.n_embd, dim=1)
        )
        bias_o = block.attn.c_proj.bias.view(1, -1)
    else:
        dims = model.config.n_embd * model.config.n_head
        device = model.transformer.wte.weight.device
        bias_q = torch.zeros(
            1,
            dims,
            device=device,
        )
        bias_k = bias_q.clone()
        bias_v = bias_q.clone()
        bias_o = torch.zeros(
            1,
            model.config.n_embd,
            device=device,
        )

    return bias_q, bias_k, bias_v, bias_o


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

    if len(results) == 1:
        results = results[0]

    return results


def sanity_check(
    model: GPT, block_idx: int, random_bias_q: bool = False, random_bias_k: bool = True
):
    block: Block = model.transformer["h"][block_idx]

    model.eval()

    n_heads = model.config.n_head
    wq, wk, wv, wo = get_attn_weights_from_block(model, block_idx=block_idx)
    wq_heads, wk_heads, wv_heads, wo_heads = split_weights_by_head(
        wq, wk, wv, wo, n_heads=n_heads
    )
    ln_1, _ = get_ln_from_block(model, block_idx)

    dim = wq.shape[0]
    dummy_input = torch.randn(1, 10, dim, device=wq.device)
    reference_output = dummy_input + block.attn(ln_1(dummy_input))

    if model.config.bias:
        bias_q, bias_k, bias_v = block.attn.c_attn.bias.view(1, -1).split(
            model.config.n_embd, dim=1
        )
        if random_bias_q:
            bias_q = torch.rand_like(bias_q)

        if random_bias_k:
            bias_k = torch.rand_like(bias_k)

        bias_o = block.attn.c_proj.bias.view(1, -1)
        bias_q_heads, bias_k_heads, bias_v_heads = split_weights_by_head(
            bias_q,
            bias_k,
            bias_v,
            n_heads=n_heads,
        )
    else:
        bias_o = 0

    x = dummy_input.clone()
    x_norm = ln_1(x)
    for head in range(n_heads):
        q = wq_heads[head]
        k = wk_heads[head]
        v = wv_heads[head]
        o = wo_heads[head]

        scale = 1.0 / (q.size(-1) ** 0.5)
        qk = q @ k.T
        vo = v @ o.T

        x_query = x_norm @ qk  # (b, s, d) @ (d, d) -> (b, s, d)
        attn_scores = x_query @ torch.transpose(
            x_norm, 1, 2
        )  # (b, s, d) @ (b, d, s') -> (b, s, s')
        updates = x_norm @ vo  # (b, s', d) @ (d, d) -> (b, s', d)

        if model.config.bias:
            bq = bias_q_heads[head]
            bk = bias_k_heads[head]
            bv = bias_v_heads[head]
        else:
            bq = 0
            bk = 0
            bv = 0

        attn_scores += (x_norm @ q) @ bk.T
        attn_scores += bq @ (k.T @ torch.transpose(x_norm, 1, 2))
        attn_scores += bq @ bk.T

        causal_mask = torch.zeros_like(attn_scores[0, ...])
        causal_mask[torch.tril(torch.ones_like(causal_mask)) == 0] = -1 * torch.inf

        attn_scores += causal_mask[None, ...]

        updates += bv @ o.T

        attn_weights = torch.softmax(scale * attn_scores, dim=2)  # (b, s, s')

        x += attn_weights @ updates  # (b, s, s') @ (b, s', d) -> (b, s, d)

    x += bias_o
    manual_output = x

    return reference_output, manual_output
