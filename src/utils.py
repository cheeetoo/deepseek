import glob
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


def init(key: jax.Array, shape: tuple[int, ...], dtype=jnp.bfloat16) -> jax.Array:
    return 0.006 * jax.random.normal(key, shape, dtype)


@dataclass
class Config:
    dim: int
    dc: int
    dim_nope_head: int
    dim_rope_head: int
    n_heads: int
    moe_inter_dim: int
    n_shared_experts: int
    n_routed_experts: int
    n_activated_experts: int
    max_seqlen: int
    rope_theta: int
    eps: float
    n_vocab: int
    n_blocks: int
    n_mtp: int
    batch_size: int
    mtp_lambda: float
    bias_update_rate: float
    aux_alpha: float


def get_adamw_state(params):
    zeros_like = lambda p: jnp.zeros_like(p, p.dtype, device=p.device)
    m = jax.tree.map(zeros_like, params)
    v = jax.tree.map(zeros_like, params)
    return m, v


def adamw(params, grads, m, v, t, wd, lr=0.001, b1=0.9, b2=0.99, eps=1e-8):
    m = jax.tree.map(lambda m, g: b1 * m + (1 - b1) * g, m, grads)
    m = jax.tree.map(lambda m, g: b1 * m + (1 - b1) * g, m, grads)
    v = jax.tree.map(lambda v, g: b2 * v + (1 - b2) * (g**2), v, grads)

    m_hat = jax.tree.map(lambda m: m / (1 - b1**t), m)
    v_hat = jax.tree.map(lambda v: v / (1 - b2**t), v)

    update_fn = lambda p, m_hat, v_hat: p - lr * (
        m_hat / (jnp.sqrt(v_hat) + eps) + wd * p
    )

    params = jax.tree.map(
        lambda p, m_hat, v_hat: None if p is None else update_fn(p, m_hat, v_hat),
        params,
        m_hat,
        v_hat,
        is_leaf=lambda x: x is None,
    )

    return params, m, v


# dataloader modified from https://github.com/karpathy/llm.c/blob/master/train_gpt2.py
def _peek_data_shard(filename):
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
    ntok = header[2]
    return ntok


def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # Skip header bytes (256 int32 values)
        f.read(256 * 4)
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    return tokens


class DataLoader:
    def __init__(self, filename_pattern: str, B: int, T: int, n_mpt: int):
        self.B = B
        self.T = T
        self.n_pred = n_mpt + 1
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, (
            f"Did not find any files that match the pattern {filename_pattern}"
        )
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= B * T + n_mpt, (
                "Shard doesn't have enough tokens for one batch"
            )
            ntok_total += shard_ntok
        self.ntok_total = ntok_total
        print(
            f"DataLoader: total number of tokens: {ntok_total:,} across {len(self.files)} files"
        )
        self.current_shard = None
        self.reset()

    def reset(self):
        if self.current_shard != 0:
            self.current_shard = 0
            self.tokens = _load_data_shard(self.files[self.current_shard])
        self.current_position = 0

    def advance(self):
        self.current_shard = (self.current_shard + 1) % len(self.files)  # type: ignore
        self.current_position = 0
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B, T, n_pred = self.B, self.T, self.n_pred
        buf = self.tokens[
            self.current_position : self.current_position + B * (T + n_pred)
        ]
        buf = buf.astype(np.int32)
        buf = buf.reshape(B, T + n_pred)
        x = buf[:, :T]
        y = np.stack([buf[:, i + 1 : i + 1 + n_pred] for i in range(T)], axis=1)
        self.current_position += B * T
        if self.current_position + B * T + n_pred > len(self.tokens):
            self.advance()
        return x, y
