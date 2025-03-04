from typing import NamedTuple

import equinox as eqx
import jax
import jax.nn as nn
import jax.numpy as jnp
from einops import einsum, reduce, repeat
from jax import Array

from utils import Config, init
from model import Gate, swiglu, RMSNorm, apply_rotary_emb


class KVCache(NamedTuple):
    k: jax.Array
    kv: jax.Array

    @classmethod
    def new(
        cls, layers: int, bsz: int, max_seq_len: int, drh: int, dc: int
    ) -> "KVCache":
        return cls(
            k=jnp.zeros((layers, bsz, max_seq_len, drh), dtype=jnp.bfloat16),
            kv=jnp.zeros((layers, bsz, max_seq_len, dc), dtype=jnp.bfloat16),
        )

    def update(
        self, xk: jax.Array, xv: jax.Array, layer_idx: int, cur_pos: int, n_rep: int
    ):
        n_toks = xk.shape[-2] + cur_pos
        ck = jax.lax.dynamic_update_slice(
            self.k, jnp.bfloat16(xk[None, ...]), (layer_idx, 0, cur_pos, 0)
        )
        ckv = jax.lax.dynamic_update_slice(
            self.kv, jnp.bfloat16(xv[None, ...]), (layer_idx, 0, cur_pos, 0)
        )
        keys = (
            repeat(ck[layer_idx, :, :n_toks], "b t drh -> b nh t drh", nh=n_rep) / n_rep
        )
        values = ckv[layer_idx, :, :n_toks]

        return keys, values, KVCache(k=ck, kv=ckv)


class IMLA(eqx.Module):
    w_dkv: Array
    w_uk: Array
    w_uv: Array
    w_dq: Array
    w_uq: Array
    w_qr: Array
    w_kr: Array
    w_o: Array

    nh: int = eqx.field(static=True)
    dh: int = eqx.field(static=True)
    drh: int = eqx.field(static=True)

    def __init__(self, cfg: Config, key: Array):
        self.nh = cfg.n_heads
        self.dh = cfg.dim_nope_head
        self.drh = cfg.dim_rope_head

        key_dkv, key_uk, key_uv, key_dq, key_uq, key_qr, key_kr, key_wo = (
            jax.random.split(key, 8)
        )

        self.w_dkv = init(key_dkv, (cfg.dc, cfg.dim))
        self.w_uk = init(key_uk, (cfg.dim_nope_head, cfg.n_heads, cfg.dc))
        self.w_uv = init(key_uv, (cfg.dim_nope_head, cfg.n_heads, cfg.dc))
        self.w_dq = init(key_dq, (cfg.dc, cfg.dim))
        self.w_uq = init(key_uq, (cfg.dim_nope_head, cfg.n_heads, cfg.dc))
        self.w_qr = init(key_qr, (cfg.dim_rope_head, cfg.n_heads, cfg.dc))
        self.w_kr = init(key_kr, (cfg.dim_rope_head, cfg.dim))
        self.w_o = init(key_wo, (cfg.dim, cfg.dim_nope_head, cfg.n_heads))

    def __call__(
        self,
        h: Array,
        freqs_cis: Array,
        mask: Array,
        kvcache: KVCache,
        layer_idx: int,
        cur_pos: int,
    ) -> tuple[Array, KVCache]:
        k_r = apply_rotary_emb(
            einsum(self.w_kr, h, "drh d, b t d -> b t drh"), freqs_cis
        )
        c_kv = einsum(self.w_dkv, h, "dc d, b t d -> b t dc")
        k_r, c_kv, kvcache = kvcache.update(k_r, c_kv, layer_idx, cur_pos, self.nh)

        k_c = einsum(self.w_uk, c_kv, "dh nh dc, b t dc -> b nh t dh")
        v_c = einsum(self.w_uv, c_kv, "dh nh dc, b t dc -> b nh t dh")

        c_q = einsum(self.w_dq, h, "dc d, b t d -> b t dc")
        q_c = einsum(self.w_uq, c_q, "dh nh dc, b t dc -> b nh t dh")

        q_r = apply_rotary_emb(
            einsum(self.w_qr, c_q, "drh nh dc, b t dc -> b nh t drh"), freqs_cis
        )

        q = jnp.concat((q_c, q_r), axis=-1)
        k = jnp.concat((k_c, k_r), axis=-1)

        logits = einsum(q, k, "b nh t d, b nh l d -> b nh t l")
        logits = logits / jnp.sqrt(self.dh + self.drh) + mask
        scores = jax.nn.softmax(logits.astype(jnp.float32), -1).astype(h.dtype)

        out = einsum(scores, v_c, "b nh t l, b nh l dh -> b nh t dh")

        return einsum(self.w_o, out, "d dh nh, b nh t dh -> b t d"), kvcache


class IGate(Gate):
    def __call__(self, x: Array) -> tuple[Array, Array]:  # type: ignore
        logits = einsum(self.w, x, "ne d, b t d -> b t ne")
        scores = nn.sigmoid(logits) + self.gate_b
        weights, indices = jax.lax.top_k(scores, self.top_k)
        weights /= weights.sum(-1, keepdims=True)
        return (weights, indices)


class IFFN(eqx.Module):
    w1_shared: Array
    w2_shared: Array
    w3_shared: Array
    w1_routed: Array
    w2_routed: Array
    w3_routed: Array
    gate: Gate

    n_activated_experts: int = eqx.field(static=True)

    def __init__(self, cfg: Config, key: Array):
        self.n_activated_experts = cfg.n_activated_experts
        key1, key2, key3, key4, key5, key6, key7 = jax.random.split(key, 7)

        self.w1_shared = init(key1, (cfg.dim, cfg.moe_inter_dim * cfg.n_shared_experts))
        self.w2_shared = init(key2, (cfg.moe_inter_dim * cfg.n_shared_experts, cfg.dim))
        self.w3_shared = init(key3, (cfg.dim, cfg.moe_inter_dim * cfg.n_shared_experts))
        self.w1_routed = init(key4, (cfg.n_routed_experts, cfg.dim, cfg.moe_inter_dim))
        self.w2_routed = init(key5, (cfg.n_routed_experts, cfg.moe_inter_dim, cfg.dim))
        self.w3_routed = init(key6, (cfg.n_routed_experts, cfg.dim, cfg.moe_inter_dim))
        self.gate = IGate(cfg, key7)

    def __call__(self, x: Array) -> Array:
        weights, indices = self.gate(x)

        shared_out = swiglu(x, self.w1_shared, self.w2_shared, self.w3_shared)

        one_hot = jax.nn.one_hot(indices, self.w1_routed.shape[0], dtype=x.dtype)
        router_probs = weights[..., None] * one_hot
        router_probs = reduce(router_probs, "b t na ne -> b t ne", "sum")

        h1 = einsum(x, self.w1_routed, "b t d, ne d c -> b t ne c")
        h2 = einsum(x, self.w3_routed, "b t d, ne d c -> b t ne c")
        h3 = nn.silu(h1 * h2)

        experts_out = einsum(h3, self.w2_routed, "b t e h, e h d -> b t e d")

        routed_out = einsum(experts_out, router_probs, "b t e d, b t e -> b t d")

        return shared_out + routed_out


class IBlock(eqx.Module):
    attn_norm: RMSNorm
    attn: IMLA
    ffn_norm: RMSNorm
    ffn: IFFN

    def __init__(self, cfg: Config, key: Array):
        key_attn, key_ffn = jax.random.split(key, 2)
        self.attn_norm = RMSNorm(cfg)
        self.attn = IMLA(cfg, key_attn)
        self.ffn_norm = RMSNorm(cfg)
        self.ffn = IFFN(cfg, key_ffn)

    def __call__(
        self,
        x: Array,
        freqs_cis: Array,
        mask: Array,
        kvcache: KVCache,
        layer_idx: int,
        cur_pos: int,
    ) -> tuple[Array, KVCache]:
        h, kvcache = self.attn(
            self.attn_norm(x), freqs_cis, mask, kvcache, layer_idx, cur_pos
        )
        x = x + h
        ffn_out = self.ffn(self.ffn_norm(x))
        return x + ffn_out, kvcache


class IMTPBlock(eqx.Module):
    norm_prev: RMSNorm
    norm_emb: RMSNorm
    proj: Array
    block: IBlock

    def __init__(self, cfg: Config, key: Array):
        key1, key2 = jax.random.split(key, 2)
        self.norm_prev = RMSNorm(cfg)
        self.norm_emb = RMSNorm(cfg)
        self.proj = init(key1, (cfg.dim, cfg.dim * 2))
        self.block = IBlock(cfg, key2)

    def __call__(
        self,
        rep: Array,
        emb: Array,
        freqs_cis: Array,
        mask: Array,
        kvcache: KVCache,
        layer_idx: int,
        cur_pos: int,
    ) -> tuple[Array, KVCache]:
        x = jnp.concat((self.norm_prev(rep), self.norm_emb(emb)), axis=-1)
        x = einsum(self.proj, x, "d c, b t c -> b t d")
        return self.block(x, freqs_cis, mask, kvcache, layer_idx, cur_pos)


class ITransformer(eqx.Module):
    tok_emb: Array
    blocks: list[IBlock]
    mtp_blocks: list[IMTPBlock]
    norm: RMSNorm
    head: Array

    def __init__(self, cfg: Config, key: Array):
        key_emb, key_blocks, key_mtp, key_head = jax.random.split(key, 4)
        block_keys = jax.random.split(key_blocks, cfg.n_blocks)
        mtp_keys = jax.random.split(key_mtp, cfg.n_mtp)

        self.tok_emb = init(key_emb, (cfg.n_vocab, cfg.dim))
        self.blocks = [IBlock(cfg, k) for k in block_keys]
        self.mtp_blocks = [IMTPBlock(cfg, k) for k in mtp_keys]
        self.norm = RMSNorm(cfg)
        self.head = init(key_head, (cfg.dim, cfg.n_vocab))

    def __call__(
        self, toks: Array, freqs_cis: Array, mask: Array, kvcache: KVCache, cur_pos: int
    ) -> tuple[Array, KVCache]:
        emb = jnp.take(self.tok_emb, toks, axis=0)

        x = emb
        for i, block in enumerate(self.blocks):
            x, kvcache = block(x, freqs_cis, mask, kvcache, i, cur_pos)

        preds_stack = [x]
        for i, block in enumerate(self.mtp_blocks):
            next_rep, kvcache = block(
                preds_stack[-1],
                emb,
                freqs_cis,
                mask,
                kvcache,
                i + len(self.blocks) - 1,
                cur_pos,
            )
            preds_stack.append(next_rep)

        x = jnp.stack(preds_stack, axis=0)  # type: ignore

        x = self.norm(x)
        return einsum(x, self.head, "mtp b t d, d nv -> b t mtp nv"), kvcache
