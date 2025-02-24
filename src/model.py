import equinox as eqx
import jax
import jax.numpy as jnp
import jax.nn as nn
from einops import einsum, rearrange, reduce, repeat
from jax import Array

from utils import Config, init


def precompute_freqs_cis(cfg: Config) -> Array:
    dim = cfg.dim_rope_head

    freqs = 1.0 / (cfg.rope_theta ** (jnp.arange(0, dim, 2) / dim))
    freqs = jnp.outer(jnp.arange(cfg.max_seqlen), freqs)

    return jnp.exp(1j * freqs)


def rope(h: Array, freqs_cis: Array) -> Array:
    dtype = h.dtype
    real, imag = jnp.split(h.astype(jnp.float32), 2, -1)
    res = (real + 1j * imag) * freqs_cis
    return jnp.concat((res.real, res.imag), axis=-1).astype(dtype)


class MLA(eqx.Module):
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
        self.dh = cfg.dim_head
        self.drh = cfg.dim_rope_head

        key_dkv, key_uk, key_uv, key_dq, key_uq, key_qr, key_kr, key_wo = (
            jax.random.split(key, 8)
        )

        self.w_dkv = init(key_dkv, (cfg.dc, cfg.dim))
        self.w_uk = init(key_uk, (cfg.dim_head, cfg.n_heads, cfg.dc))
        self.w_uv = init(key_uv, (cfg.dim_head, cfg.n_heads, cfg.dc))
        self.w_dq = init(key_dq, (cfg.dc, cfg.dim))
        self.w_uq = init(key_uq, (cfg.dim_head, cfg.n_heads, cfg.dc))
        self.w_qr = init(key_qr, (cfg.dim_rope_head, cfg.n_heads, cfg.dc))
        self.w_kr = init(key_kr, (cfg.dim_rope_head, cfg.dim))
        self.w_o = init(key_wo, (cfg.dim, cfg.dim_head, cfg.n_heads))

    def __call__(self, h: Array, freqs_cis: Array, mask: Array):
        c_kv = einsum(self.w_dkv, h, "dc d, b t d -> b t dc")
        k_c = einsum(self.w_uk, c_kv, "dh nh dc, b t dc -> b nh t dh")
        v_c = einsum(self.w_uv, c_kv, "dh nh dc, b t dc -> b nh t dh")

        c_q = einsum(self.w_dq, h, "dc d, b t d -> b t dc")
        q_c = einsum(self.w_uq, c_q, "dh nh dc, b t dc -> b nh t dh")

        q_r = rope(einsum(self.w_qr, c_q, "drh nh dc, b t dc -> b nh t drh"), freqs_cis)
        k_r = rope(einsum(self.w_kr, h, "drh d, b t d -> b t drh"), freqs_cis)

        q = jnp.concat((q_c, q_r), axis=-1)
        k_r = repeat(k_r, "b t drh -> b nh t drh", nh=self.nh)
        k = jnp.concat((k_c, k_r), axis=-1)

        logits = einsum(q, k, "b nh t d, b nh l d -> b nh t l")
        logits = logits / jnp.sqrt(self.dh + self.drh) + mask
        scores = jax.nn.softmax(logits.astype(jnp.float32), -1).astype(h.dtype)

        out = einsum(scores, v_c, "b nh t l, b nh t dh -> b nh l dh")

        return einsum(self.w_o, out, "d dh nh, b nh t dh -> b t d")


class Gate(eqx.Module):
    w: Array

    gate_b: Array

    top_k: int = eqx.field(static=True)
    n_experts: int = eqx.field(static=True)

    def __init__(self, cfg: Config, key: Array):
        self.top_k = cfg.n_activated_experts
        self.n_experts = cfg.n_routed_experts

        self.w = init(key, (cfg.n_routed_experts, cfg.dim))
        self.gate_b = jnp.zeros(cfg.n_routed_experts, jnp.bfloat16)

    def __call__(self, x: Array) -> tuple[Array, Array, Array]:
        logits = einsum(self.w, x, "ne d, b t d -> b t ne")
        scores = nn.sigmoid(logits) + self.gate_b
        weights, indices = jax.lax.top_k(scores, self.top_k)
        weights /= weights.sum(-1, keepdims=True)
        return (weights, indices, scores)


def swiglu(x: Array, w1: Array, w2: Array, w3: Array) -> Array:
    return nn.silu((x @ w1) * (x @ w3)) @ w2


class FFN(eqx.Module):
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
        self.gate = Gate(cfg, key7)

    def __call__(self, x: Array) -> tuple[Array, Array]:
        B, T, _ = x.shape

        weights, indices, affinities = self.gate(x)
        weights = rearrange(weights, "b t tk -> (b t tk)")
        indices = rearrange(indices, "b t tk -> (b t tk)")

        x_flat = repeat(x, "b t d -> (b t ne) d", ne=self.n_activated_experts)

        out = jax.vmap(
            lambda i, w, x: w
            * swiglu(x, self.w1_routed[i], self.w2_routed[i], self.w3_routed[i])
        )(indices, weights, x_flat)  # (b t ne) 1 d

        out = reduce(out, "(b t ne) d -> b t d", b=B, t=T, reduction="sum")

        out += swiglu(x, self.w1_shared, self.w2_shared, self.w3_shared)

        return out, affinities


class RMSNorm(eqx.Module):
    w: Array
    eps: float = eqx.field(static=True)

    def __init__(self, cfg: Config):
        self.w = jnp.ones(cfg.dim)
        self.eps = cfg.eps

    def __call__(self, x: Array) -> Array:
        return (
            self.w
            * x.astype(jnp.float32)
            * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + self.eps)
        ).astype(x.dtype)


class Block(eqx.Module):
    attn_norm: RMSNorm
    attn: MLA
    ffn_norm: RMSNorm
    ffn: FFN

    def __init__(self, cfg: Config, key: Array):
        key_attn, key_ffn = jax.random.split(key, 2)
        self.attn_norm = RMSNorm(cfg)
        self.attn = MLA(cfg, key_attn)
        self.ffn_norm = RMSNorm(cfg)
        self.ffn = FFN(cfg, key_ffn)

    def __call__(self, x: Array, freqs_cis: Array, mask: Array) -> tuple[Array, Array]:
        x = x + self.attn(self.attn_norm(x), freqs_cis, mask)
        ffn_out, affinities = self.ffn(self.ffn_norm(x))
        x = x + ffn_out
        return (x, affinities)


class MTPBlock(eqx.Module):
    norm_prev: RMSNorm
    norm_emb: RMSNorm
    proj: Array
    block: Block

    def __init__(self, cfg: Config, key: Array):
        key1, key2 = jax.random.split(key, 2)
        self.norm_prev = RMSNorm(cfg)
        self.norm_emb = RMSNorm(cfg)
        self.proj = init(key1, (cfg.dim, cfg.dim * 2))
        self.block = Block(cfg, key2)

    def __call__(
        self, rep: Array, emb: Array, freqs_cis: Array, mask: Array
    ) -> tuple[Array, Array]:
        x = jnp.concat((self.norm_prev(rep), self.norm_emb(emb)), axis=-1)
        x = einsum(self.proj, x, "d c, b t c -> b t d")
        x, affinities = self.block(x, freqs_cis, mask)
        return (x, affinities)


class Transformer(eqx.Module):
    tok_emb: Array
    blocks: list[Block]
    mtp_blocks: list[MTPBlock]
    norm: RMSNorm
    head: Array

    def __init__(self, cfg: Config, key: Array):
        key_emb, key_blocks, key_mtp, key_head = jax.random.split(key, 4)
        block_keys = jax.random.split(key_blocks, cfg.n_blocks)
        mtp_keys = jax.random.split(key_mtp, cfg.n_mtp)

        self.tok_emb = init(key_emb, (cfg.n_vocab, cfg.dim))
        self.blocks = eqx.filter_vmap(lambda k: Block(cfg, k))(block_keys)
        self.mtp_blocks = eqx.filter_vmap(lambda k: MTPBlock(cfg, k))(mtp_keys)
        self.norm = RMSNorm(cfg)
        self.head = init(key_head, (cfg.dim, cfg.n_vocab))

    def __call__(
        self, toks: Array, freqs_cis: Array, mask: Array
    ) -> tuple[Array, Array]:
        emb = self.tok_emb[toks]

        def block_step(x: Array, block: Block) -> tuple[Array, Array]:
            x, affinities = block(x, freqs_cis, mask)
            return x, affinities

        x, affinities_block = jax.lax.scan(block_step, emb, self.blocks)  # type: ignore

        def mtp_step(
            rep: Array, mtp_block: MTPBlock
        ) -> tuple[Array, tuple[Array, Array]]:
            next_rep, affinities = mtp_block(rep, emb, freqs_cis, mask)
            return next_rep, (next_rep, affinities)

        _, (preds_stack, affinities_mtp) = jax.lax.scan(mtp_step, x, self.mtp_blocks)  # type: ignore

        x = jnp.concat([x[None, ...], preds_stack], axis=0)  # type: ignore

        affinities = jnp.concat((affinities_block, affinities_mtp), axis=0)

        x = self.norm(x)
        return einsum(x, self.head, "mtp b t d, d nv -> b t mtp nv"), affinities
