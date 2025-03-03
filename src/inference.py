import time
import equinox as eqx
import jax
import jax.numpy as jnp
import tiktoken

from model import precompute_yarn_freqs_cis
from model_inference import ITransformer, KVCache
from utils import Config, InferenceConfig

cfg = Config(
    dim=1024,
    dc=512,
    dim_nope_head=64,
    dim_rope_head=32,
    n_heads=32,
    moe_inter_dim=256,
    n_shared_experts=2,
    n_routed_experts=64,
    n_activated_experts=6,
    max_seqlen=128,
    rope_theta=10_000,
    eps=1e-6,
    n_vocab=50260,
    n_blocks=5,
    n_mtp=1,
    batch_size=128,
    mtp_lambda=0.3,
    bias_update_rate=0.001,
    aux_alpha=0.0001,
    inference_cfg=InferenceConfig(
        max_seqlen=128 * 4,
        beta_fast=32,
        beta_slow=1,
        rope_factor=40,
        temperature=0.6,
        top_p=0.9,
        batch_size=1,
    ),
)

enc = tiktoken.get_encoding("gpt2")

MESSAGE = "TEST TEST TEST"

model = ITransformer(cfg, jax.random.PRNGKey(0))
with jax.default_device(jax.devices("cpu")[0]):
    model = ITransformer(cfg, jax.random.PRNGKey(0))
    model = eqx.tree_deserialise_leaves("./model.eqx", model)

kvcache = KVCache.new(
    cfg.n_blocks + cfg.n_mtp,
    cfg.inference_cfg.batch_size,  # type: ignore
    cfg.inference_cfg.max_seqlen,  # type: ignore
    cfg.dim_rope_head,
    cfg.dc,
)
freqs_cis = precompute_yarn_freqs_cis(cfg)

tok_ids = jnp.array(enc.encode(MESSAGE)).reshape(1, -1)

max_seqlen = cfg.inference_cfg.max_seqlen  # type: ignore
mask = jnp.triu(jnp.full((max_seqlen, max_seqlen), -jnp.inf, dtype=jnp.bfloat16), k=1)


def sample_top_p(probs, p, key):
    idxs = jnp.argsort(-probs, axis=-1)
    sorted_probs = jnp.take_along_axis(probs, idxs, axis=-1)
    mask = jnp.cumsum(sorted_probs, axis=-1) - sorted_probs > p
    filtered = jnp.where(mask, 0.0, sorted_probs)
    norm_probs = filtered / jnp.sum(filtered, axis=-1, keepdims=True)
    sample = jax.random.categorical(key, jnp.log(jnp.maximum(norm_probs, 1e-8)))[
        ..., None
    ]
    return jnp.take_along_axis(idxs, sample, axis=-1)


def generate(tok_ids, gen_len) -> jax.Array:
    key = jax.random.PRNGKey(0)
    cur_pos = 0

    for _ in range(gen_len):
        n_toks = tok_ids.shape[-1] - cur_pos
        new_mask = mask[:n_toks, :n_toks]
        new_freqs_cis = freqs_cis[:n_toks, :]
        logits, kvcache = model(
            tok_ids[:, cur_pos:], new_freqs_cis, new_mask, kvcache, cur_pos
        )
        scores = jax.nn.softmax(logits[:, -1, 0, :] / cfg.inference_cfg.temperature, -1)  # type: ignore
        next_token = sample_top_p(scores, cfg.inference_cfg.top_p, key)  # type: ignore
        tok_ids = jnp.concat((tok_ids, next_token), axis=-1)
        key, _ = jax.random.split(key)
        cur_pos += n_toks
    return tok_ids


st = time.time()
generated_toks = generate(tok_ids, 20)
et = time.time() - st

toks_per_second = 20 / et
print(f"{tok_ids.shape=} {generated_toks.shape=}")
print("new toks: ", enc.decode(list(generated_toks.flatten())))
print("tok/s: ", toks_per_second)
