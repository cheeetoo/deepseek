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


def get_nucleus(probs, p):
    idxs = jnp.argsort(-probs, axis=-1)
    sorted_probs = jnp.take_along_axis(probs, idxs, axis=-1)
    mask = jnp.cumsum(sorted_probs, axis=-1) - sorted_probs > p
    return idxs[..., ~mask]


def is_in_nucleus(probs, p, token_id):
    nucleus = get_nucleus(probs, p)
    return jnp.any(nucleus == token_id[..., None], axis=-1)


def find_first_mismatch(ref_probs, p, draft_tokens):
    in_nucleus = is_in_nucleus(ref_probs, p, draft_tokens)
    first_mismatch = jnp.argmax(~in_nucleus, axis=-1)
    all_match = jnp.all(in_nucleus, axis=-1)
    return jnp.where(all_match, draft_tokens.shape[-1], first_mismatch)


def generate(tok_ids, gen_len, kvcache: KVCache) -> tuple[jax.Array, KVCache]:
    start_toks = tok_ids.shape[-1]
    key = jax.random.PRNGKey(0)
    cur_pos = 0
    n_mtp = cfg.n_mtp
    top_p: float = cfg.inference_cfg.top_p  # type: ignore

    while cur_pos < (gen_len + start_toks):
        n_toks = tok_ids.shape[-1] - cur_pos
        new_mask = mask[:n_toks, : tok_ids.shape[-1]]
        new_freqs_cis = freqs_cis[cur_pos : tok_ids.shape[-1], :]

        logits, kvcache = model(
            tok_ids[:, cur_pos:], new_freqs_cis, new_mask, kvcache, cur_pos
        )
        scores = jax.nn.softmax(logits / cfg.inference_cfg.temperature, -1)  # type: ignore

        draft_tokens = []
        for i in range(n_mtp):
            key, subkey = jax.random.split(key)
            draft_token = sample_top_p(scores[:, -1, i, :], top_p, subkey)  # type: ignore
            draft_tokens.append(draft_token)

        draft_tokens_array = jnp.concatenate(
            [t.reshape(cfg.inference_cfg.batch_size, -1) for t in draft_tokens], axis=-1
        )
        draft_ids = jnp.concatenate([tok_ids, draft_tokens_array], axis=-1)

        verify_n_toks = draft_ids.shape[-1] - cur_pos
        verify_mask = mask[:verify_n_toks, : draft_ids.shape[-1]]
        verify_freqs_cis = freqs_cis[cur_pos : draft_ids.shape[-1], :]

        ref_logits, kvcache = model(
            draft_ids[:, cur_pos:], verify_freqs_cis, verify_mask, kvcache, cur_pos
        )
        ref_scores = jax.nn.softmax(ref_logits / cfg.inference_cfg.temperature, -1)[
            :, :, 0, :
        ]  # type: ignore

        first_mismatch = find_first_mismatch(
            ref_scores[:, :-1], top_p, draft_tokens_array[:, :-1]
        )

        accepted_tokens = jnp.concat(
            (tok_ids, draft_tokens_array[:, :first_mismatch]), axis=-1
        )

        if first_mismatch < draft_tokens_array.shape[-1]:
            key, subkey = jax.random.split(key)
            mismatch_token = sample_top_p(ref_scores[:, first_mismatch], top_p, subkey)
            tok_ids = jnp.concat(
                (
                    accepted_tokens,
                    mismatch_token.reshape(cfg.inference_cfg.batch_size, -1),
                ),
                axis=-1,
            )
        else:
            tok_ids = accepted_tokens

        cur_pos += first_mismatch + 1

    return tok_ids, kvcache


st = time.time()
generated_toks, kvcache = generate(tok_ids, 20, kvcache)
et = time.time() - st

toks_per_second = 20 / et
print("new toks: ", enc.decode(list(generated_toks.flatten())))
print("tok/s: ", toks_per_second)

