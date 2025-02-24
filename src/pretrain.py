from functools import partial

import jax
import jax.numpy as jnp
from jax import Array
import equinox as eqx
from einops import rearrange, reduce

from model import Transformer, precompute_freqs_cis
from utils import Config, DataLoader, adamw, get_adamw_state

# TODO: set real values for run
cfg = Config(
    dim=4,
    dc=2,
    dim_head=2,
    dim_rope_head=2,
    n_heads=4,
    moe_inter_dim=8,
    n_shared_experts=1,
    n_routed_experts=8,
    n_activated_experts=2,
    max_seqlen=4,
    rope_theta=4000,
    eps=1e-6,
    n_vocab=50257,
    n_blocks=4,
    n_mtp=2,
    batch_size=2,
    mtp_lambda=0.001,
    bias_update_rate=0.001,
    aux_alpha=0.001,
)


model = Transformer(cfg, jax.random.PRNGKey(23))
m, v = get_adamw_state(model)
freqs_cis = precompute_freqs_cis(cfg)
mask = jnp.triu(jnp.full((cfg.max_seqlen, cfg.max_seqlen), -jnp.inf), k=1)

gate_b_filter_spec = jax.tree_util.tree_map_with_path(
    lambda p, _: p[-1].name != "gate_b", model
)

loader = DataLoader(
    "./data/tinyshakespeare/tiny_shakespeare_train.bin",
    cfg.batch_size,
    cfg.max_seqlen,
    cfg.n_mtp,
)


@partial(jax.value_and_grad, has_aux=True)
def loss_fn(model: Transformer, x: Array, y: Array):
    logits, affinities = model(x, freqs_cis, mask)
    pred = jax.nn.softmax(logits, axis=-1)  # b, t, mtp, nv
    labels = jax.nn.one_hot(y, num_classes=cfg.n_vocab)
    l_ce = -(labels * jnp.log(pred)).sum(-1)
    l_ce = l_ce * jnp.array([1] + [cfg.mtp_lambda / cfg.n_mtp] * cfg.n_mtp)
    l_ce = l_ce.mean()

    s_prime = affinities / affinities.sum(axis=-1, keepdims=True)  # gates, b, t, ne

    p = reduce(s_prime, "gates b t ne -> gates ne", "mean")

    _, indices = jax.lax.top_k(affinities, cfg.n_activated_experts)
    toks_per_expert = jax.vmap(lambda x: jnp.bincount(x, length=cfg.n_routed_experts))(
        rearrange(indices, "gates b t topk -> (gates b t) topk")
    )
    toks_per_expert = reduce(
        toks_per_expert.astype(float),
        "(gates b t) ne -> gates ne",
        "mean",
        gates=cfg.n_blocks + cfg.n_mtp,
        b=cfg.batch_size,
    )

    f = (cfg.n_routed_experts / cfg.n_activated_experts) * toks_per_expert

    aux_loss = cfg.aux_alpha * jnp.sum(f * p)
    jax.debug.print("ce {l_mpt} aux {aux}", l_mpt=l_ce, aux=aux_loss)

    return l_ce + aux_loss, toks_per_expert


# @jax.jit
def train_step(model, m, v, x, y, t):
    (loss, toks_per_expert), grads = loss_fn(model, x, y)

    model, gate_biases = eqx.partition(model, gate_b_filter_spec)

    model, m, v = adamw(model, grads, m, v, t, wd=1e-5, lr=3e-4)

    flat_gate_bs, treedef = jax.tree.flatten(gate_biases)

    e = jnp.mean(toks_per_expert, axis=-1, keepdims=True) - toks_per_expert
    update = cfg.bias_update_rate * jnp.sign(e)
    flat_gate_bs = [b + u for b, u in zip(flat_gate_bs, update)]
    gate_biases = jax.tree.unflatten(treedef, flat_gate_bs)

    model = eqx.combine(model, gate_biases)
    return model, m, v, loss


STEPS = 3

for t in range(1, STEPS - 1):
    x, y = loader.next_batch()
    model, m, v, loss = train_step(model, m, v, x, y, t)
    print(loss)
