from functools import partial
import time
import argparse

import jax
import jax.numpy as jnp
from jax import Array
from jax.sharding import NamedSharding, PartitionSpec as P
import equinox as eqx
from einops import rearrange, reduce

from model import Transformer, precompute_rope_freqs_cis

from sharding import shard_model, mesh
from sharding import AxisNames
from utils import Config, DataLoader, adamw, get_adamw_state


def parse_args():
    parser = argparse.ArgumentParser(description="Transformer pretraining script")

    # Model configuration
    parser.add_argument("--dim", type=int, default=1024, help="Model dimension")
    parser.add_argument("--dc", type=int, default=512, help="DC dimension")
    parser.add_argument(
        "--dim_nope_head", type=int, default=64, help="Dimension per nope head"
    )
    parser.add_argument(
        "--dim_rope_head", type=int, default=32, help="Dimension per rope head"
    )
    parser.add_argument("--n_heads", type=int, default=32, help="Number of heads")
    parser.add_argument(
        "--moe_inter_dim", type=int, default=256, help="MoE intermediate dimension"
    )
    parser.add_argument(
        "--n_shared_experts", type=int, default=2, help="Number of shared experts"
    )
    parser.add_argument(
        "--n_routed_experts", type=int, default=64, help="Number of routed experts"
    )
    parser.add_argument(
        "--n_activated_experts", type=int, default=6, help="Number of activated experts"
    )
    parser.add_argument(
        "--max_seqlen", type=int, default=128, help="Maximum sequence length"
    )
    parser.add_argument(
        "--rope_theta", type=int, default=10_000, help="RoPE theta parameter"
    )
    parser.add_argument("--eps", type=float, default=1e-6, help="Epsilon parameter")
    parser.add_argument("--n_vocab", type=int, default=50260, help="Vocabulary size")
    parser.add_argument("--n_blocks", type=int, default=5, help="Number of blocks")
    parser.add_argument("--n_mtp", type=int, default=1, help="Number of MTP layers")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument(
        "--mtp_lambda", type=float, default=0.3, help="MTP lambda parameter"
    )
    parser.add_argument(
        "--bias_update_rate", type=float, default=0.001, help="Bias update rate"
    )
    parser.add_argument(
        "--aux_alpha", type=float, default=0.0001, help="Auxiliary loss alpha"
    )

    # Training parameters
    parser.add_argument(
        "--steps", type=int, default=10_000, help="Number of training steps"
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--wd", type=float, default=1e-5, help="Weight decay")
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/edu_fineweb10B/edu_fineweb_train_00000*",
        help="Path to training data",
    )

    # Model saving
    parser.add_argument(
        "--save_path",
        type=str,
        default="./model.eqx",
        help="Path to save the trained model",
    )

    return parser.parse_args()


args = parse_args()
cfg = Config(
    dim=args.dim,
    dc=args.dc,
    dim_nope_head=args.dim_nope_head,
    dim_rope_head=args.dim_rope_head,
    n_heads=args.n_heads,
    moe_inter_dim=args.moe_inter_dim,
    n_shared_experts=args.n_shared_experts,
    n_routed_experts=args.n_routed_experts,
    n_activated_experts=args.n_activated_experts,
    max_seqlen=args.max_seqlen,
    rope_theta=args.rope_theta,
    eps=args.eps,
    n_vocab=args.n_vocab,
    n_blocks=args.n_blocks,
    n_mtp=args.n_mtp,
    batch_size=args.batch_size,
    mtp_lambda=args.mtp_lambda,
    bias_update_rate=args.bias_update_rate,
    aux_alpha=args.aux_alpha,
    inference_cfg=None,
)

with jax.default_device(jax.devices("cpu")[0]):
    model = Transformer(cfg, jax.random.PRNGKey(0))
model = shard_model(model)

model_shardings = jax.tree.map(lambda x: x.sharding, model)
inp_sharding = NamedSharding(mesh, P(AxisNames.dp, None))

m, v = get_adamw_state(model)

freqs_cis = precompute_rope_freqs_cis(cfg)
mask = jnp.triu(
    jnp.full((cfg.max_seqlen, cfg.max_seqlen), -jnp.inf, dtype=jnp.bfloat16), k=1
)

gate_b_filter_spec = jax.tree_util.tree_map_with_path(
    lambda p, _: p[-1].name != "gate_b", model
)

loader = DataLoader(
    args.data_path,
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
        toks_per_expert.astype(jnp.bfloat16),
        "(gates b t) ne -> gates ne",
        "mean",
        gates=cfg.n_blocks + cfg.n_mtp,
        b=cfg.batch_size,
    )

    f = (cfg.n_routed_experts / cfg.n_activated_experts) * toks_per_expert
    aux_loss = cfg.aux_alpha * jnp.sum(f * p)

    return l_ce + aux_loss, toks_per_expert


@partial(
    jax.jit,
    in_shardings=(  # type: ignore
        model_shardings,
        model_shardings,
        model_shardings,
        inp_sharding,
        None,
        None,
    ),
    out_shardings=(model_shardings, model_shardings, model_shardings, None),  # type: ignore
)
def train_step(model, m, v, x, y, t):
    (loss, toks_per_expert), grads = loss_fn(model, x, y)

    model, gate_biases = eqx.partition(model, gate_b_filter_spec)

    model, m, v = adamw(model, grads, m, v, t, wd=args.wd, lr=args.lr)

    flat_gate_bs, treedef = jax.tree.flatten(gate_biases)

    e = jnp.mean(toks_per_expert, axis=-1, keepdims=True) - toks_per_expert
    update = cfg.bias_update_rate * jnp.sign(e)
    flat_gate_bs = [b + u for b, u in zip(flat_gate_bs, update)]
    gate_biases = jax.tree.unflatten(treedef, flat_gate_bs)

    model = eqx.combine(model, gate_biases)
    return model, m, v, loss


running_loss = 0
running_times = 0
tflops = None

for t in range(0, args.steps):
    print(f"step {t}")
    x, y = loader.next_batch()
    x = jax.device_put(x, inp_sharding)

    st = time.monotonic()
    model, m, v, loss = train_step(model, m, v, x, y, t + 1)
    et = time.monotonic() - st

    running_loss += loss
    running_times += et
    if t % 10 == 0:
        if tflops is None:
            compiled = train_step.lower(model, m, v, x, y, t).compile()
            tflops = compiled.cost_analysis()["flops"] / 1e12  # type: ignore

        avg_time = running_times / 10
        achieved_tflops = tflops / avg_time
        mfu = (achieved_tflops / 492) * 100

        print(f"Loss: {running_loss / 10}, MFU: {mfu:.4f}%")
        running_loss = 0
        running_times = 0

print(f"Training complete. Saving model to {args.save_path}...")
with jax.default_device(jax.devices("cpu")[0]):
    cpu_model = jax.tree.map(lambda x: jax.device_get(x), model)
eqx.tree_serialise_leaves(args.save_path, cpu_model)
