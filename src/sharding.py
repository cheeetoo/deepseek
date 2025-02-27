import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.tree_util import GetAttrKey


class AxisNames:
    dp = "replicate"
    tp = "data"


devices = jax.devices()
mesh: Mesh = Mesh(np.array(devices).reshape(2, 4), (AxisNames.dp, AxisNames.tp))


def get_partition(path: tuple[GetAttrKey, ...]) -> P:
    name = path[-1].name

    match name:
        case "tok_emb":
            return P(AxisNames.tp, None)
        case "w":
            if "norm" in path[-2].name:
                if len(path) == 2:
                    return P(None)
                else:
                    return P(None, None)
            else:
                return P(None, None, AxisNames.tp)
        case "w_dkv":
            return P(None, None, AxisNames.tp)
        case "w_uk":
            return P(None, None, AxisNames.tp, None)
        case "w_uv":
            return P(None, None, AxisNames.tp, None)
        case "w_dq":
            return P(None, None, AxisNames.tp)
        case "w_uq":
            return P(None, None, AxisNames.tp, None)
        case "w_qr":
            return P(None, None, AxisNames.tp, None)
        case "w_kr":
            return P(None, None, AxisNames.tp)
        case "w_o":
            return P(None, None, None, AxisNames.tp)
        case "w1_shared":
            return P(None, None, AxisNames.tp)
        case "w2_shared":
            return P(None, AxisNames.tp, None)
        case "w3_shared":
            return P(None, None, AxisNames.tp)
        case "w1_routed":
            return P(None, AxisNames.tp, None, None)
        case "w2_routed":
            return P(None, AxisNames.tp, None, None)
        case "w3_routed":
            return P(None, AxisNames.tp, None, None)
        case "gate_b":
            return P(None, None)
        case "proj":
            return P(None, AxisNames.tp, None)
        case "head":
            return P(AxisNames.tp, None)
        case _:
            raise ValueError(f"Unrecognized weight: {name}")


def shard_model(model):
    return jax.tree_util.tree_map_with_path(
        lambda p, v: jax.device_put(v, NamedSharding(mesh, get_partition(p))), model
    )
