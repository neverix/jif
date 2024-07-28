from typing import Literal
from penzai.experimental.v2.models.transformer import model_parts
from penzai.experimental.v2 import pz
import dataclasses


@dataclasses.dataclass(kw_only=True)
class DiTConfig:
    n_layers: int = 6
    d_model: int = 512

    n_kv_heads: int = 4
    q_rep: int = 2
    qk_dim: int = 64
    v_dim: int = 128

    d_ff: int = 1024
    ff_act: Literal["gelu"] = "gelu"

    cond_dim: int = 256
    freq_embed_dim: int = 256
    time_act: Literal["silu"] = "silu"


@pz.pytree_dataclass(has_implicitly_inherited_fields=True)
class TimestepEmbedder(pz.nn.Sequential):
    ...
