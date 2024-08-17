# based on https://github.com/neverix/Score-Entropy-Discrete-Diffusion/blob/main/model/transformer.py

import dataclasses
import math
from typing import Literal, Optional

import jax
import jax.numpy as jnp
from penzai import pz
from penzai.models.transformer import model_parts
from penzai.nn.linear_and_affine import constant_initializer, zero_initializer

ACT_FN_MAP = {"silu": jax.nn.silu, "gelu": jax.nn.gelu}


@dataclasses.dataclass(kw_only=True)
class DiTConfig:
    vocab_size: int
    n_layers: int = 6
    d_model: int = 512

    n_kv_heads: int = 12
    q_rep: int = 1
    qk_dim: int = 64
    v_dim: int = 64

    d_ff: int = 1536
    ff_act: Literal["gelu"] = "gelu"

    cond_dim: int = 128
    freq_embed_dim: int = 128
    time_act: Literal["silu"] = "silu"

    epsilon: float = 1e-5
    
    act_dtype: str = "bfloat16"
    res_dtype: str = "float32"
    param_dtype: str = "bfloat16"
    ln_dtype: str = "float32"
    rope_wavelength: float = 10_000.0

    @property
    def activation_dtype(self):
        return getattr(jnp, self.act_dtype)

    @property
    def resid_dtype(self):
        return getattr(jnp, self.res_dtype)

    @property
    def parameter_dtype(self):
        return getattr(jnp, self.param_dtype)

    @property
    def layernorm_dtype(self):
        return getattr(jnp, self.ln_dtype)


@pz.pytree_dataclass
class AdaLNCondition(pz.nn.Layer):
    projection_scale: pz.nn.Affine
    projection_bias: Optional[pz.nn.Affine]

    @classmethod
    def from_config(cls, config: DiTConfig, init_base_rng: jax.Array | None, name: str, use_bias: bool = True):
        scale_affine = pz.nn.Affine.from_config(
            init_base_rng=init_base_rng,
            input_axes={"cond_embedding": config.cond_dim},
            output_axes={"embedding": config.d_model},
            name=f"{name}/projection_scale",
            dtype=config.parameter_dtype,
            linear_initializer=zero_initializer,
            bias_initializer=constant_initializer(1.0),
        )
        bias_affine = None
        if use_bias:
            bias_affine = pz.nn.Affine.from_config(
                init_base_rng=init_base_rng,
                input_axes={"cond_embedding": config.cond_dim},
                output_axes={"embedding": config.d_model},
                name=f"{name}/projection_bias",
                dtype=config.parameter_dtype,
                linear_initializer=zero_initializer,
                bias_initializer=zero_initializer,
            )
        return cls(
            projection_scale=scale_affine,
            projection_bias=bias_affine,
        )

    def __call__(self, arg, **side_inputs):
        cond = side_inputs["timestep_cond"]
        return arg * self.projection_scale(cond) + (self.projection_bias(cond) if self.projection_bias is not None else 0)


@pz.pytree_dataclass(has_implicitly_inherited_fields=True)
class AdaLN(pz.nn.Sequential):
    @classmethod
    def wrap_with_config(cls, config: DiTConfig, init_base_rng: jax.Array | None, name: str, child: pz.nn.Layer, scale_output: bool = True):
        return cls([
            pz.nn.CastToDType(dtype=config.layernorm_dtype),
            pz.nn.RMSStandardize(across=("embedding",), epsilon=config.epsilon),
            pz.nn.CastToDType(dtype=config.activation_dtype),
            AdaLNCondition.from_config(config, init_base_rng=init_base_rng, name=f"{name}/adaln_input", use_bias=True),
            child,
        ] + ([AdaLNCondition.from_config(config, init_base_rng=init_base_rng, name=f"{name}/adaln_output", use_bias=False)] if scale_output else [])
          + [pz.nn.CastToDType(dtype=config.resid_dtype)])


@pz.pytree_dataclass
class DebugPrint(pz.nn.Layer):
    def __call__(self, arg):
        import random
        j = random.randrange(0, arg.named_shape["kv_heads"])
        jax.debug.print("{}", arg[{"batch": 0, "q_rep": 0, "kv_heads": j}].unwrap("seq", "kv_seq"))
        return arg


def build_dit_attn(name: str, init_base_rng: jax.Array | None, config: DiTConfig):
    hidden_size = config.d_model
    num_heads = config.n_kv_heads
    q_rep = config.q_rep
    qk_dim = config.qk_dim
    v_dim = config.v_dim
    return pz.nn.Attention(
        input_to_query=pz.nn.Sequential([
            pz.nn.Affine.from_config(
                input_axes={"embedding": hidden_size},
                output_axes={
                    "kv_heads": num_heads,
                    "q_rep": q_rep,
                    "qk_dim": qk_dim,
                },
                dtype=config.parameter_dtype,
                name=f"{name}/query_proj",
                init_base_rng=init_base_rng,
            ),
            pz.nn.ApplyRoPE(
                positions_input_name="positions",
                embedding_axis="qk_dim",
                max_wavelength=config.rope_wavelength,
            ),
            pz.nn.ConstantRescale(
                by=jnp.array(
                    qk_dim**-0.5, dtype=config.activation_dtype
                )
            ),
        ]),
        input_to_key=pz.nn.Sequential([
            pz.nn.Affine.from_config(
                input_axes={"embedding": hidden_size},
                output_axes={
                    "kv_heads": num_heads,
                    "qk_dim": qk_dim,
                },
                dtype=config.parameter_dtype,
                name=f"{name}/key_proj",
                init_base_rng=init_base_rng,
            ),
            pz.nn.ApplyRoPE(
                positions_input_name="positions",
                embedding_axis="qk_dim",
                max_wavelength=config.rope_wavelength,
            ),
            pz.nn.CastToDType(config.activation_dtype),
        ]),
        input_to_value=pz.nn.Sequential([
            pz.nn.Affine.from_config(
                input_axes={"embedding": hidden_size},
                output_axes={
                    "kv_heads": num_heads,
                    "v_dim": v_dim,
                },
                dtype=config.parameter_dtype,
                name=f"{name}/value_proj",
                init_base_rng=init_base_rng,
            ),
            pz.nn.CastToDType(config.activation_dtype),
        ]),
        query_key_to_attn=pz.nn.Sequential([
            pz.nn.NamedEinsum(
                (
                    {"seq": "tq", "kv_heads": "h", "q_rep": "r", "qk_dim": "p"},
                    {"seq": "tkv", "kv_heads": "h", "qk_dim": "p"},
                ),
                {"seq": "tq", "kv_heads": "h", "q_rep": "r", "kv_seq": "tkv"},
            ),
            pz.nn.ApplyExplicitAttentionMask(
                mask_input_name="attention_mask",
                masked_out_value=jnp.array(
                    jnp.finfo(config.activation_dtype).min, dtype=config.activation_dtype
                ),
            ),
            pz.nn.Softmax("kv_seq"),
            # DebugPrint(),
        ]),
        attn_value_to_output=pz.nn.Sequential([
            pz.nn.NamedEinsum(
                (
                    {"seq": "tq", "kv_heads": "h", "q_rep": "r", "kv_seq": "tkv"},
                    {"seq": "tkv", "kv_heads": "h", "v_dim": "p"},
                ),
                {"seq": "tq", "kv_heads": "h", "q_rep": "r", "v_dim": "p"},
            ),
            pz.nn.Affine.from_config(
                input_axes={
                    "kv_heads": num_heads,
                    "q_rep": q_rep,
                    "v_dim": v_dim,
                },
                output_axes={"embedding": hidden_size},
                dtype=config.parameter_dtype,
                name=f"{name}/output_proj",
                init_base_rng=init_base_rng,
            ),
        ]),
    )


def build_dit_ff(name: str, init_base_rng: jax.Array | None, config: DiTConfig):
    return pz.nn.Sequential([
        pz.nn.Affine.from_config(
            init_base_rng=init_base_rng,
            input_axes={"embedding": config.d_model},
            output_axes={"neurons": config.d_ff},
            name=f"{name}/in_linear",
            dtype=config.parameter_dtype,
        ),
        pz.nn.Elementwise(ACT_FN_MAP[config.ff_act]),
        pz.nn.Affine.from_config(
            init_base_rng=init_base_rng,
            input_axes={"neurons": config.d_ff},
            output_axes={"embedding": config.d_model},
            name=f"{name}/out_linear",
            dtype=config.parameter_dtype,
        ),
    ])


def build_dit_block(name: str, init_base_rng: jax.Array | None, config: DiTConfig):
    return model_parts.TransformerBlock(sublayers=[
        pz.nn.Residual(AdaLN.wrap_with_config(
            config=config,
            init_base_rng=init_base_rng,
            name=f"{name}/adaln_attn",
            child=build_dit_attn(name=f"{name}/attn", init_base_rng=init_base_rng, config=config),
        )),
        pz.nn.Residual(AdaLN.wrap_with_config(
            config=config,
            init_base_rng=init_base_rng,
            name=f"{name}/adaln_ff",
            child=build_dit_ff(name=f"{name}/ff", init_base_rng=init_base_rng, config=config),
        )),
    ])


def build_dit_model(config: DiTConfig, init_base_rng: jax.Array | None, name: str = "dit_model"):
    return pz.nn.Sequential([
        pz.nn.EmbeddingLookup(
            pz.nn.EmbeddingTable.from_config(
                name=f"{name}/embedder",
                init_base_rng=init_base_rng,
                vocab_size=config.vocab_size + 1,
                embedding_axes={"embedding": config.d_model},
                dtype=config.parameter_dtype,
            ),
        ),
        pz.nn.CastToDType(dtype=config.resid_dtype),
        pz.nn.LayerStack.from_sublayer_builder(
            builder=build_dit_block,
            stack_axis="blocks",
            stack_axis_size=config.n_layers,
            init_base_rng=init_base_rng,
            builder_kwargs=dict(name=f"{name}/blocks", config=config),
        ),
        pz.nn.CastToDType(dtype=jnp.float32),
        AdaLN.wrap_with_config(
            config=config,
            init_base_rng=init_base_rng,
            name=f"{name}/final_adaln",
            child=pz.nn.Affine.from_config(
                init_base_rng=init_base_rng,
                input_axes={"embedding": config.d_model},
                output_axes={"vocabulary": config.vocab_size},
                name=f"{name}/unembedder",
                dtype=config.parameter_dtype,
            ),
            scale_output=False
        )
    ])


def timestep_embedding(t, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                        These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
    half = dim // 2
    freqs = jnp.exp(
        -math.log(max_period) * jnp.arange(0, half, dtype=jnp.float32) / half
    )
    args = t.astype(jnp.float32) * freqs
    embedding = pz.nx.nmap(lambda x: jnp.concatenate([jnp.cos(x), jnp.sin(x)], axis=-1))(args)
    if dim % 2:
        embedding = jnp.concatnate([embedding, jnp.zeros_like(embedding[..., :1])], axis=-1)
    return embedding.tag("timestep_embedding")


@pz.pytree_dataclass
class DitWithTimestep(pz.nn.Layer):
    cond_dim: int = dataclasses.field(metadata={"pytree_node": False})
    freq_embed_dim: int = dataclasses.field(metadata={"pytree_node": False})
    config: DiTConfig = dataclasses.field(metadata={"pytree_node": False})
    timestep_mlp: pz.nn.Layer
    model: pz.nn.Layer

    @classmethod
    def from_config(cls, config: DiTConfig, init_base_rng: jax.Array | None, name: str = "dit"):
        return cls(
            cond_dim=config.cond_dim,
            freq_embed_dim=config.freq_embed_dim,
            timestep_mlp=pz.nn.Sequential([
                pz.nn.Affine.from_config(init_base_rng=init_base_rng,
                                         input_axes={"timestep_embedding": config.freq_embed_dim},
                                         output_axes={"cond_embedding": config.cond_dim},
                                         name=f"{name}/timestep_mlp/in_linear"),
                pz.nn.Elementwise(ACT_FN_MAP[config.time_act]),
                pz.nn.Affine.from_config(init_base_rng=init_base_rng,
                                         input_axes={"cond_embedding": config.cond_dim},
                                         output_axes={"cond_embedding": config.cond_dim},
                                         name=f"{name}/timestep_mlp/out_linear"),
            ]),
            model=build_dit_model(config, init_base_rng=init_base_rng, name=name),
            config=config,
        )

    def __call__(self, arg, **side_inputs):
        t = side_inputs["timestep"]
        t_embed = timestep_embedding(t, self.freq_embed_dim)
        timestep_cond = self.timestep_mlp(t_embed).astype(self.config.parameter_dtype)
        return self.model(arg, **{**side_inputs, "timestep_cond": timestep_cond})

    @classmethod
    def wrap_inputs(cls, x, mask_token, mask=None, t=None):
        if mask is None:
            mask = jnp.full_like(x, True)
        if t is None:
            t = ((x == mask_token) & mask).mean(axis=-1) / jnp.mean(mask, axis=-1)
        positions = pz.nx.wrap(jnp.arange(x.shape[-1]), "seq")
        x = pz.nx.wrap(x, "batch", "seq")
        t = pz.nx.wrap(t, "batch")
        attention_mask = pz.nx.wrap(mask, "batch", "seq")
        return x, dict(timestep=t, positions=positions, attention_mask=attention_mask)


if __name__ == "__main__":
    config = DiTConfig(vocab_size=128)
    model = DitWithTimestep.from_config(config, jax.random.PRNGKey(0))
    x = jnp.zeros((4, 16), jnp.int32)
    t = jnp.zeros((4,), jnp.float32)
    positions = jnp.arange(16)
    y = model(pz.nx.wrap(x, "batch", "seq"), timestep=pz.nx.wrap(t, "batch"), positions=pz.nx.wrap(positions, "seq"))
    print(y)
