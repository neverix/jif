import optax
from typing import NamedTuple
import jax
import jax.numpy as jnp
from optax._src import base
from dataclasses import dataclass, replace
import equinox as eqx
import numpy as np


def dct_matrix(n):
    steps = 0.5 + jnp.arange(n)
    return jnp.sqrt(2/n) * jnp.cos(jnp.pi * steps * steps[:, None] / n)


def project_dct(param, *, chunk_size, transpose=False):
    mat = dct_matrix(chunk_size)
    if transpose:
        mat = mat.T
    orig_shape = param.shape
    ndim = param.ndim
    param = param.reshape(*(
        s
        for d in param.shape
        for s in (d // chunk_size, chunk_size)
    ))
    letters = "abcdefghijklmnopqrstuvwxyz"
    letters = letters + letters.upper()
    base_letters = letters[:ndim]
    dct_letters = letters[ndim:ndim+ndim]
    dct_result_letters = letters[ndim*2:ndim*3]
    einsum_expr = "".join(
        [f"{b}{d}" for b, d in zip(base_letters, dct_letters)]
        + [f",{d}{r}" for d, r in zip(dct_letters, dct_result_letters)]
        + ["->"] +
        [f"{b}{r}" for b, r in zip(base_letters, dct_result_letters)]
    )
    unflattened_result = jnp.einsum(einsum_expr, param, *(mat for _ in dct_letters))
    return unflattened_result.reshape(orig_shape)


def move_bulk_last(x, chunk_size):
    x = x.reshape(*(
        s
        for d in x.shape
        for s in (d // chunk_size, chunk_size)
    ))
    dims = list(range(x.ndim))
    x = x.transpose(*(dims[::2] + dims[1::2]))
    x = x.reshape(*x.shape[:x.ndim // 2], -1)
    return x


def extract_last_bulk(x, chunk_size):
    x = x.reshape(*x.shape[:-1], *(chunk_size for _ in range(x.ndim - 1)))
    dims = list(range(x.ndim))
    dims = np.array(dims).reshape(2, -1).T.flatten().tolist()
    x = x.transpose(*dims)
    return x.reshape(*(
        a * b
        for a, b in zip(x.shape[::2], x.shape[1::2])
    ))
    


@dataclass(frozen=True)
class ProjectConfig:
    chunk_size: int
    k: int

class DemoState(eqx.Module):
    count: jnp.ndarray
    mu: base.Updates
    last_q: base.Updates
    config: ProjectConfig = eqx.field(static=True)


class DemoTopk(eqx.Module):
    values: jax.Array
    indices: jax.Array

def extract_dct(param, *, config: ProjectConfig):
    def map_fn(x):
        projected = project_dct(x, chunk_size=config.chunk_size)
        bulked = move_bulk_last(projected, config.chunk_size)
        values, indices = jax.lax.top_k(
            bulked,
            k=min(config.k, bulked.shape[-1]),
        )
        return DemoTopk(values, indices)
    
    return jax.tree.map(map_fn, param)

def reconstruct_dct(q, *, config: ProjectConfig):
    def map_fn(t: DemoTopk):
        ndim = t.values.ndim - 1
        unchunked_dims = t.values.shape[:-1]
        unchunked_size = np.prod(unchunked_dims)
        bulked = jnp.zeros(
            shape=(unchunked_size, config.chunk_size ** ndim,),
            dtype=t.values.dtype,
        )
        bulked = jax.vmap(
            lambda a, b, c: a.at[b].set(c)
        )(
            bulked,
            t.indices.reshape(unchunked_size, -1),
            t.values.reshape(unchunked_size, -1),
        )
        bulked = bulked.reshape(*unchunked_dims, config.chunk_size ** ndim)
        unbulked = extract_last_bulk(bulked, config.chunk_size)
        unprojected = project_dct(unbulked, chunk_size=config.chunk_size, transpose=True)
        return unprojected
    
    return jax.tree.map(map_fn, q, is_leaf=lambda x: isinstance(x, DemoTopk))
        

def scale_by_demo(b1=0.999, eps=1e-8, config: ProjectConfig = ProjectConfig(chunk_size=8, k=64)) -> base.GradientTransformationExtraArgs:
    def init_fn(params):
        for param in jax.tree.flatten(params)[0]:
            assert all(d % config.chunk_size == 0 for d in param.shape), f"chunk size must divide all dimensions; got param with shape {param.shape}"
        return DemoState(
            count=jnp.zeros((), jnp.int32),
            mu=jax.tree.map(lambda x: jnp.zeros_like(x), params),
            last_q=jax.tree.map(lambda x: jnp.zeros_like(x), extract_dct(params, config=config)),
            config=config,
        )
    
    def update_fn(updates, state, params=None, *, in_kwargs):
        mu = jax.tree.map(lambda x, y: b1 * x + y, state.mu, updates)
        q = extract_dct(mu, config=state.config)
        unprojected_q = reconstruct_dct(q, config=state.config)
        # mu = jax.tree.map(lambda x, y: x - y, mu, unprojected_q)
        # if additional_q is not None:
        #     unprojected_q = jax.tree.map(lambda x, y: x + y, unprojected_q, additional_q)
        update = unprojected_q
        update = jax.tree.map(lambda x: jnp.sign(x), update)
        return update, replace(
            state,
            mu=mu,
            last_q=q,
            count=state.count + 1,
        )
    
    return base.GradientTransformationExtraArgs(
        init_fn, update_fn
    )


def demo(lr_fn, **kwargs):
    return optax.chain(
        scale_by_demo(**kwargs),
        optax.scale_by_learning_rate(lr_fn),
    )