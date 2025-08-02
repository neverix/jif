# https://arxiv.org/abs/2502.17405
import jax
import jax.numpy as jnp
from optax._src import base
import optax
from typing import NamedTuple
from penzai import pz
import chex

class FlermState(NamedTuple):
    ema_z2: base.Updates
    ema_zzt: base.Updates
    ema_ztz: base.Updates
    initial_fslr: base.Updates
    count: chex.Array

def scale_by_flerm(
    learning_rate: base.ScalarOrSchedule,
    model_params,
    beta: float = 0.95,
    eps: float = 1e-8,
    warmup_steps: int = 100,
    estimate_every: int = 100
) -> base.GradientTransformationExtraArgs:
    def init_fn(params):
        return FlermState(
            ema_z2=jax.tree.map(lambda x: jnp.ones_like(x, shape=(1,)), params),
            ema_zzt=jax.tree.map(lambda x: jnp.ones_like(x, shape=(1,)), params),
            ema_ztz=jax.tree.map(lambda x: jnp.ones_like(x, shape=(1,)), params),
            initial_fslr=jax.tree.map(lambda x: jnp.ones_like(x, shape=(1,)), params),
            count=jnp.zeros([], jnp.int32)
        )
    
    @jax.grad
    def results_proj_grad(params, in_kwargs):
        model = pz.bind_variables(model_params, params)
        arg, extra = model.wrap_inputs(**in_kwargs)
        result = model(arg, **extra)
        result_proj = (result.data_array * jax.random.normal(jax.random.key(0), result.data_array.shape)).sum()
        return result_proj
    
    def update_fn(updates, state, params, *, in_kwargs):
        grads = results_proj_grad(params, in_kwargs)
        
        if callable(learning_rate):
            update_size = learning_rate(state.count)
        else:
            update_size = learning_rate
        z = jax.tree.map(lambda x, y: x * y / update_size, grads, updates)
        
        ema_z2 = jax.tree.map(lambda x, y: beta * x + (1 - beta) * (z ** 2).sum(), state.ema_z2, z)
        ema_zzt = jax.tree.map(lambda x, y: beta * x + (1 - beta) * (z.sum(axis=0) ** 2).sum() if x.ndim == 2 else x, state.ema_zzt, z)
        ema_ztz = jax.tree.map(lambda x, y: beta * x + (1 - beta) * (z.sum(axis=1) ** 2).sum() if x.ndim == 2 else x, state.ema_ztz, z)
        
        fslr = jax.tree.map(lambda x, y, z: (x * y / z) ** 0.5 if x.ndim == 2 else (1 / z), state.ema_zzt, state.ema_ztz, state.ema_z2)
        initial_fslr = 
        updates = jax.tree.map(lambda x, y: x / y, updates, zzz)
        
        return updates, FlermState(ema_z2, ema_zzt, ema_ztz, initial_fslr, state.count + 1)
    
    return base.GradientTransformationExtraArgs(init_fn, update_fn)