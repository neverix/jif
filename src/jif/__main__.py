import random
from functools import partial

import fire
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import sharding
from penzai import pz
from penzai.models.transformer import model_parts
from penzai.toolshed import basic_training, sharding_util
from tqdm import trange

import wandb

from .data import collate, get_data
from .diffusion import MDLMDiffusion
from .model import DiTConfig, DitWithTimestep


def clone_schedule_free(optimizer):
    def init(params):
        state = optimizer.init(params)
        state = state._replace(z=jax.tree.map(lambda x: x.copy(), state.z))
        return state

    def update_fn(*args, **kwargs):
        return optimizer.update(*args, **kwargs)

    return optax.GradientTransformation(init, update_fn)


def main(
    batch_size=256,
    seq_len = 128,
    diffusion_eps = 1e-3,
    ema_decay=0.995,
    n_steps=100_000,
    lr=1e-3,
    n_classes = 258,
    bos_token=256,
    pad_token=257,
    schedule_free=False,
    b1=0.9,
    b2=0.98,
    warmup_steps=100,
    n_mp=1,
    seed=0,
    grad_clip_norm=10.0,
    wandb_every=1,
    sample_every=100,
):
    random.seed(seed)
    np.random.seed(seed)

    run = wandb.init(project="jif")
    wandb_config = run.config

    diffusion = MDLMDiffusion(n_classes, diffusion_eps, bos_token=bos_token)
    config = DiTConfig(vocab_size=n_classes,)

    mesh = sharding.Mesh(np.array(jax.devices("tpu")).reshape((-1, n_mp)), ("dp", "mp"))

    wandb_config.seed = seed
    wandb_config.grad_clip_norm = grad_clip_norm
    wandb_config.warmup_steps = warmup_steps
    wandb_config.n_classes = n_classes
    wandb_config.bos_token = bos_token
    wandb_config.pad_token = pad_token
    wandb_config.n_steps = n_steps
    wandb_config.lr = lr
    wandb_config.diffusion_eps = diffusion_eps
    wandb_config.ema_decay = ema_decay
    wandb_config.batch_size = batch_size
    wandb_config.seq_len = seq_len
    wandb_config.schedule_free = schedule_free
    wandb_config.b1 = b1
    wandb_config.b2 = b2
    wandb_config.grad_clip_norm = grad_clip_norm
    wandb_config.n_mp = n_mp
    wandb_config.wandb_every = wandb_every
    wandb_config.sample_every = sample_every
    wandb_config.dp = mesh.shape["dp"]
    for k, v in config.__dict__.items():
        setattr(wandb_config, k, v)

    key = jax.random.key(seed)
    model_key, run_key, sample_key = jax.random.split(key, 3)

    data_sharding = sharding.NamedSharding(mesh, sharding.PartitionSpec("dp", None))
    axis_name_to_mesh_name = {"batch": "dp", "neurons": "mp", "kv_heads": "mp"}
    model = sharding_util.sharded_init(DitWithTimestep.from_config,
                                       config, model_key,
                                       mesh=mesh,
                                       axis_name_to_mesh_name=axis_name_to_mesh_name)
    model = (model.select().at_instances_of(pz.nn.Residual)
             .insert_after(sharding_util.ConstrainShardingByName(
                 mesh, axis_name_to_mesh_name=axis_name_to_mesh_name)))
    
    def score_fn(model, x, _t):
        mask = x == n_classes
        x, side_inputs = model.wrap_inputs(x, mask)
        y = model(x, **side_inputs)
        return y.unwrap("batch", "seq", "vocabulary")

    def get_loss(model, rng, state, sample):
        if ema_decay is not None:
            ema = state["ema"]
            unfrozen_model = jax.tree.map(lambda x: x.unfreeze_as_copy() if isinstance(x, pz.ParameterValue) else x, model, is_leaf=lambda x: isinstance(x, pz.ParameterValue))
            new_params = [x.value.unwrap(*x.value.named_shape.keys()) for x in pz.unbind_params(unfrozen_model)[1]]
            ema = jax.tree.map(lambda x, y: ema_decay * x + (1 - ema_decay) * y, ema, new_params)
            new_state = {"ema": ema}
        else:
            new_state = state
            
        err, loss = diffusion.get_loss(rng, partial(score_fn, model), sample)
        return loss.mean(), new_state, {"loss": loss.mean(), "err": err}

    if not schedule_free:
        lr_fn = optax.warmup_cosine_decay_schedule(0, lr, warmup_steps, n_steps)
        optimizer = optax.adamw(lr_fn, b1=b1, b2=b2)
    else:
        lr_fn = optax.warmup_cosine_decay_schedule(0, lr, warmup_steps, n_steps, end_value=lr)
        optimizer = optax.adamw(lr_fn, b1=0., b2=b2)
        optimizer = clone_schedule_free(optax.contrib.schedule_free(optimizer, lr_fn, b1=b1))
        ema_decay = None
    trainer = basic_training.StatefulTrainer.build(
        model=model,
        optimizer_def=optax.chain(optax.clip_by_global_norm(grad_clip_norm), optimizer),
        root_rng=run_key,
        loss_fn=get_loss,
        initial_loss_fn_state=dict(ema=([x.value.unwrap(*x.value.named_shape.keys()).astype(jnp.float32).copy() for x in pz.unbind_params(model)[1]] if ema_decay is not None else None)),
        donate_states=True)

    @partial(pz.variable_jit, static_argnames=("batch_size", "seq_len", "num_steps"))
    def get_samples(trainer, batch_size, seq_len, key, num_steps=None):
        if num_steps is None:
            num_steps = seq_len * 16
        trainer_state = trainer.state.value
        model = trainer.model
        if ema_decay is not None:
            ema_params = trainer_state.loss_fn_state["ema"]
            ema_model = jax.tree.map(lambda x: x.unfreeze_as_copy() if isinstance(x, pz.ParameterValue) else x, model, is_leaf=lambda x: isinstance(x, pz.ParameterValue))
            ema_treedef, param_types = pz.unbind_params(ema_model)
            ema_params = [pz.ParameterValue(value=pz.nx.wrap(ep.astype(config.parameter_dtype), *pt.value.named_shape.keys()), label=pt.label,) for ep, pt in zip(ema_params, param_types)]
            ema_model = pz.bind_variables(ema_treedef, ema_params)
        else:
            optim_state = trainer_state.opt_state[-1]  # remove gradient processors
            ema_treedef, params = pz.unbind_params(model, freeze=True)
            ema_model_params = optax.contrib.schedule_free_eval_params(optim_state, params)
            ema_model = pz.bind_variables(ema_treedef, ema_model_params)

        samples = diffusion.sample(partial(score_fn, ema_model), key, num_steps, (batch_size, seq_len,))
        return samples

    detokenize, data_generator = get_data()
    for step, (sample, _, _) in zip((bar := trange(n_steps)), collate(data_generator, batch_size, seq_len, pad_token_id=pad_token)):
        sample = jnp.asarray(np.asarray(sample, dtype=np.uint32), device=data_sharding)
        out = trainer.step(sample=sample)
        out["err"].throw()
        if step % wandb_every == 0:
            bar.set_postfix(loss=out["loss"])
            wandb.log(dict(loss=out["loss"]), step=step)
        if step % sample_every == 0:
            print(f"Sampling at step {step}...")
            print(detokenize(get_samples(trainer, 4, seq_len, jax.random.fold_in(sample_key, step)).tolist()))


if __name__ == "__main__":
    fire.Fire(main)
