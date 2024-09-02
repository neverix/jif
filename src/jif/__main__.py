import random
from functools import partial

import fire
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
from jax import sharding
from penzai import pz
from penzai.models.transformer import model_parts
from penzai.toolshed import basic_training, sharding_util
from tqdm.auto import tqdm, trange

import wandb

from .data import get_data
from .diffusion import MDLMDiffusion
from .model import DiTConfig, DitWithTimestep


def clone_schedule_free(optimizer):
    def init(params):
        state = optimizer.init(params)
        state = state._replace(z=jax.tree.map(lambda x: x.copy(), state.z))
        return state

    return optax.GradientTransformation(init, optimizer.update)


def train(
    batch_size=256,
    seq_len=128,
    diffusion_eps = 1e-3,
    ema_decay=0.995,
    n_steps=10_000,
    lr=1e-3,
    schedule_free=False,
    b1=0.9,
    b2=0.98,
    warmup_steps=100,
    n_mp=1,
    seed=0,
    grad_clip_norm=10.0,
    sample_steps=512,
    ema_dtype="bfloat16",
    accurate_flops_calc=False,
    profile=False,
    size="small",
    quiet=False,
):
    profile = profile and not quiet
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    data_generator, detokenize, n_classes, bos_token = get_data(batch_size, seq_len)

    diffusion = MDLMDiffusion(n_classes, diffusion_eps, bos_token=bos_token)

    mesh = sharding.Mesh(np.array(jax.devices("tpu")).reshape((-1, n_mp)), ("dp", "mp"))
    data_sharding = sharding.NamedSharding(mesh, sharding.PartitionSpec("dp", None))
    axis_name_to_mesh_name = {"batch": "dp", "neurons": "mp", "kv_heads": "mp", "vocabulary": "mp"}
    n_layers, d_model = {
        "small": (3, 256),
        "medium": (4, 384),
        "big": (6, 512),
    }[size]
    batch_size = {
        "small": 1024,
        "medium": 512,
        "big": 256,
    }[size]
    wandb_every, sample_every = {
        "small": (100, 1000),
        "medium": (50, 250),
        "big": (10, 100),
    }[size]
    config = DiTConfig(vocab_size=n_classes, axis_name_to_mesh_name=axis_name_to_mesh_name, mesh=mesh,
                       n_layers=n_layers, d_model=d_model, n_kv_heads=d_model//64, q_rep=1, qk_dim=64, v_dim=64,
                       d_ff=d_model * 3)

    if not quiet:
        run = wandb.init(project="jif")
        wandb_config = run.config

        wandb_config.size = size
        wandb_config.n_layers = n_layers
        wandb_config.d_model = d_model
        wandb_config.profile = profile
        wandb_config.seed = seed
        wandb_config.accurate_flops_calc = accurate_flops_calc
        wandb_config.grad_clip_norm = grad_clip_norm
        wandb_config.warmup_steps = warmup_steps
        wandb_config.n_classes = n_classes
        wandb_config.sample_steps = sample_steps
        wandb_config.bos_token = bos_token
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
        wandb_config.wandb_every = wandb_every
        wandb_config.sample_every = sample_every
        wandb_config.dp = mesh.shape["dp"]
        wandb_config.mp = mesh.shape["mp"]
        wandb_config.ema_dtype = ema_dtype
        for k, v in config.__dict__.items():
            setattr(wandb_config, "model." + k, v)

    ema_dtype = getattr(jnp, ema_dtype)

    key = jax.random.key(seed)
    model_key, run_key, sample_key = jax.random.split(key, 3)

    model = sharding_util.sharded_init(DitWithTimestep.from_config,
                                       config, model_key,
                                       mesh=mesh,
                                       axis_name_to_mesh_name=axis_name_to_mesh_name)
    param_count = sum(v.value.data_array.size for v in pz.unbind_params(model)[1])
    if not quiet:
        print(f"Parameter count: {param_count}")
    model = (model.select().at_instances_of(pz.nn.Residual)
             .insert_after(sharding_util.ConstrainShardingByName(
                 mesh, axis_name_to_mesh_name=axis_name_to_mesh_name)))
    
    def score_fn(model, x, _t):
        mask = x == n_classes
        x, side_inputs = model.wrap_inputs(x, mask)
        y = model(x, **side_inputs)
        return y.unwrap("batch", "seq", "vocabulary")

    def get_loss(model, rng, state, sample, update_state=True):
        if ema_decay is not None and update_state:
            ema = state["ema"]
            unfrozen_model = jax.tree.map(lambda x: x.unfreeze_as_copy() if isinstance(x, pz.ParameterValue) else x, model, is_leaf=lambda x: isinstance(x, pz.ParameterValue))
            new_params = [x.value.unwrap(*x.value.named_shape.keys()) for x in pz.unbind_params(unfrozen_model)[1]]
            ema = jax.tree.map(lambda x, y: ema_decay * x + (1 - ema_decay) * y, ema, new_params)
            new_state = {"ema": ema}
        else:
            new_state = state
            
        # err, loss = diffusion.get_loss(rng, partial(score_fn, model), sample)
        loss = diffusion.get_loss(rng, partial(score_fn, model), sample)
        err = None
        return loss.mean(), new_state, {"loss": loss.mean(), "err": err}

    if not schedule_free:
        lr_fn = optax.warmup_cosine_decay_schedule(0, lr, warmup_steps, n_steps)
        optimizer = optax.adamw(lr_fn, b1=b1, b2=b2)
    else:
        lr_fn = optax.warmup_cosine_decay_schedule(0, lr, warmup_steps, n_steps, end_value=lr)
        optimizer = optax.adamw(lr_fn, b1=0., b2=b2)
        optimizer = clone_schedule_free(optax.contrib.schedule_free(optimizer, lr_fn, b1=b1))
    trainer = basic_training.StatefulTrainer.build(
        model=model,
        optimizer_def=optax.chain(optax.clip_by_global_norm(grad_clip_norm), optimizer),
        root_rng=run_key,
        loss_fn=get_loss,
        initial_loss_fn_state=dict(ema=([x.value.unwrap(*x.value.named_shape.keys()).astype(ema_dtype).copy() for x in pz.unbind_params(model)[1]] if ema_decay is not None else None)),
        donate_states=True)

    @partial(pz.variable_jit, static_argnames=("batch_size", "seq_len", "num_steps"))
    def get_samples(trainer, batch_size, seq_len, key, num_steps=None):
        if num_steps is None:
            num_steps = sample_steps
        trainer_state = trainer.state.value
        model = trainer.model
        if schedule_free:
            optim_state = trainer_state.opt_state[-1]  # remove gradient processors
            treedef, params = pz.unbind_params(model, freeze=True)
            model_params = optax.contrib.schedule_free_eval_params(optim_state, params)
            model = pz.bind_variables(treedef, model_params)
        
        if ema_decay is not None:
            ema_params = trainer_state.loss_fn_state["ema"]
            ema_model = jax.tree.map(lambda x: x.unfreeze_as_copy() if isinstance(x, pz.ParameterValue) else x, model, is_leaf=lambda x: isinstance(x, pz.ParameterValue))
            ema_treedef, param_types = pz.unbind_params(ema_model)
            ema_params = [pz.ParameterValue(value=pz.nx.wrap(ep.astype(config.parameter_dtype), *pt.value.named_shape.keys()), label=pt.label,) for ep, pt in zip(ema_params, param_types)]
            ema_model = pz.bind_variables(ema_treedef, ema_params)
        else:
            ema_model = model

        samples = diffusion.sample(partial(score_fn, ema_model), key, num_steps, (batch_size, seq_len,))
        return samples

    slots = pz.unbind_variables(trainer.model)[0]
    @partial(jax.jit, static_argnames=("update_state",))
    @jax.grad
    def get_loss_grad(model_vars, rng, state, sample, update_state=True):
        model = pz.bind_variables(slots, [v.unfreeze_as_copy() for v in model_vars])
        return get_loss(model, rng, state, sample, update_state=update_state)[0]

    model_flops = None
    losses = []
    log_dict = {}
    for step, sample in zip((bar := trange(n_steps)), data_generator()):
        sample = jax.device_put(jnp.asarray(sample.numpy().astype(np.uint32), device=jax.devices("cpu")[0]), data_sharding)
        if not quiet:
            if model_flops is None:
                if accurate_flops_calc:
                    model_variables = pz.unbind_variables(trainer.model)[1]
                    loss_compiled = get_loss_grad.lower([v.freeze() for v in model_variables],
                                                jax.random.key(0), None,
                                                sample=sample, update_state=False).compile()
                    model_flops = loss_compiled.cost_analysis()[0]["flops"]
                    del model, model_variables
                else:
                    model_flops = batch_size * seq_len * 6 * param_count
                print("Model FLOPs:", model_flops)
                print(f" (per token: {model_flops / seq_len / batch_size:.2f})")
        if step == 5 and profile:
            jax.profiler.start_trace("/tmp/tensorboard")
        out = trainer.step(sample=sample)
        losses.append(out["loss"])
        if out.get("err") is not None:
            out["err"].throw()
        if step == 25 and profile:
            jax.profiler.stop_trace()
            
        log_dict = dict(loss=out["loss"], loss_sma=np.mean(losses[-100:]))
        itps = bar.format_dict["rate"]
        if itps is not None and model_flops is not None:
            one_v4_chip_flops = 275 * 1e12  # https://cloud.google.com/tpu/docs/v4
            total_flops = one_v4_chip_flops * len(jax.devices("tpu"))
            log_dict["mfu"] = (itps * model_flops) / total_flops
        bar.set_postfix(**log_dict)
        if step % wandb_every == 0:
            if not quiet:
                wandb.log(log_dict, step=step)
        if not quiet:
            if step % sample_every == 0:
                print(f"Sampling at step {step}...")
                print(detokenize(get_samples(trainer, 4, seq_len, jax.random.fold_in(sample_key, step)).tolist()))
    return log_dict


def main(*args, **kwargs):
    from matplotlib import pyplot as plt
    from collections import defaultdict
    from itertools import product
    lrs = np.linspace(1e-5, 2e-3, 10).tolist()
    models = ["small", "medium", "big"]
    colors = ["r", "g", "b"]
    lrs_sampled = defaultdict(list)
    losses = defaultdict(list)
    all_configs = list(product(lrs, models))
    random.Random(0).shuffle(all_configs)
    for lr, model in tqdm(all_configs):
        print("Training", model, "model with", lr, "learning rate")
        lrs_sampled[model].append(lr)
        log_dict = train(*args, **kwargs | {"n_steps": 2_000, "lr": lr, "size": "small"},
                         quiet=True, profile=False)
        losses[model].append(log_dict["loss"])
        for model, color in zip(models, colors):
            plt.plot(lrs_sampled[model], losses[model], label=model, c=color)
            plt.scatter(lrs_sampled[model], losses[model], label=model, marker="x", c=color)
        plt.xlabel("Learning rate")
        plt.ylabel("Final loss")
        plt.legend()
        plt.savefig("lr_search.png")
        plt.close()


if __name__ == "__main__":
    fire.Fire(main)
