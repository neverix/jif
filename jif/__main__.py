import random
from dataclasses import replace
import threading
from functools import partial
import os
import io
import socket

import equinox as eqx
import fire
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
from jax import sharding
from penzai import pz
from penzai.toolshed import sharding_util
from tqdm.auto import tqdm, trange
import tqdm as mtqdm
import datasets

import wandb

from .data import get_data
from .diffusion import MDLMDiffusion
from .model import DiTConfig, DitWithTimestep
from . import basic_training
from .muon import muon
from .demo import demo, DemoState, reconstruct_dct
from .optax_utils import vmaptax, squeezeflaptax
from .raleigh import RaleighCommunicator


def train(
    batch_size=None,
    seq_len=128,
    diffusion_eps = 1e-3,
    ema_decay=0.995,
    n_steps=100_000,
    lr=1e-3,
    schedule_free=False,
    use_muon=False,
    use_demo=True,
    b1=0.9,
    b2=0.98,
    warmup_steps=100,
    n_mp=1,
    seed=0,
    params_seed=-1,
    grad_clip_norm=10.0,
    sample_steps=512,
    ema_dtype="bfloat16",
    use_flash_attention=True,
    accurate_flops_calc=False,
    profile=False,
    # size="small",
    size="big",
    quiet=False,
    fix_batch_size=False,
    loss_sma=256,
    dit_conditioning=True,
    use_modula=False,
    raleigh_ports=[],
    raleigh_friends=[]
):
    if not use_demo:
        raleigh_friends = []

    profile = profile and not quiet
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    mesh = sharding.Mesh(np.array(jax.devices("tpu")).reshape((-1, n_mp)), ("dp", "mp"))
    data_sharding = sharding.NamedSharding(mesh, sharding.PartitionSpec("dp", None))
    axis_name_to_mesh_name = {"batch": "dp", "neurons": "mp", "kv_heads": "mp", "vocabulary": "mp"}
    n_layers, d_model = {
        "small": (3, 256),
        "medium": (4, 384),
        "big": (6, 512),
    }[size]
    if batch_size is None:
        if fix_batch_size:
            batch_size = {
                "small": 256,
                "medium": 256,
                "big": 256,
            }[size]
        else:
            batch_size = {
                "small": 1024,
                "medium": 512,
                "big": 256,
            }[size]
    wandb_every, sample_every = {
        "small": (100, 1000),
        "medium": (50, 500),
        "big": (25, 300),
    }[size]
    data_generator, detokenize, n_classes, bos_token = get_data(batch_size, seq_len, seed=seed)
    diffusion = MDLMDiffusion(n_classes, diffusion_eps, bos_token=bos_token)
    config = DiTConfig(vocab_size=n_classes, axis_name_to_mesh_name=axis_name_to_mesh_name, mesh=mesh,
                       n_layers=n_layers, d_model=d_model, n_kv_heads=d_model//64, q_rep=1, qk_dim=64, v_dim=64,
                       d_ff=d_model * 3, use_modula=use_modula, dit_conditioning=dit_conditioning, use_flash_attention=use_flash_attention)

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
        wandb_config.dit_conditioning = dit_conditioning
        wandb_config.use_flash_attention = use_flash_attention
        for k, v in config.__dict__.items():
            setattr(wandb_config, "model." + k, v)

    ema_dtype = getattr(jnp, ema_dtype)

    key = jax.random.key(seed)
    run_key, sample_key, data_root_rng = jax.random.split(key, 3)
    
    if params_seed == -1:
        params_seed = seed
    model_key = jax.random.key(params_seed)

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
    
    def score_fn(model, x, t):
        x, side_inputs = model.wrap_inputs(x, t=t)
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
            
        data = diffusion.perturb(rng, sample)
        loss = data.loss(score_fn(model, data.data_perturbed, data.t))
        err = None
        return loss.mean(), new_state, {"loss": loss.mean(), "err": err}

    if use_muon:
        lr_fn = optax.warmup_cosine_decay_schedule(0, lr, warmup_steps, n_steps)
        optimizer = muon(lr_fn)
    elif use_demo:
        lr_fn = optax.warmup_cosine_decay_schedule(0, lr * 4e-1, warmup_steps, n_steps)
        # TODO tune lr
        adam_lr_fn = optax.warmup_cosine_decay_schedule(0, lr, warmup_steps, n_steps)
        optimizer = demo(lr_fn)
        # optimizer = optax.partition({
        #     "adam": optax.adamw(adam_lr_fn, b1=b1, b2=b2),
        #     "demo": demo(lr_fn, config=ProjectConfig(chunk_size=8, k=4)),
        # }, param_labels=lambda params: jax.tree.map((lambda x:
        #     # TODO
        #     # "adam" if x.ndim == 1 else "demo"
        #     "demo"
        # ), params))
    else:
        if not schedule_free:
            lr_fn = optax.warmup_cosine_decay_schedule(0, lr, warmup_steps, n_steps)
            optimizer = optax.adamw(lr_fn, b1=b1, b2=b2)
        else:
            lr_fn = optax.warmup_cosine_decay_schedule(0, lr, warmup_steps, n_steps, end_value=lr)
            optimizer = optax.adamw(lr_fn, b1=0., b2=b2)
            optimizer = clone_schedule_free(optax.contrib.schedule_free(optimizer, lr_fn, b1=b1))
    vmaptax_dims = (config.n_layers,)
    trainer = basic_training.StatefulTrainer.build(
        model=model,
        optimizer_def=vmaptax(squeezeflaptax(optax.chain(optax.clip_by_global_norm(grad_clip_norm), optimizer)), vmap_with_dims=vmaptax_dims),
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

        table = model.select().at_instances_of(pz.nn.EmbeddingDecode).get().table
        samples = diffusion.sample(partial(score_fn, ema_model), key, num_steps, (batch_size, seq_len,), vocab_size=table.embeddings.value.named_shape[table.vocabulary_axis])
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
    
    if raleigh_friends:
        communicator = RaleighCommunicator(raleigh_ports, raleigh_friends)
        communicator.start()
        for _ in range(len(raleigh_friends)):
            assert communicator.results_queue.get()[0] == "ready"
        print("Communicator ready")
    
    for step, sample in zip((bar := trange(n_steps)), data_generator()):
        sample = jax.device_put(jnp.asarray(sample.numpy().astype(np.uint32), device=jax.devices("cpu")[0]), data_sharding)
        if not quiet:
            if model_flops is None:
                if accurate_flops_calc:
                    model_variables = pz.unbind_variables(trainer.model)[1]
                    loss_compiled = get_loss_grad.lower([v.freeze() for v in model_variables],
                                                jax.random.key(0), None,
                                                sample=sample, update_state=False).compile()
                    model_flops = loss_compiled.cost_analysis()["flops"]
                    del model, model_variables
                else:
                    model_flops = batch_size * seq_len * 6 * param_count
                print("Model FLOPs:", model_flops)
                print(f" (per token: {model_flops / seq_len / batch_size:.2f})")
        if step == 5 and profile:
            jax.profiler.start_trace("/tmp/tensorboard")

        if use_demo:
            state = trainer.state.value
            opt_state = state.opt_state
            
            leaves, optim_treedef = jax.tree.flatten(opt_state, is_leaf=lambda x: isinstance(x, DemoState))
            demo_state_indices, demo_states = zip(*[(i, leaf) for i, leaf in enumerate(leaves) if isinstance(leaf, DemoState)])
            assert len(demo_states) > 0
            
            last_q = [state.last_q for state in demo_states]
            last_q_cpu = jax.device_put(last_q, jax.devices("cpu")[0])
            
            if raleigh_friends:
                if not communicator.is_alive():
                    raise RuntimeError(f"Communicator died")
                last_q_bytes = io.BytesIO()
                eqx.tree_serialise_leaves(last_q_bytes, last_q_cpu)
                for _ in range(len(raleigh_friends)):
                    communicator.queue.put_nowait(("last_qs", last_q_bytes.getvalue()))
                received_last_qs_bytes = {friend: [] for friend in raleigh_friends}
                received_last_qs_cpu = {friend: None for friend in raleigh_friends}
                while True:
                    if all(received_last_qs_cpu.values()):
                        break
                    msg_type, friend, received_last_q_bytes = communicator.results_queue.get()
                    assert msg_type == "last_qs"
                    received_last_qs_bytes[friend].append(received_last_q_bytes)
                    received_length = sum(map(len, received_last_qs_bytes[friend]))
                    if received_length > len(last_q_bytes.getvalue()):
                        raise RuntimeError(f"Received more bytes than expected from {friend}")
                    if received_length == len(last_q_bytes.getvalue()):
                        recieved_last_q_bytes = b"".join(received_last_qs_bytes[friend])
                        received_last_qs_cpu[friend] = eqx.tree_deserialise_leaves(io.BytesIO(recieved_last_q_bytes), last_q_cpu)
                        break
                received_last_qs_cpu = list(received_last_qs_cpu.values())
            else:
                received_last_qs_cpu = [last_q_cpu]
            
            # TODO figure out why device_put_like doesn't work
            # received_last_qs = [device_put_like(rlq, last_qs) for rlq in received_last_qs_cpu]
            received_last_qs = jax.tree.map(lambda x: jax.device_put(x, jax.sharding.NamedSharding(mesh, sharding.PartitionSpec(*((None,) * x.ndim)))), received_last_qs_cpu)
            
            for received_last_q in received_last_qs:
                demo_states = update_demo_states(demo_states, received_last_q, vmaptax_dims)
            
            reverse_demo_state_indices = {i: j for j, i in enumerate(demo_state_indices)}
            new_opt_state = jax.tree.unflatten(optim_treedef, [
                leaf if i not in demo_state_indices else demo_states[reverse_demo_state_indices[i]]
                for i, leaf in enumerate(leaves)
            ])
                
            
            trainer.state.value = basic_training.InternalTrainerState(
                opt_state=new_opt_state,
                **{k: v for k, v in vars(state).items() if k != "opt_state"}
            )
        
        step_rng = jax.random.fold_in(data_root_rng, trainer.state.value.step)
        diffusion_input_additional = diffusion.perturb(step_rng, sample)
        out = trainer.step(sample=sample, optimizer_extra_args={"in_kwargs": {"x": diffusion_input_additional.data_perturbed, "t": diffusion_input_additional.t}})
        
        if out.get("err") is not None:
            out["err"].throw()
        loss = float(out["loss"])
        losses.append(loss)
        if step == 100 and profile:
            jax.profiler.stop_trace()
            
        log_dict = dict(loss=loss, loss_sma=np.mean(losses[-loss_sma:]))
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


def device_put_like(x, y):
    return jax.tree.map(lambda x, y: jax.device_put(x, y.sharding), x, y)


@eqx.filter_jit
def update_demo_states(demo_states, received_last_qs, vmaptax_dims):
    return [
        vmap_update_demo_state(state, received_last_q)
        if state.mu.shape[0] in vmaptax_dims
        else update_demo_state(state, received_last_q)
        for state, received_last_q in zip(demo_states, received_last_qs)]


def update_demo_state(state, received_last_q):
    reconstructed = reconstruct_dct(received_last_q, config=state.config)
    return replace(state, last_unprojected_q=state.last_unprojected_q + reconstructed)


vmap_update_demo_state = eqx.filter_vmap(update_demo_state)
update_demo_state = eqx.filter_jit(update_demo_state)


def clone_schedule_free(optimizer):
    def init(params):
        state = optimizer.init(params)
        state = state._replace(z=jax.tree.map(lambda x: x.copy(), state.z))
        return state

    return optax.GradientTransformation(init, optimizer.update)


def train_parallel(
    n_threads: int = 2,
    **kwargs
):
    def relative_to(i, j):
        return (i - j - 1) % n_threads
    
    n_needed_ports = n_threads * (n_threads - 1)
    free_ports = []
    for _ in range(n_needed_ports):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("0.0.0.0", 0))
        free_ports.append(s.getsockname()[1])
        s.close()
    # free_ports = [13370 + i for i in range(n_threads * (n_threads - 1))]
    
    datasets.disable_progress_bars()
    os.environ["TQDM_DISABLE"] = "1"
    def train_thread(thread_id):
        return train(**(kwargs | {
            "seed": thread_id, "params_seed": 0, "quiet": thread_id != 0,
            "raleigh_ports": [free_ports[(n_threads - 1) * thread_id + i] for i in range(n_threads - 1)],
            "raleigh_friends": [("127.0.0.1", free_ports[(n_threads - 1) * thread_id + relative_to(thread_id, i)]) for i in range(n_threads) if i != thread_id]
        }))

    threads = [threading.Thread(target=train_thread, args=(i,)) for i in range(n_threads)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


if __name__ == "__main__":
    fire.Fire(train_parallel)
