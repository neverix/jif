import fire
from tqdm import trange
from functools import partial

import jax
import wandb
import optax
import jax.numpy as jnp
from jax.tree_util import GetAttrKey, SequenceKey, DictKey
from penzai.toolshed import basic_training
from penzai import pz

from .model import DiTConfig, DitWithTimestep
from .data import collate, get_data
from .diffusion import MDLMDiffusion


def main(
    batch_size = 128,
    seq_len = 64,
    diffusion_eps = 1e-3,
    ema_decay=0.99,
    n_steps=100_000,
    lr=1e-4,
):
    run = wandb.init(project="jif")
    wandb_config = run.config

    n_classes = 256
    diffusion = MDLMDiffusion(n_classes, diffusion_eps)
    config = DiTConfig(vocab_size=n_classes)

    wandb_config.n_classes = n_classes
    wandb_config.n_steps = n_steps
    wandb_config.lr = lr
    wandb_config.diffusion_eps = diffusion_eps
    wandb_config.ema_decay = ema_decay
    wandb_config.batch_size = batch_size
    wandb_config.seq_len = seq_len
    for k, v in config.__dict__.items():
        setattr(wandb_config, k, v)

    model = DitWithTimestep.from_config(config, jax.random.PRNGKey(0))
    
    def score_fn(model, x, _t):
        mask = x == n_classes
        x, side_inputs = model.wrap_inputs(x, mask)
        y = model(x, **side_inputs)
        return y.unwrap("batch", "seq", "vocabulary")

    def get_loss(model, rng, state, sample):
        ema = state["ema"]
        unfrozen_model = jax.tree.map(lambda x: x.unfreeze_as_copy() if isinstance(x, pz.ParameterValue) else x, model, is_leaf=lambda x: isinstance(x, pz.ParameterValue))
        new_params = [x.value.unwrap(*x.value.named_shape.keys()) for x in pz.unbind_params(unfrozen_model)[1]]
        ema = jax.tree.map(lambda x, y: ema_decay * x + (1 - ema_decay) * y, ema, new_params)
        new_state = {"ema": ema}

        sample = jnp.full_like(sample, 4)
        loss = diffusion.get_loss(rng, partial(score_fn, model), sample)
        return loss.mean(), new_state, {"loss": loss.mean()}

    trainer = basic_training.StatefulTrainer.build(
        model=model,
        optimizer_def=optax.chain(optax.adamw(optax.warmup_cosine_decay_schedule(0, lr, 100, n_steps))),
        root_rng=jax.random.key(0),
        loss_fn=get_loss,
        initial_loss_fn_state=dict(ema=[x.value.unwrap(*x.value.named_shape.keys()).copy() for x in pz.unbind_params(model)[1]]),
        donate_states=True)

    def get_samples(trainer_state, batch_size, seq_len, key, num_steps=None):
        if num_steps is None:
            num_steps = seq_len * 2
        ema_params = trainer_state.loss_fn_state["ema"]
        ema_model = jax.tree.map(lambda x: x.unfreeze_as_copy() if isinstance(x, pz.ParameterValue) else x, model, is_leaf=lambda x: isinstance(x, pz.ParameterValue))
        ema_treedef, param_types = pz.unbind_params(ema_model)
        ema_params = [pz.ParameterValue(value=pz.nx.wrap(ep, *pt.value.named_shape.keys()), label=pt.label,) for ep, pt in zip(ema_params, param_types)]
        ema_model = pz.bind_variables(ema_treedef, ema_params)
        samples = diffusion.sample(partial(score_fn, ema_model), key, num_steps, (batch_size, seq_len,))
        return samples

    detokenize, data_generator = get_data()
    for i, (sample, _, _) in zip((bar := trange(n_steps)), collate(data_generator, batch_size, seq_len)):
        sample = jnp.array(sample)
        out = trainer.step(sample=sample)
        if i % 2 == 0:
            bar.set_postfix(loss=out["loss"])
            wandb.log(dict(loss=out["loss"]), step=i)
        if i % 100 == 0:
            print(f"Sampling at step {i}...")
            print(detokenize(get_samples(trainer.state.value, batch_size, seq_len, jax.random.key(i)).tolist()))


if __name__ == "__main__":
    fire.Fire(main)
