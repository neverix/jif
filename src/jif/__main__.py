import fire
from tqdm import tqdm
from functools import partial

import jax
import wandb
import optax
import jax.numpy as jnp
from penzai.toolshed import basic_training
from penzai import pz

from .model import DiTConfig, DitWithTimestep
from .data import collate, get_data
from .diffusion import AbsorbingDiffusion


def main(
    batch_size = 128,
    seq_len = 128,
    diffusion_eps = 1e-3
):
    run = wandb.init(project="jif")
    wandb_config = run.config

    n_classes = 128
    diffusion = AbsorbingDiffusion(n_classes, diffusion_eps)
    config = DiTConfig(vocab_size=128)

    wandb_config.n_classes = n_classes
    wandb_config.diffusion_eps = diffusion_eps
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
        del state
        loss = diffusion.get_loss(rng, partial(score_fn, model), jnp.array(sample))
        return loss.mean(), None, {"loss": loss.mean()}
    
    trainer = basic_training.StatefulTrainer.build(
        model=model,
        optimizer_def=optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(1e-4)),
        root_rng=jax.random.key(0),
        loss_fn=get_loss,
        donate_states=True)
    
    for i, (sample, _, _) in enumerate((bar := tqdm(collate(get_data(), batch_size, seq_len)))):
        out = trainer.step(sample=sample)
        if i % 10 == 0:
            bar.set_postfix(loss=out["loss"])
            wandb.log(dict(loss=out["loss"]))


if __name__ == "__main__":
    fire.Fire(main)
