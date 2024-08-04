import fire
from tqdm import tqdm
from functools import partial

import jax
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
):
    n_classes = 128
    diffusion = AbsorbingDiffusion(n_classes, 1e-3)
    config = DiTConfig(vocab_size=128)
    model = DitWithTimestep.from_config(config, jax.random.PRNGKey(0))
    
    def score_fn(model, x, _t):
        mask = x == n_classes
        x, side_inputs = model.wrap_inputs(x, mask)
        y = model(x, **side_inputs)
        return y.unwrap("batch", "seq", "vocabulary")
    
    def get_loss(model, rng, state, sample):
        del state
        loss = diffusion.get_loss(rng, partial(score_fn, model), jnp.array(sample))
        return loss.mean(), None, {"loss": loss}
    
    trainer = basic_training.StatefulTrainer.build(
        model=model,
        optimizer_def=optax.adamw(3e-4),
        root_rng=jax.random.key(0),
        loss_fn=get_loss)
    
    for sample, _, _ in (bar := tqdm(collate(get_data(), batch_size, seq_len))):
        out = trainer.step(sample=sample)
        bar.set_postfix(loss=out["loss"])


if __name__ == "__main__":
    fire.Fire(main)
