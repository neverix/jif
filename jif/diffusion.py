from dataclasses import dataclass
from functools import partial
from typing import Optional

import jax
import jax.experimental
import jax.numpy as jnp
from jax.experimental import checkify
from jax.experimental.shard_alike import shard_alike
from penzai import pz


@pz.pytree_dataclass
class DiffusionInput(pz.Struct):
    data: jax.Array
    data_perturbed: jax.Array
    t: jax.Array
    alpha: jax.Array
    rate: jax.Array
    diffusion: "MDLMDiffusion"

    def loss(self, logits):
        logits = self.diffusion.process_logits(logits)
        labels = self.diffusion.replace_bos(self.data)
        lse = jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
        gain = jnp.take_along_axis(logits - lse, labels[..., None], axis=-1).squeeze(-1)
        weights = self.rate / (1 - self.alpha)
        weights, gain = shard_alike(weights, gain)
        loss = jnp.where(self.data_perturbed == self.diffusion.n_classes, weights * gain, 0)
        return loss


@pz.pytree_dataclass
class MDLMDiffusion(pz.Struct):
    n_classes: int
    noise_eps: float = 1e-3
    z_loss_coeff: float = 1e-4
    bos_token: Optional[int] = None
    
    def replace_bos(self, x):
        if self.bos_token is not None:
            x = x.at[..., 0].set(self.bos_token)
        return x

    def perturb(self, key, data):
        t = jnp.linspace(1.0, 0.0, data.shape[0], dtype=jnp.float32)
        for _ in range(data.ndim - 1):
            t = t[..., None]
        t = t + jnp.zeros_like(data, dtype=jnp.float32)
        t, data = shard_alike(t, data) # just to make sure
        alpha, rate = self.alpha(t), self.alpha_rate(t)
        data_perturbed = self.sample_transition(key, data, alpha)
        data_perturbed = self.replace_bos(data_perturbed)
        for _ in range(data.ndim - 1):
            t = t[..., 0]
        return DiffusionInput(data, data_perturbed, t, alpha, rate, self)

    def sample_transition(self, key, data, alpha):
        mask_chance = 1 - alpha
        mask_indices = jax.random.bernoulli(key, mask_chance, data.shape)
        data_perturbed = jnp.where(mask_indices, self.n_classes, data)
        return data_perturbed

    def alpha(self, t):
        return 1 - self.noise_eps - (1 - self.noise_eps) * t

    def alpha_rate(self, _t):
        return jnp.full_like(_t, -(1 - self.noise_eps))

    def process_logits(self, logits):
        logits = logits.at[..., self.n_classes:].set(-1e10)
        assert logits.shape[-1] >= self.n_classes + 1
        return logits

    # @partial(jax.jit, static_argnames=("use_caching", "denoise", "batch_shape", "n_steps", "projector"))
    def sample(self, score_fn, key, n_steps, batch_shape, denoise=True, projector=lambda x: x, use_caching=False, *, vocab_size: int):
        assert not use_caching
        x = jnp.full(batch_shape, self.n_classes)
        timesteps = jnp.linspace(1, 0, n_steps + (1 if denoise else 0))
        alphas = self.alpha(timesteps)
        full_projector = lambda x: self.replace_bos(projector(x))
        x = full_projector(x)

        def update(i, carry):
            key, x, last_probs, was_updated = carry
            key, subkey = jax.random.split(key)
            a_prev = jnp.full(x.shape, alphas[i])
            a_post = jnp.full(x.shape, alphas[i + 1])
            def compute_probs():
                logits = self.process_logits(score_fn(x, a_prev[..., 0])).astype(jnp.float32)
                probs = jax.nn.softmax(logits, axis=-1)
                return probs
            # TODO any way to bucket at large batch sizes?
            probs = jax.lax.switch(was_updated.any().astype(jnp.uint8), (lambda: last_probs, compute_probs))
            probs_full = (probs * (a_post - a_prev)[..., None]).at[..., self.n_classes].set(1 - a_post) / (1 - a_prev)[..., None]
            new_x = jnp.where(x == self.n_classes, jax.random.categorical(subkey, jnp.log(1e-10 + probs_full)), x)
            new_x = full_projector(new_x)
            return key, new_x, probs, new_x != x
        key, x, _, _ = jax.lax.fori_loop(0, n_steps, update, (key, x, jnp.zeros(x.shape + (vocab_size,), dtype=jnp.float32), jnp.ones(batch_shape, dtype=jnp.bool_)))

        if denoise:
            # denoising step
            t = jnp.full(x.shape, alphas[-1])
            x = jnp.where(x == self.n_classes, self.process_logits(score_fn(x, t[..., 0])).argmax(-1), x)
        return full_projector(x)
