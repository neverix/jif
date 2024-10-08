from dataclasses import dataclass
from typing import Optional

import jax
import jax.experimental
import jax.numpy as jnp
from jax.experimental import checkify
from jax.experimental.shard_alike import shard_alike


@dataclass
class MDLMDiffusion:
    n_classes: int
    noise_eps: float = 1e-3
    z_loss_coeff: float = 1e-4
    bos_token: Optional[int] = None
    
    def replace_bos(self, x):
        if self.bos_token is not None:
            x = x.at[..., 0].set(self.bos_token)
        return x

    # checkify not supported by flash attention
    # @checkify.checkify
    def get_loss(self, key, score_fn, data):
        t = jnp.linspace(1.0, 0.0, data.shape[0], dtype=jnp.float32)
        for _ in range(data.ndim - 1):
            t = t[..., None]
        t = t + jnp.zeros_like(data, dtype=jnp.float32)
        t, data = shard_alike(t, data) # just to make sure
        alpha, rate = self.alpha(t), self.alpha_rate(t)
        data_perturbed = self.sample_transition(key, data, alpha)
        data_perturbed = self.replace_bos(data_perturbed)
        logits = self.process_logits(score_fn(data_perturbed, alpha))
        labels = self.replace_bos(data)[..., None]
        # TODO
        gain = jnp.take_along_axis(jax.nn.log_softmax(logits, -1), labels, -1).squeeze(-1)
        # lse = jax.nn.logsumexp(logits, axis=-1)
        # llh = jnp.take_along_axis(logits, labels, -1).squeeze(-1) - lse
        # z_loss = jnp.square(lse)
        # checkify.check(jnp.all(llh <= 0), "llh must be nonpositive")
        # checkify.check(jnp.all(z_loss >= 0), "z_loss must be non-negative")
        # checkify.check(jnp.all(jnp.isfinite(llh)), "llh must be finite")
        # checkify.check(jnp.all(jnp.isfinite(z_loss)), "z_loss must be finite")
        # gain = llh - self.z_loss_coeff * z_loss
        weights = rate / (1 - alpha)
        loss = jnp.where(data_perturbed == self.n_classes, weights * gain, 0)
        return loss

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
        logits = logits - (jnp.arange(logits.shape[-1], dtype=logits.dtype) >= self.n_classes) * 1e10
        assert logits.shape[-1] >= self.n_classes + 1
        return logits

    # @partial(jax.jit, static_argnames=("use_caching", "denoise", "batch_shape", "n_steps"))
    def sample(self, score_fn, key, n_steps, batch_shape, denoise=True, projector=lambda x: x, use_caching=False):
        assert not use_caching
        x = jnp.full(batch_shape, self.n_classes)
        timesteps = jnp.linspace(1, 0, n_steps + (1 if denoise else 0))
        alphas = self.alpha(timesteps)
        full_projector = lambda x: self.replace_bos(projector(x))
        x = full_projector(x)
        
        # we don't actually do this computation
        vocab_size = score_fn(x, alphas[0]).shape[-1]

        def update(i, carry):
            key, x, last_probs, was_updated = carry
            key, subkey = jax.random.split(key)
            a_prev = jnp.full(x.shape, alphas[i])
            a_post = jnp.full(x.shape, alphas[i + 1])
            def compute_probs():
                logits = self.process_logits(score_fn(x, a_prev)).astype(jnp.float32)
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
            x = jnp.where(x == self.n_classes, self.process_logits(score_fn(x, t)).argmax(-1), x)
        return full_projector(x)
