from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp


@dataclass
class AbsorbingDiffusion:
    n_classes: int
    noise_eps: float = 1e-3
    z_loss_coeff: float = 0

    def get_loss(self, key, score_fn, data):
        noise_key, transition_key = jax.random.split(key, 2)
        t = jax.random.uniform(noise_key, data.shape[:-1] + (1,))
        total_noise, rate_noise = self.noise_schedule(t)
        data_perturbed = self.sample_transition(transition_key, data, total_noise)
        logits = score_fn(data_perturbed, total_noise)
        loss = (self.score_entropy(logits, total_noise, data_perturbed, data) * rate_noise).sum(1)
        return loss

    # https://github.com/neverix/Score-Entropy-Discrete-Diffusion/blob/f7221e3b835045f75444c7429955aa420111cc7d/graph_lib.py#L228C1-L232C22
    def sample_transition(self, key, i, sigma):
        move_chance = 1 - jnp.exp(-sigma)
        move_indices = jax.random.bernoulli(key, move_chance, i.shape)
        i_pert = jnp.where(move_indices, self.n_classes, i)
        return i_pert

    # https://github.com/neverix/Score-Entropy-Discrete-Diffusion/blob/f7221e3b835045f75444c7429955aa420111cc7d/graph_lib.py#L244
    def score_entropy(self, score, sigma, x, x0):
        if score.shape[-1] == self.n_classes:
            score = jnp.pad(score, ((0, 0),) * (score.ndim - 1) + ((0, 1),),)
        assert score.shape[-1] == self.n_classes + 1
        esigm1 = jnp.where(
            sigma < 0.5,
            jnp.expm1(sigma),
            jnp.exp(sigma) - 1
        )

        ratio = 1 / jnp.repeat(esigm1, x.shape[-1], -1)
        other_ind = x0

        rel_ind = x == self.n_classes
        nll_loss = -jnp.take_along_axis(jax.nn.log_softmax(score[..., :-1]), other_ind[..., None], -1).squeeze(-1)
        z_loss = self.z_loss_coeff * (jax.nn.logsumexp(score[..., :-1], axis=-1) ** 2)
        fake_entropy = ratio * (nll_loss + z_loss)
        losses = jnp.where(rel_ind, fake_entropy, jnp.zeros(fake_entropy.shape, fake_entropy.dtype))
        return losses

    def sample(self, score_fn, key, n_steps, batch_shape, denoise=True, projector=lambda x: x):
        # https://github.com/neverix/Score-Entropy-Discrete-Diffusion/blob/f7221e3b835045f75444c7429955aa420111cc7d/sampling.py#L78
        def update_fn(score_fn, x, t, step_size):
            curr_sigma = self.noise_schedule(t)[0]
            next_sigma = self.noise_schedule(t - step_size)[0]
            dsigma = curr_sigma - next_sigma

            score = jnp.exp(score_fn(x, curr_sigma))

            stag_score = self.staggered_score(score, dsigma)
            probs = stag_score * self.transp_transition(x, dsigma)
            return probs

        def denoise_update_fn(score_fn, x, t):
            sigma = self.noise_schedule(t)[0]

            score = jnp.exp(score_fn(x, sigma))
            stag_score = self.staggered_score(score, sigma)
            probs = stag_score * self.transp_transition(x, sigma)
            # truncate probabilities
            probs = probs[..., :-1]

            return probs

        x = self.sample_limit(batch_shape)
        timesteps = jnp.linspace(1, self.noise_eps, n_steps + 1)
        dt = (1 - self.noise_eps) / max(n_steps, 1)

        def update(i, carry):
            key, x = carry
            key, subkey = jax.random.split(key)
            t = timesteps[i] * jnp.ones(x.shape)
            x = projector(x)
            probs = update_fn(score_fn, x, t, dt)
            # jax.debug.print("{}", (probs[..., -1] != 0) == (x == self.n_classes))
            x = jax.random.categorical(subkey, jnp.log(probs + 1e-10))
            # x = jax.random.categorical(subkey, probs / probs.sum(axis=-1, keepdims=True))
            # gumbel_norm = 1e-10 - jnp.log(jax.random.uniform(subkey, shape=probs.shape) + 1e-10)
            # x = (probs / gumbel_norm).argmax(axis=-1)
            return key, x
        key, x = jax.lax.fori_loop(0, n_steps, update, (key, x))

        if denoise:
            # denoising step
            x = projector(x)
            t = timesteps[-1] * jnp.ones(x.shape)
            probs = denoise_update_fn(score_fn, x, t)
            # x = jax.random.categorical(key, probs)
            x = probs.argmax(-1)
        return x

    # https://github.com/neverix/Score-Entropy-Discrete-Diffusion/blob/f7221e3b835045f75444c7429955aa420111cc7d/noise_lib.py#L56
    def noise_schedule(self, t):
        total_noise = -jnp.log1p(-(1 - self.noise_eps) * t)
        rate_noise = (1 - self.noise_eps) / (1 - (1 - self.noise_eps) * t)
        return total_noise, rate_noise

    # https://github.com/neverix/Score-Entropy-Discrete-Diffusion/blob/f7221e3b835045f75444c7429955aa420111cc7d/graph_lib.py#L234C1-L239C21
    def staggered_score(self, score, dsigma):
        dse = jnp.exp(dsigma)
        # extra_const = (1 - dse) * score.sum(axis=-1)
        # extra_const = (1 - dse) * score[..., :-1].sum(axis=-1) + score[..., -1] - dse * score[..., -1]
        # extra_const = (1 - dse) * score[..., :-1].sum(axis=-1) + score[..., -1] - dse * score[..., -1]
        extra_const = (1 - dse) * score[..., :-1].sum(axis=-1) + score[..., -1]
        score = (score * dse[..., None]).at[..., -1].set(extra_const)
        return score

    # https://github.com/neverix/Score-Entropy-Discrete-Diffusion/blob/f7221e3b835045f75444c7429955aa420111cc7d/graph_lib.py#L218
    def transp_transition(self, i, sigma):
        sigma = sigma.reshape(*sigma.shape, *((1,) * (i.ndim + 1 - sigma.ndim)))
        edge = (
            jnp.exp(-sigma) * jax.nn.one_hot(i, num_classes=self.n_classes + 1)
            + jnp.where(
                i == self.n_classes,
                1 - jnp.exp(-sigma).squeeze(-1),
                0
            )[..., None])
        return edge

    def sample_limit(self, dims):
        return jnp.full(dims, self.n_classes)


@dataclass
class MDLMDiffusion:
    n_classes: int
    noise_eps: float = 1e-3

    def get_loss(self, key, score_fn, data):
        t = jnp.linspace(1.0, 0.0, data.shape[0], dtype=jnp.float32)
        for s in data.shape[1:]:
            t = jnp.repeat(t[..., None], s, -1)
        alpha, rate = self.alpha(t), self.alpha_rate(t)
        data_perturbed = self.sample_transition(key, data, alpha)
        logits = self.process_logits(score_fn(data_perturbed, alpha))
        llh = jnp.take_along_axis(jax.nn.log_softmax(logits, axis=-1), data[..., None], -1).squeeze(-1)
        loss = ((rate / (1 - alpha)) * llh * (data_perturbed == self.n_classes))
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
        logits = logits[..., :self.n_classes]
        assert logits.shape[-1] == self.n_classes
        return logits

    # @partial(jax.jit, static_argnames=("use_caching", "denoise", "batch_shape", "n_steps"))
    def sample(self, score_fn, key, n_steps, batch_shape, denoise=True, projector=lambda x: x, use_caching=False):
        assert not use_caching
        x = jnp.full(batch_shape, self.n_classes)
        timesteps = jnp.linspace(1, 0, n_steps + (1 if denoise else 0))
        alphas = self.alpha(timesteps)

        def update(i, carry):
            key, x = carry
            key, subkey = jax.random.split(key)
            a_prev = jnp.full(x.shape, alphas[i])
            a_post = jnp.full(x.shape, alphas[i + 1])
            x = projector(x)
            logits = self.process_logits(score_fn(x, a_prev))
            probs = jax.nn.softmax(logits, axis=-1)
            probs = jnp.concatenate((probs * (a_post - a_prev)[..., None], (1 - a_post)[..., None]), axis=-1) / (1 - a_prev)[..., None]
            x = jax.random.categorical(subkey, jnp.log(1e-10 + probs))
            return key, x
        key, x = jax.lax.fori_loop(0, n_steps, update, (key, x))

        if denoise:
            # denoising step
            x = projector(x)
            t = jnp.full(x.shape, alphas[-1])
            x = score_fn(x, t).argmax(-1)
        return x
