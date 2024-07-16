from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass
class AbsorbingDiffusion:
    n_classes: int
    noise_eps: float = 1e-3

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
        esigm1 = jnp.where(
            sigma < 0.5,
            jnp.expm1(sigma),
            jnp.exp(sigma) - 1
        )

        ratio = 1 / jnp.repeat(esigm1, x.shape[-1], -1)
        other_ind = x0

        # negative_term
        neg_term = ratio * jnp.take_along_axis(score, other_ind[..., None], -1).squeeze(-1)

        # positive term
        pos_term = jnp.exp(score[..., :-1]).sum(axis=-1)

        # constant term
        const = ratio * (jnp.log(ratio) - 1)

        rel_ind = x == self.n_classes
        entropy = jnp.where(rel_ind, pos_term - neg_term + const, jnp.zeros(x.shape, score.dtype))
        return entropy

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

        x = self.sample_limit(batch_shape)
        timesteps = jnp.linspace(1, self.noise_eps, n_steps + 1)
        dt = (1 - self.noise_eps) / max(n_steps, 1)

        def update(i, carry):
            key, x = carry
            key, subkey = jax.random.split(key)
            t = timesteps[i] * jnp.ones(x.shape)
            x = projector(x)
            probs = update_fn(score_fn, x, t, dt)
            x = jax.random.categorical(subkey, probs)
            return key, x
        key, x = jax.lax.fori_loop(0, n_steps, update, (key, x))

        if denoise:
            # denoising step
            x = projector(x)
            t = timesteps[-1] * jnp.ones(x.shape)
            probs = update_fn(score_fn, x, t, dt)
            x = probs[..., :-1].argmax(-1)
        return x

    # https://github.com/neverix/Score-Entropy-Discrete-Diffusion/blob/f7221e3b835045f75444c7429955aa420111cc7d/noise_lib.py#L56
    def noise_schedule(self, t):
        total_noise = -jnp.log1p(-(1 - self.noise_eps) * t)
        rate_noise = (1 - self.noise_eps) / (1 - (1 - self.noise_eps) * t)
        return total_noise, rate_noise

    # https://github.com/neverix/Score-Entropy-Discrete-Diffusion/blob/f7221e3b835045f75444c7429955aa420111cc7d/graph_lib.py#L234C1-L239C21
    def staggered_score(self, score, dsigma):
        dse = jnp.exp(dsigma)
        extra_const = (1 - dse) * score.sum(axis=-1)
        score = (score * dse[..., None]).at[..., -1].add(extra_const)
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
