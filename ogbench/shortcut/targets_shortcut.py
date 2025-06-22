"""borrowed from shorcut-models"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np


@dataclass
class ShortCutSampler:
    # Probability of dropping the goal label in flow-matching
    drop_prob: float = 0.1
    # Integer representing the 'null' condition for dropped labels
    num_classes: int = 1000
    # Total number of denoising steps (must be power-of-two)
    denoise_timesteps: int = 128
    # Scale for classifier-free guidance in bootstrap targets
    cfg_scale: float = 4.0
    # Number of examples between each bootstrap batch
    bootstrap_every: int = 8
    # Whether to use EMA model for bootstrap targets (1 = EMA, 0 = student)
    bootstrap_ema: int = 1
    # Enable classifier-free guidance during bootstrap ('bootstrap_cfg' flag)
    bootstrap_cfg: bool = False
    # Bias setting for sampling dt_base (0 = uniform, non-zero = biased schedule)
    bootstrap_dt_bias: int = 0

    _batch_size: int | None = None


@dataclass
class FlowShortCutSampler(ShortCutSampler):
    def get_targets(self, key, train_state, images, labels, force_t=-1, force_dt=-1):
        """here images are the input and output, labels are guidance
        compare to: obs-input action-output goal-guidance
        """

        label_key, time_key, noise_key = jax.random.split(key, 3)
        info = {}

        bs = images.shape[0]

        # 1) =========== Sample dt. ============
        bbs = bs // self.bootstrap_every  # bootstrap batch size
        log2_sections = np.log2(self.denoise_timesteps).astype(np.int32)

        if self.bootstrap_dt_bias == 0:
            dt_base = jnp.repeat(log2_sections - 1 - jnp.arange(log2_sections), bbs // log2_sections)
            dt_base = jnp.concatenate([dt_base, jnp.zeros(bbs - dt_base.shape[0])])
            num_dt_cfg = bbs // log2_sections
        else:
            dt_base = jnp.repeat(
                log2_sections - 1 - jnp.arange(log2_sections - 2),
                (bbs // 2) // log2_sections,
            )
            dt_base = jnp.concatenate([dt_base, jnp.ones(bbs // 4), jnp.zeros(bbs // 4)])
            dt_base = jnp.concatenate([dt_base, jnp.zeros(bbs - dt_base.shape[0])])
            num_dt_cfg = (bbs // 2) // log2_sections

        force_dt_vec = jnp.ones(bbs, dtype=jnp.float32) * force_dt
        dt_base = jnp.where(force_dt_vec != -1, force_dt_vec, dt_base)
        dt = 1 / (2 ** (dt_base))  # [1, 1/2, 1/4, 1/8, 1/16, 1/32]
        dt_base_bootstrap = dt_base + 1
        dt_bootstrap = dt / 2

        # 2) =========== Sample t. ============
        dt_sections = jnp.power(2, dt_base)  # [1, 2, 4, 8, 16, 32]
        t = jax.random.randint(time_key, (bootstrap_batchsize,), minval=0, maxval=dt_sections).astype(jnp.float32)
        t = t / dt_sections  # Between 0 and 1.
        force_t_vec = jnp.ones(bootstrap_batchsize, dtype=jnp.float32) * force_t
        t = jnp.where(force_t_vec != -1, force_t_vec, t)
        t_full = t[:, None, None, None]

        # 3) =========== Generate Bootstrap Targets ============
        x_1 = images[:bootstrap_batchsize]
        x_0 = jax.random.normal(noise_key, x_1.shape)
        x_t = (1 - (1 - 1e-5) * t_full) * x_0 + t_full * x_1
        bst_labels = labels[:bootstrap_batchsize]
        call_model_fn = train_state.call_model if self.bootstrap_ema == 0 else train_state.call_model_ema
        if not self.bootstrap_cfg:
            v_b1 = call_model_fn(x_t, t, dt_base_bootstrap, bst_labels, train=False)
            t2 = t + dt_bootstrap
            x_t2 = x_t + dt_bootstrap[:, None, None, None] * v_b1
            x_t2 = jnp.clip(x_t2, -4, 4)
            v_b2 = call_model_fn(x_t2, t2, dt_base_bootstrap, bst_labels, train=False)
            v_target = (v_b1 + v_b2) / 2
        else:
            x_t_extra = jnp.concatenate([x_t, x_t[:num_dt_cfg]], axis=0)
            t_extra = jnp.concatenate([t, t[:num_dt_cfg]], axis=0)
            dt_base_extra = jnp.concatenate([dt_base_bootstrap, dt_base_bootstrap[:num_dt_cfg]], axis=0)
            labels_extra = jnp.concatenate(
                [bst_labels, jnp.ones(num_dt_cfg, dtype=jnp.int32) * self.num_classes],
                axis=0,
            )
            v_b1_raw = call_model_fn(x_t_extra, t_extra, dt_base_extra, labels_extra, train=False)
            v_b_cond = v_b1_raw[: x_1.shape[0]]
            v_b_uncond = v_b1_raw[x_1.shape[0] :]
            v_cfg = v_b_uncond + self.cfg_scale * (v_b_cond[:num_dt_cfg] - v_b_uncond)
            v_b1 = jnp.concatenate([v_cfg, v_b_cond[num_dt_cfg:]], axis=0)

            t2 = t + dt_bootstrap
            x_t2 = x_t + dt_bootstrap[:, None, None, None] * v_b1
            x_t2 = jnp.clip(x_t2, -4, 4)
            x_t2_extra = jnp.concatenate([x_t2, x_t2[:num_dt_cfg]], axis=0)
            t2_extra = jnp.concatenate([t2, t2[:num_dt_cfg]], axis=0)
            v_b2_raw = call_model_fn(x_t2_extra, t2_extra, dt_base_extra, labels_extra, train=False)
            v_b2_cond = v_b2_raw[: x_1.shape[0]]
            v_b2_uncond = v_b2_raw[x_1.shape[0] :]
            v_b2_cfg = v_b2_uncond + self.cfg_scale * (v_b2_cond[:num_dt_cfg] - v_b2_uncond)
            v_b2 = jnp.concatenate([v_b2_cfg, v_b2_cond[num_dt_cfg:]], axis=0)
            v_target = (v_b1 + v_b2) / 2

        v_target = jnp.clip(v_target, -4, 4)
        bst_v = v_target
        bst_dt = dt_base
        bst_t = t
        bst_xt = x_t
        bst_l = bst_labels

        # 4) =========== Generate Flow-Matching Targets ============

        labels_dropout = jax.random.bernoulli(label_key, self.drop_prob, (labels.shape[0],))
        labels_dropped = jnp.where(labels_dropout, self.num_classes, labels)
        info['dropped_ratio'] = jnp.mean(labels_dropped == self.num_classes)

        # Sample t.
        t = jax.random.randint(time_key, (images.shape[0],), minval=0, maxval=self.denoise_timesteps).astype(
            jnp.float32
        )
        t /= self.denoise_timesteps
        force_t_vec = jnp.ones(images.shape[0], dtype=jnp.float32) * force_t
        t = jnp.where(force_t_vec != -1, force_t_vec, t)  # If force_t is not -1, then use force_t.
        t_full = t[:, None, None, None]  # [batch, 1, 1, 1]

        # Sample flow pairs x_t, v_t.
        x_0 = jax.random.normal(noise_key, images.shape)
        x_1 = images
        x_t = x_t = (1 - (1 - 1e-5) * t_full) * x_0 + t_full * x_1
        v_t = v_t = x_1 - (1 - 1e-5) * x_0
        dt_flow = np.log2(self.denoise_timesteps).astype(jnp.int32)
        dt_base = jnp.ones(images.shape[0], dtype=jnp.int32) * dt_flow

        # ==== 5) Merge Flow+Bootstrap ====
        bst_size = bs // self.bootstrap_every
        bst_size_data = bs - bst_size
        x_t = jnp.concatenate([bst_xt, x_t[:bst_size_data]], axis=0)
        t = jnp.concatenate([bst_t, t[:bst_size_data]], axis=0)
        dt_base = jnp.concatenate([bst_dt, dt_base[:bst_size_data]], axis=0)
        v_t = jnp.concatenate([bst_v, v_t[:bst_size_data]], axis=0)
        labels_dropped = jnp.concatenate([bst_l, labels_dropped[:bst_size_data]], axis=0)
        info['bootstrap_ratio'] = jnp.mean(dt_base != dt_flow)

        info['v_magnitude_bootstrap'] = jnp.sqrt(jnp.mean(jnp.square(bst_v)))
        info['v_magnitude_b1'] = jnp.sqrt(jnp.mean(jnp.square(v_b1)))
        info['v_magnitude_b2'] = jnp.sqrt(jnp.mean(jnp.square(v_b2)))

        return x_t, v_t, t, dt_base, labels_dropped, info


class PolicyShortCutSampler(ShortCutSampler):
    """PSGP
    policy shortcut sampler
    targets shortcut policy steps
    """

    def select_fwd(self):
        """select either noisy model (student) or smooth ema model (teacher) to give bootstrap tgt"""
        _fwd = train_state.call_model if self.bootstrap_ema == 0 else train_state.call_model_ema
        return _fwd

    @property
    def bs(self):
        assert self._batch_size is not None, 'batch size must be set first'
        return self._batch_size

    def generate_dt(self, rng, train=True):
        """
        Generate dt for bootstrap sampling.
        """
        bs = self.bs
        assert not train, 'nope'
        assert self.bootstrap_dt_bias == 0, 'nope'

        log2_sections = np.log2(self.denoise_timesteps).astype(np.int32)
        dt_base = jnp.repeat(log2_sections - 1 - jnp.arange(log2_sections), bs // log2_sections)
        dt_base = jnp.concatenate([dt_base, jnp.zeros(bs - dt_base.shape[0])])
        num_dt_cfg = bs // log2_sections

        # force_dt_vec = jnp.ones(bs, dtype=jnp.float32) * force_dt
        # dt_base = jnp.where(force_dt_vec != -1, force_dt_vec, dt_base)
        dt = 1 / (2 ** (dt_base))  # [1, 1/2, 1/4, 1/8, 1/16, 1/32]
        dt_base_bootstrap = dt_base + 1
        dt_bootstrap = dt / 2

        # select 1 at random
        n, m = 1, bs
        vec = jnp.arange(m)  # your data
        idx = jax.random.choice(rng, m, (n,), replace=False)  # indices without replacement
        sample = vec[idx]  # the n elements
        return sample

        return dt_base

    def policy_shortcut(self, rng, network, ds, force_t=-1, force_dt=-1):
        """here images are the input and output, labels are guidance
        compare to: obs-input action-output goal-guidance
        """

        fwd = network.select('actor_flow')
        goal_rng, time_rng, noise_rng = jax.random.split(rng, 3)
        info = {}

        # 1) =========== Sample dt. ============
        bbs = self.bs // self.bootstrap_every  # bootstrap batch size
        log2_sections = np.log2(self.denoise_timesteps).astype(np.int32)

        if self.bootstrap_dt_bias == 0:
            dt_base = jnp.repeat(log2_sections - 1 - jnp.arange(log2_sections), bbs // log2_sections)
            dt_base = jnp.concatenate([dt_base, jnp.zeros(bbs - dt_base.shape[0])])
            num_dt_cfg = bbs // log2_sections
        else:
            dt_base = jnp.repeat(
                log2_sections - 1 - jnp.arange(log2_sections - 2),
                (bbs // 2) // log2_sections,
            )
            dt_base = jnp.concatenate([dt_base, jnp.ones(bbs // 4), jnp.zeros(bbs // 4)])
            dt_base = jnp.concatenate([dt_base, jnp.zeros(bbs - dt_base.shape[0])])
            num_dt_cfg = (bbs // 2) // log2_sections

        force_dt_vec = jnp.ones(bbs, dtype=jnp.float32) * force_dt
        dt_base = jnp.where(force_dt_vec != -1, force_dt_vec, dt_base)
        dt = 1 / (2 ** (dt_base))  # [1, 1/2, 1/4, 1/8, 1/16, 1/32]
        dt_base_bootstrap = dt_base + 1
        dt_bootstrap = dt / 2

        # 2) =========== Sample t. ============
        """
        dt_sections = jnp.power(2, dt_base)  # [1, 2, 4, 8, 16, 32]
        t = jax.random.randint(time_rng, (bbs,), minval=0, maxval=dt_sections).astype(jnp.float32)
        t = t / dt_sections  # Between 0 and 1.
        force_t_vec = jnp.ones(bbs, dtype=jnp.float32) * force_t
        t = jnp.where(force_t_vec != -1, force_t_vec, t)
        t_full = t[:, None]  # b,act
        """

        # dt_base_bootstrap is +1 to denote 1/2 size step
        # 1..7 where 7 is smallest horizon ie 1 policy step
        # horizon maps 7 to 1/128 to horizon=2 so that dt 6 = 2 steps of len 7
        horizon = self.denoise_timesteps / (2**dt_base)
        bbatch = ds.sample(len(dt), horizon=horizon, sparse=True)
        obs, actions, goals = (
            bbatch['observations'],
            bbatch['actions'],
            bbatch['actor_goals'],
        )

        # 3) =========== Generate Bootstrap Targets ============
        x_1 = actions
        x_0 = jax.random.normal(noise_rng, x_1.shape)
        t = jax.random.uniform(time_rng, (bbs, 1))
        eps = 1 - 1e-5  # epsilon to avoid numerical issues
        x_t = (1 - eps * t[:, None]) * x_0 + t[:, None] * x_1
        gdt = dt_base_bootstrap

        if not self.bootstrap_cfg:
            # 3A) Simple two‐step prediction (no CFG)
            v_b1 = fwd(obs[:, 0], x_t[:, 0], t, goals[:, 0], gdt[:, None])  # train=False
            # t2 = t + dt_bootstrap
            # x_t2 = x_t + dt_bootstrap[:, None, None, None] * v_b1
            # x_t2 = jnp.clip(x_t2, -4, 4)
            v_b2 = fwd(obs[:, 1], x_t[:, 1], t, goals[:, 1], gdt[:, None])
            v_target = (v_b1 + v_b2) / 2

        else:
            # 3B) Two‐step with Classifier‐Free Guidance (CFG)
            # — first, build a doubled‐up batch: unconditional + conditional inputs
            raise NotImplementedError()
            x_t_extra = jnp.concatenate([x_t, x_t[:num_dt_cfg]], axis=0)
            t_extra = jnp.concatenate([t, t[:num_dt_cfg]], axis=0)
            dt_base_extra = jnp.concatenate([gdt, gdt[:num_dt_cfg]], axis=0)
            goals_extra = jnp.concatenate(
                [bst_goals, jnp.ones(num_dt_cfg, dtype=jnp.int32) * self.num_classes],
                axis=0,
            )
            v_b1_raw = fwd(x_t_extra, t_extra, dt_base_extra, goals_extra, train=False)
            v_b_cond = v_b1_raw[: x_1.shape[0]]
            v_b_uncond = v_b1_raw[x_1.shape[0] :]
            v_cfg = v_b_uncond + self.cfg_scale * (v_b_cond[:num_dt_cfg] - v_b_uncond)
            v_b1 = jnp.concatenate([v_cfg, v_b_cond[num_dt_cfg:]], axis=0)

            t2 = t + dt_bootstrap
            x_t2 = x_t + dt_bootstrap[:, None, None, None] * v_b1
            x_t2 = jnp.clip(x_t2, -4, 4)
            x_t2_extra = jnp.concatenate([x_t2, x_t2[:num_dt_cfg]], axis=0)
            t2_extra = jnp.concatenate([t2, t2[:num_dt_cfg]], axis=0)
            v_b2_raw = fwd(x_t2_extra, t2_extra, dt_base_extra, goals_extra, train=False)
            v_b2_cond = v_b2_raw[: x_1.shape[0]]
            v_b2_uncond = v_b2_raw[x_1.shape[0] :]
            v_b2_cfg = v_b2_uncond + self.cfg_scale * (v_b2_cond[:num_dt_cfg] - v_b2_uncond)
            v_b2 = jnp.concatenate([v_b2_cfg, v_b2_cond[num_dt_cfg:]], axis=0)
            v_target = (v_b1 + v_b2) / 2

        v_target = jnp.clip(v_target, -4, 4)
        bst_obs = obs[:, 0]
        bst_v = v_target
        bst_dt = dt_base  # ensures that target is conditioned on 2*dt
        bst_t = t
        bst_g = goals[:, 1]  # ensure goal for the bootstrap must be after obs[1]

        choice = jax.random.randint(noise_rng, (), 0, 3)
        bst_xt = jnp.stack([x_t[:, 0], x_t[:, 1], (x_t[:, 0] + x_t[:, 1]) / 2])[choice]

        # 4) =========== Generate Flow-Matching Targets ============
        batch = ds.sample(nbbs := (self.bs - len(dt)))  # for non-bootstrap batch
        obs, actions, goals = (
            batch['observations'],
            batch['actions'],
            batch['actor_goals'],
        )

        Obs = jt.Float[jt.Array, '#bs obs']
        Goal = jt.Float[jt.Array, '#bs goal']  # TODO goal should be obs
        Act = jt.Float[jt.Array, '#bs act']
        B = jt.Float[jt.Array, 'bs']
        B1 = jt.Float[jt.Array, 'bs 1']  # for single timestep

        # maybe drop goal condition
        unc_embed: Goal = network.select('unc_embed')()
        do_gcfg: B = jax.random.bernoulli(goal_rng, self.drop_prob, (nbbs,))
        goals = jnp.where(do_gcfg[:, None], unc_embed, goals)
        info['dropped_ratio'] = do_gcfg.sum() / len(do_gcfg)

        # Sample t.
        """
        rand_t = partial(jax.random.randint, minval=0, maxval=self.denoise_timesteps)
        t: B = rand_t(time_rng, (nbbs,)).astype(jnp.float32)
        t /= self.denoise_timesteps
        force_t_vec = jnp.ones(nbbs, dtype=jnp.float32) * force_t
        # If force_t is not -1, then use force_t.
        t = jnp.where(force_t_vec != -1, force_t_vec, t)
        t_full: Act = t[:, None]
        """

        # Sample flow pairs x_t, v_t.
        x_1: Act = actions
        x_0: Act = jax.random.normal(noise_rng, actions.shape)
        t: B1 = jax.random.uniform(time_rng, (nbbs, 1))
        x_t = (1 - eps * t) * x_0 + t * x_1
        v_t = x_1 - eps * x_0
        v_t = x_1 - x_0
        dt_flow = np.log2(self.denoise_timesteps).astype(jnp.int32)
        # when denoise steps is 128, dt_flow is 7
        # dt=7 refers to the smallest denoise step, i.e. 1/128
        dt_base: B = jnp.ones(nbbs, dtype=jnp.int32) * dt_flow

        # ==== 5) Merge Flow+Bootstrap ====
        bst_size = self.bs // self.bootstrap_every
        bst_size_data = self.bs - bst_size

        obs = jnp.concatenate([bst_obs, obs[:bst_size_data]], axis=0)
        x_t = jnp.concatenate([bst_xt, x_t], axis=0)
        t = jnp.concatenate([bst_t, t[:bst_size_data]], axis=0)
        dt_base = jnp.concatenate([bst_dt, dt_base[:bst_size_data]], axis=0)
        v_t = jnp.concatenate([bst_v, v_t[:bst_size_data]], axis=0)
        goals = jnp.concatenate([bst_g, goals[:bst_size_data]], axis=0)

        # true if bootstrap[1] is out of range
        is_pad = jnp.concatenate([bbatch['is_pad'][:, 1], batch['is_pad']], axis=0)

        info['bootstrap_ratio'] = jnp.mean(dt_base != dt_flow)
        info['v_magnitude_bootstrap'] = jnp.sqrt(jnp.mean(jnp.square(bst_v)))
        info['v_magnitude_b1'] = jnp.sqrt(jnp.mean(jnp.square(v_b1)))
        info['v_magnitude_b2'] = jnp.sqrt(jnp.mean(jnp.square(v_b2)))

        # out_batch is bbs and nbbs combo
        out_batch = {
            'obs': obs,  # the input obs of first policy time
            'is_pad': is_pad,
            'x_t': x_t,
            'goals': goals,  # the goals of bbs and nbbs with maybe drop
            'v_t': v_t,  # target vel is mean of two bootstrap steps @ t,t+d
            't': t,
            'dt': dt_base[:, None],
        }
        return out_batch, info
        # return x_t, v_t, t, dt_base, goals, info
