from dataclasses import dataclass, field
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from shortcut.targets_shortcut import PolicyShortCutSampler, ShortCutSampler
from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.mytypes import Batched
from utils.networks import GCActorVectorField, UnconditionalEmbedding


class CFGRLAgent(flax.struct.PyTreeNode):
    """Classifier-free guidance reinforcement learning (CFGRL) agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def actor_loss(self, batch, grad_params, rng=None):
        """Compute the behavioral flow-matching actor loss."""
        batch_size, action_dim = batch['actions'].shape
        # rng for each input to maybe condition cfg
        rng, x_rng, t_rng, gcfg_rng, scfg_rng = jax.random.split(rng, 5)

        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['actions']
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        unc_embed = self.network.select('unc_embed')(params=grad_params)  # (1, goal_dim)
        do_gcfg = jax.random.bernoulli(gcfg_rng, p=0.1, shape=(batch_size,))
        goals = jnp.where(do_gcfg[:, None], unc_embed, batch['actor_goals'])

        unc_step_embed = self.network.select('unc_step_embed')(params=grad_params)  # (1, 4)
        step_embed = self.network.select('unc_step_embed')(batch['actor_offsets'], params=grad_params)
        do_scfg = jax.random.bernoulli(scfg_rng, p=0.5, shape=(batch_size,))
        mask = jnp.logical_and(do_scfg, do_gcfg)
        steps = jnp.where(mask[:, None], unc_step_embed, step_embed[:, 0])

        pred = self.network.select('actor_flow')(
            batch['observations'],
            x_t,
            t,
            goals=goals,
            goal_steps=steps,
            params=grad_params,
        )
        actor_loss = jnp.mean((pred - vel) ** 2)

        return actor_loss, {
            'actor_loss': actor_loss,
        }

    def loss_fn(self, obs, x_t, t, v_t, g, gdt, grad_params):
        pred = self.network.select('actor_flow')(
            obs,
            x_t,
            t,
            goals=g,
            goal_steps=gdt,
            params=grad_params,
        )
        mse_v: Batched = jnp.mean((pred - v_t) ** 2, axis=(1, 2, 3))
        loss = jnp.mean(mse_v)

        info = {
            'actor_loss': loss,
        }
        return loss, info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, targets_rng = jax.random.split(rng, 3)
        # x_t, v_t, t, dt_base, labels, info = get_targets(
        # self.config.shortcut, targets_rng, train_state, images, labels,
        # )
        # actor_loss, actor_info = self.loss_fn(
        # batch,
        # grad_params, actor_rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = actor_loss
        return loss, info

    @jax.jit
    def shortcut_loss(self, batch, info, grad_params):
        pred = self.network.select('actor_flow')(
            batch['obs'],
            batch['x_t'],
            batch['t'],
            goals=batch['goals'],
            goal_steps=batch['dt'],
            params=grad_params,
        )
        actor_loss = jnp.mean((pred - batch['v_t']) ** 2, where=(~batch['is_pad'][:, None]).astype(bool))
        return actor_loss, (info | {'actor_loss': actor_loss})

    def update_shortcut(self, ds):
        new_rng, rng, shortcut_rng = jax.random.split(self.rng, 3)

        batch, info = self.config.shortcut.policy_shortcut(shortcut_rng, self.network, ds)

        def loss_fn(grad_params):
            return self.shortcut_loss(batch, info, grad_params)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        goal_steps=None,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the actor."""
        if self.config.encoder is not None:
            observations = self.network.select('actor_flow_encoder')(observations)

        action_seed, noise_seed = jax.random.split(seed)
        actions = jax.random.normal(
            action_seed,
            (*observations.shape[:-1], self.config.action_dim),
        )

        unc_embed = self.network.select('unc_embed')()[0]
        unc_step_embed = self.network.select('unc_step_embed')()[0]
        for i in range(self.config.flow_steps):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config.flow_steps)

            unc_vels = self.network.select('actor_flow')(
                observations,
                actions,
                t,
                goals=unc_embed,
                goal_steps=unc_step_embed,
                is_encoded=True,
            )
            g_vels = self.network.select('actor_flow')(
                observations,
                actions,
                t,
                goals=goals,
                goal_steps=unc_step_embed,
                is_encoded=True,
            )

            steps = self.network.select('unc_step_embed')(goal_steps)[0] if goal_steps is not None else unc_step_embed
            gt_vels = self.network.select('actor_flow')(
                observations,
                actions,
                t,
                goals=goals,
                goal_steps=steps,
                is_encoded=True,
            )
            vels = unc_vels + self.config.cfg * (g_vels - unc_vels) + self.config.cfg2 * (gt_vels - g_vels)

            actions = actions + vels / self.config.flow_steps

        actions = jnp.clip(actions, -1, 1)

        return actions

    @classmethod
    def create(
        cls,
        seed,
        example_batch,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            example_batch: Example batch.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_observations = example_batch['observations']
        ex_actions = example_batch['actions']
        ex_goals = example_batch['value_goals']
        ex_times = ex_actions[..., :1]
        ex_offsets = example_batch['actor_offsets']
        action_dim = ex_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config.encoder is not None:
            encoder_module = encoder_modules[config.encoder]
            encoders['actor_flow'] = encoder_module()

        # Define networks.
        actor_flow_def = GCActorVectorField(
            hidden_dims=config.actor_hidden_dims,
            action_dim=action_dim,
            layer_norm=config.actor_layer_norm,
            encoder=encoders.get('actor_flow'),
        )

        unc_embed_def = UnconditionalEmbedding(
            goal_dim=ex_goals.shape[-1],
        )

        step_dim = 1
        unc_step_embed_def = UnconditionalEmbedding(num_embeddings=1000, goal_dim=step_dim)

        network_info = dict(
            actor_flow=(
                actor_flow_def,
                (
                    ex_observations,
                    ex_actions,
                    ex_times,
                    ex_goals,
                    jnp.zeros((ex_offsets.shape[0], step_dim), dtype=jnp.uint8),
                ),
            ),
            unc_embed=(unc_embed_def, ()),
            unc_step_embed=(unc_step_embed_def, (ex_offsets)),
        )
        if encoders.get('actor_flow') is not None:
            # Add actor_flow_encoder to ModuleDict to make it separately callable.
            network_info['actor_flow_encoder'] = (
                encoders.get('actor_flow'),
                (ex_observations,),
            )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config.lr)
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        config.action_dim = action_dim
        return cls(rng, network=network, config=config)


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='cfgrl',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor.
            flow_steps=16,  # Number of flow steps.
            cfg=4.0,  # CFG coefficient.
            cfg2=4.0,  # CFG coefficient for time conditioning.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            # Dataset hyperparameters.
            dataset_class='GCDataset',  # Dataset class name.
            value_p_curgoal=0.0,  # Unused (defined for compatibility with GCDataset).
            value_p_trajgoal=1.0,  # Unused (defined for compatibility with GCDataset).
            value_p_randomgoal=0.0,  # Unused (defined for compatibility with GCDataset).
            value_geom_sample=False,  # Unused (defined for compatibility with GCDataset).
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            gc_negative=True,  # Unused (defined for compatibility with GCDataset).
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
        )
    )
    return config


@dataclass
class CFGRLConfig:
    """Configuration for the CFGRL agent."""

    ### Agent hyperparameters
    agent_name: str = 'cfgrl'
    lr: float = 3e-4
    batch_size: int = 1024
    actor_hidden_dims: tuple = (512, 512, 512, 512)
    actor_layer_norm: bool = False
    discount: float = 0.99
    flow_steps: int = 16
    cfg: float = 3.0
    cfg2: float = 3.0
    encoder: str | None = None  # Visual encoder name (None, 'impala_small', etc.).
    action_dim: int | None = None  # Action dimension (will be set automatically).
    dataset_class: str = 'GCDataset'
    value_p_curgoal: float = 0.0

    ### Dataset hyperparameters
    value_p_curgoal: float = 0.0  # Unused (defined for compatibility with GCDataset).
    value_p_trajgoal: float = 1.0  # Unused (defined for compatibility with GCDataset).
    value_p_randomgoal: float = 0.0  # Unused (defined for compatibility with GCDataset).
    value_geom_sample: bool = False  # Unused (defined for compatibility with GCDataset).
    actor_p_curgoal: float = 0.0  # prob goal=state_cur
    actor_p_trajgoal: float = 1.0  # prob goal=state_future
    actor_p_randomgoal: float = 0.0  # prob goal=state_random
    actor_geom_sample: bool = False  # Whether to use geometric sampling for future actor goals.
    gc_negative: bool = True  # Unused (defined for compatibility with GCDataset).
    gc_negative: bool = True
    p_aug: float = 0.0
    frame_stack: int | None = None  # Number of frames to stack.

    shortcut: ShortCutSampler | None = field(default_factory=PolicyShortCutSampler)

    def __post_init__(self):
        self.shortcut._batch_size = self.batch_size
