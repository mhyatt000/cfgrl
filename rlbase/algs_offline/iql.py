from absl import app, flags
from functools import partial
import numpy as np
import tqdm
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import optax
import wandb
from ml_collections import config_flags
from flax.training import checkpoints
import ml_collections
from typing import Any

from utils.gc_dataset import GCDataset
from utils.wandb import setup_wandb, default_wandb_config, get_flag_dict
from utils.evaluation import evaluate
from utils.train_state import TrainState, target_update, supply_rng
from utils.networks import Policy, ValueCritic, Critic, ensemblize
from envs.env_helper import make_env, get_dataset


###############################
#  Configs
###############################


FLAGS = flags.FLAGS
flags.DEFINE_string('env_name', 'gc-antmaze-large-diverse-v2', 'Environment name.')
flags.DEFINE_integer('seed', np.random.choice(1000000), 'Random seed.')
flags.DEFINE_integer('eval_episodes', 20, 'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 50000, 'Eval interval.')
flags.DEFINE_integer('video_interval', 250000, 'Video interval.')
flags.DEFINE_integer('batch_size', 1024, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_integer('use_validation', 0, 'Whether to use validation or not.')

# These variables are passed to the IQLAgent class.
agent_config = ml_collections.ConfigDict({
    'goal_conditioned': 0,
    'actor_lr': 3e-4,
    'value_lr': 3e-4,
    'critic_lr': 3e-4,
    'num_qs': 2,
    'actor_hidden_dims': (512, 512, 512),
    'hidden_dims': (512, 512, 512),
    'discount': 0.99,
    'expectile': 0.9,
    'temperature': 3.0, # 0 for behavior cloning.
    'dropout_rate': 0,
    'use_tanh': 0,
    'state_dependent_std': 0,
    'use_layer_norm': 1,
    'activation': 'mish',
    'fixed_std': 0,
    'tau': 0.005,
    'opt_decay_schedule': 'none',
    'actor_loss_type': 'awr', # or ddpg
    'ddpg_bc_weight': 0.0,
    'target_extraction': 1,
})

wandb_config = default_wandb_config()
wandb_config.update({
    'project': 'rlbase_default',
    'name': 'iql_{env_name}',
})

config_flags.DEFINE_config_dict('wandb', wandb_config, lock_config=False)
config_flags.DEFINE_config_dict('agent', agent_config, lock_config=False)
config_flags.DEFINE_config_dict('gcdataset', GCDataset.get_default_config(), lock_config=False)

###############################
#  Agent. Contains the neural networks, training logic, and sampling.
###############################

def expectile_loss(diff, expectile=0.8):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)

class IQLAgent(flax.struct.PyTreeNode):
    rng: Any
    critic: TrainState
    target_critic: TrainState
    value: TrainState
    actor: TrainState
    config: dict = flax.struct.field(pytree_node=False)

    @jax.jit
    def update(agent, batch):
        def critic_loss_fn(critic_params):
            next_v = agent.value(batch['next_observations'])
            target_q = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_v
            qs = agent.critic(batch['observations'], batch['actions'], params=critic_params) # [num_q, batch]
            critic_loss = ((qs - target_q[None])**2).mean()
            return critic_loss, {
                'critic_loss': critic_loss,
                'q': qs[0].mean(),
            }
        
        def value_loss_fn(value_params):
            qs = agent.target_critic(batch['observations'], batch['actions'])
            q = jnp.min(qs, axis=0) # Min over ensemble.
            v = agent.value(batch['observations'], params=value_params)
            value_loss = expectile_loss(q-v, agent.config['expectile']).mean()
            return value_loss, {
                'value_loss': value_loss,
                'v': v.mean(),
                'v_min': v.min(),
                'v_max': v.max(),
            }
        
        def actor_loss_fn(actor_params):
            if agent.config['actor_loss_type'] == 'awr':
                v = agent.value(batch['observations_policy'])
                if agent.config['target_extraction']:
                    qs = agent.target_critic(batch['observations_policy'], batch['actions'])
                else:
                    qs = agent.critic(batch['observations_policy'], batch['actions'])
                q = jnp.min(qs, axis=0) # Min over ensemble.
                exp_a = jnp.exp((q - v) * agent.config['temperature'])
                exp_a = jnp.minimum(exp_a, 100.0)

                # exp_a = (q > 0).astype(jnp.float32)

                actions = batch['actions']
                dist = agent.actor(batch['observations_policy'], params=actor_params)
                log_probs = dist.log_prob(actions)
                actor_loss = -(exp_a * log_probs).mean()

                action_std = dist.stddev().mean()
                return actor_loss, {
                    'actor_q': q.mean(),
                    'actor_loss': actor_loss,
                    'action_std': action_std,
                    'adv': (q - v).mean(),
                    'adv_min': (q - v).min(),
                    'adv_max': (q - v).max(),
                    'accept_prob': ((q-v) > 0).mean(),
                    'bc_log_probs': log_probs.mean(),
                    'actor_mse': ((actions - dist.mean())**2).mean(),
                    'actor_losses': exp_a * log_probs,
                }
            elif agent.config['actor_loss_type'] == 'ddpg':
                dist = agent.actor(batch['observations'], params=actor_params)
                normalized_actions = jnp.tanh(dist.loc)
                qs = agent.critic(batch['observations'], normalized_actions)
                q = jnp.min(qs, axis=0) # Min over ensemble.

                q_loss = -q.mean()
                bc_loss = -dist.log_prob(batch['actions']).mean()

                actor_loss = q_loss.mean() + agent.config['ddpg_bc_weight'] * bc_loss
                return actor_loss, {
                    'actor_loss': actor_loss,
                    'bc_loss': bc_loss,
                    'actor_q': q.mean(),
                }
            else:
                raise NotImplementedError
        
        new_critic, critic_info = agent.critic.apply_loss_fn(loss_fn=critic_loss_fn, has_aux=True)
        new_target_critic = target_update(agent.critic, agent.target_critic, agent.config['tau'])
        new_value, value_info = agent.value.apply_loss_fn(loss_fn=value_loss_fn, has_aux=True)
        new_actor, actor_info = agent.actor.apply_loss_fn(loss_fn=actor_loss_fn, has_aux=True)

        return agent.replace(critic=new_critic, target_critic=new_target_critic, value=new_value, actor=new_actor), {
            **critic_info, **value_info, **actor_info
        }

    @jax.jit
    def sample_actions(agent, observations: np.ndarray, *, seed: Any, temperature: float = 1.0) -> jnp.ndarray:
        if type(observations) is dict:
            observations = jnp.concatenate([observations['observation'], observations['goal']], axis=-1)
        actions = agent.actor(observations, temperature=temperature).sample(seed=seed)
        if agent.config['actor_loss_type'] == 'ddpg':
            actions = jnp.tanh(actions)
        actions = jnp.clip(actions, -1, 1)
        return actions

# Initializes all the networks, etc. for the agent.
def create_agent(
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
            **kwargs):

        print('Extra kwargs:', kwargs)
        config = kwargs

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key = jax.random.split(rng, 4)

        action_dim = actions.shape[-1]
        if config['actor_hidden_dims'] == 'small':
            config['actor_hidden_dims'] = (256, 256)

        mlp_kwargs = dict(activation=config['activation'], use_layer_norm=config['use_layer_norm'])

        actor_def = Policy(config['actor_hidden_dims'], action_dim=action_dim, 
            fixed_std=config['fixed_std'] if config['fixed_std'] != 0 else None,
            log_std_min=-5.0, state_dependent_std=config['state_dependent_std'], tanh_squash_distribution=config['use_tanh'], mlp_kwargs=mlp_kwargs)
        
        if config['opt_decay_schedule'] == "cosine":
            schedule_fn = optax.cosine_decay_schedule(-config['actor_lr'], config['max_steps'])
            actor_tx = optax.chain(optax.scale_by_adam(),
                                    optax.scale_by_schedule(schedule_fn))
        else:
            actor_tx = optax.adam(learning_rate=config['actor_lr'])

        actor_params = actor_def.init(actor_key, observations)['params']
        actor = TrainState.create(actor_def, actor_params, tx=actor_tx)

        critic_def = ensemblize(Critic, num_qs=config['num_qs'])(config['hidden_dims'], mlp_kwargs=mlp_kwargs)
        critic_params = critic_def.init(critic_key, observations, actions)['params']
        critic = TrainState.create(critic_def, critic_params, tx=optax.adam(learning_rate=config['critic_lr']))
        target_critic = TrainState.create(critic_def, critic_params)

        value_def = ValueCritic(config['hidden_dims'], mlp_kwargs=mlp_kwargs)
        value_params = value_def.init(value_key, observations)['params']
        value = TrainState.create(value_def, value_params, tx=optax.adam(learning_rate=config['value_lr']))

        config_dict = flax.core.FrozenDict(**config)
        return IQLAgent(rng, critic=critic, target_critic=target_critic, value=value, actor=actor, config=config_dict)


###############################
#  Run Script. Loads data, logs to wandb, and runs the training loop.
###############################


def main(_):
    if FLAGS.agent.goal_conditioned:
        assert 'gc' in FLAGS.env_name
    else:
        assert 'gc' not in FLAGS.env_name

    np.random.seed(FLAGS.seed)

    # Create wandb logger
    setup_wandb(FLAGS.agent.to_dict(), **FLAGS.wandb)
    
    env = make_env(FLAGS.env_name)
    eval_env = make_env(FLAGS.env_name)

    dataset = get_dataset(env, FLAGS.env_name)

    if FLAGS.agent.goal_conditioned:
        dataset = GCDataset(dataset, **FLAGS.gcdataset.to_dict())
        example_batch = dataset.sample(1)
        example_obs = np.concatenate([example_batch['observations'], example_batch['goals']], axis=-1)
        debug_batch = dataset.sample(100)
        print("Masks Look Like", debug_batch['masks'])
        print("Rewards Look Like", debug_batch['rewards'])
    else:
        example_obs = dataset.sample(1)['observations']
        example_batch = dataset.sample(1)
    print("Obs shape:", example_obs.shape)

    if FLAGS.use_validation:
        dataset, dataset_valid = dataset.train_valid_split(0.9)


    agent = create_agent(FLAGS.seed,
                    example_obs,
                    example_batch['actions'],
                    max_steps=FLAGS.max_steps,
                    **FLAGS.agent)
    
    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1),
                       smoothing=0.1,
                       dynamic_ncols=True):

        batch = dataset.sample(FLAGS.batch_size)
        if FLAGS.agent.goal_conditioned:
            batch['observations_policy'] = np.concatenate([batch['observations'], batch['policy_goals']], axis=-1)
            batch['observations'] = np.concatenate([batch['observations'], batch['goals']], axis=-1)
            batch['next_observations'] = np.concatenate([batch['next_observations'], batch['goals']], axis=-1)
        else:
            batch['observations_policy'] = batch['observations']

        agent, update_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            wandb.log(train_metrics, step=i)

            if FLAGS.use_validation:
                batch = dataset_valid.sample(FLAGS.batch_size)
                if FLAGS.agent.goal_conditioned:
                    batch['observations_policy'] = np.concatenate([batch['observations'], batch['policy_goals']], axis=-1)
                    batch['observations'] = np.concatenate([batch['observations'], batch['goals']], axis=-1)
                    batch['next_observations'] = np.concatenate([batch['next_observations'], batch['goals']], axis=-1)
                else:
                    batch['observations_policy'] = batch['observations']
                _, valid_update_info = agent.update(batch)
                valid_metrics = {f'validation/{k}': v for k, v in valid_update_info.items()}
                wandb.log(valid_metrics, step=i)

                wandb.log({'training/actor_valid_difference': (valid_update_info['actor_loss'] - update_info['actor_loss'])}, step=i)
                wandb.log({'training/critic_valid_difference': (valid_update_info['critic_loss'] - update_info['critic_loss'])}, step=i)

        if i % FLAGS.eval_interval == 0 or i == 1:
            record_video = i % FLAGS.video_interval == 0
            policy_fn = partial(supply_rng(agent.sample_actions), temperature=0.0)
            eval_info, trajs = evaluate(policy_fn, eval_env, num_episodes=FLAGS.eval_episodes, record_video=record_video, return_trajectories=True)

            eval_metrics = {}
            for k in ['episode.return', 'episode.length']:
                eval_metrics[f'evaluation/{k}'] = eval_info[k]
                print(f'evaluation/{k}: {eval_info[k]}')
            try:
                eval_metrics['evaluation/episode.return.normalized'] = eval_env.get_normalized_score(eval_info['episode.return'])
                print(f'evaluation/episode.return.normalized: {eval_metrics["evaluation/episode.return.normalized"]}')
            except:
                pass
            if record_video:
                wandb.log({'video': eval_info['video']}, step=i)

            # Antmaze Specific Logging
            if 'antmaze-large' in FLAGS.env_name or 'maze2d-large' in FLAGS.env_name:
                import envs.d4rl.d4rl_ant as d4rl_ant
                # Make an image of the trajectories.
                traj_image = d4rl_ant.trajectory_image(eval_env, trajs)
                eval_metrics['trajectories'] = wandb.Image(traj_image)

                # Make an image of the value function.
                if 'antmaze-large' in FLAGS.env_name or 'maze2d-large' in FLAGS.env_name:
                    def get_gcvalue(state, goal):
                        obgoal = jnp.concatenate([state, goal], axis=-1)
                        return agent.value(obgoal)
                    pred_value_img = d4rl_ant.value_image(eval_env, dataset, get_gcvalue)
                    eval_metrics['v'] = wandb.Image(pred_value_img)

                # Maze2d Action Distribution
                if 'maze2d-large' in FLAGS.env_name:
                    # Make a plot of the actions.
                    traj_actions = np.concatenate([t['action'] for t in trajs], axis=0) # (T, A)
                    import matplotlib.pyplot as plt
                    plt.figure()
                    plt.scatter(traj_actions[::100, 0], traj_actions[::100, 1], alpha=0.4)
                    plt.xlim(-1.05, 1.05)
                    plt.ylim(-1.05, 1.05)
                    wandb.log({'actions_traj': wandb.Image(plt)}, step=i)

                    data_actions = batch['actions']
                    import matplotlib.pyplot as plt
                    plt.figure()
                    plt.scatter(data_actions[:, 0], data_actions[:, 1], alpha=0.2)
                    plt.xlim(-1.05, 1.05)
                    plt.ylim(-1.05, 1.05)
                    wandb.log({'actions_data': wandb.Image(plt)}, step=i)

            wandb.log(eval_metrics, step=i)

if __name__ == '__main__':
    app.run(main)