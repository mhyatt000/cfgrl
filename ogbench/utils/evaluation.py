from collections import defaultdict

import jax
import numpy as np
from tqdm import trange


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def flatten(d, parent_key='', sep='.'):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


class PSGPolicy:
    """policy shortcut guided policy"""

    def __init__(self, agent, rng, max_dt=1):
        self.agent = agent
        self.rng = rng
        self._t = 0  # time until resample
        self.scale = self.agent.config.shortcut.denoise_timesteps
        self.max_dt = max_dt

    def sample_actions(self, *args, **kwargs):
        if self._t == 0:
            act_rng, dt_rng, self.rng = jax.random.split(self.rng, 3)
            self.dt = jax.random.randint(dt_rng, (1,), self.max_dt, 8)
            self._t += self.scale / (2**self.dt)
            self.action = self.agent.sample_actions(*args, **kwargs, goal_steps=self.dt, seed=act_rng)

        self._t -= 1
        return self.action


def evaluate(
    agent,
    env,
    task_id=None,
    config=None,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_temperature=0,
    eval_gaussian=None,
):
    """Evaluate the agent in the environment.

    Args:
        agent: Agent.
        env: Environment.
        task_id: Task ID to be passed to the environment.
        config: Configuration dictionary.
        num_eval_episodes: Number of episodes to evaluate the agent.
        num_video_episodes: Number of episodes to render. These episodes are not included in the statistics.
        video_frame_skip: Number of frames to skip between renders.
        eval_temperature: Action sampling temperature.
        eval_gaussian: Standard deviation of the Gaussian noise to add to the actions.

    Returns:
        A tuple containing the statistics, trajectories, and rendered videos.
    """
    rng = jax.random.PRNGKey(np.random.randint(0, 2**32))
    act_rng, rng = jax.random.split(rng)
    agent = PSGPolicy(agent, act_rng, max_dt=4)
    # actor_fn = supply_rng(agent.sample_actions, rng=act_rng)
    trajs = []
    stats = defaultdict(list)

    renders = []
    for i in trange(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)
        should_render = i >= num_eval_episodes

        observation, info = env.reset(options=dict(task_id=task_id, render_goal=should_render))
        # nstep = np.random.randint(0, 1000, size=len(observation)) if config is not None else 0
        goal = info.get('goal')
        goal_frame = info.get('goal_rendered')
        done = False
        step = 0
        render = []
        while not done:
            # dt_rng, rng = jax.random.split(rng)
            # dt   = jax.random.randint(dt_rng, (), 1, 8)
            # dt = agent.config.shortcut.generate_dt(rng=dt_rng, train=False)
            # action = actor_fn(
            action = agent.sample_actions(
                observations=observation,
                goals=goal,
                # goal_steps=dt,
                temperature=eval_temperature,
            )
            action = np.array(action)
            if eval_gaussian is not None:
                action = np.random.normal(action, eval_gaussian)
            action = np.clip(action, -1, 1)

            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

            if should_render and (step % video_frame_skip == 0 or done):
                frame = env.render().copy()
                if goal_frame is not None:
                    render.append(np.concatenate([goal_frame, frame], axis=0))
                else:
                    render.append(frame)

            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )
            add_to(traj, transition)
            observation = next_observation
        if i < num_eval_episodes:
            add_to(stats, flatten(info))
            trajs.append(traj)
        else:
            renders.append(np.array(render))

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, trajs, renders
