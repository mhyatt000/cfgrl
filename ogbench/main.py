from collections import defaultdict
from dataclasses import asdict, dataclass, field
import json
import os
import random
import time

import jax
import numpy as np
from rich.pretty import pprint
import tqdm

# config_flags.DEFINE_config_file('agent', 'agents/cfgrl.py', lock_config=False)
import tyro
import wandb

from agents import agents
from agents.cfgrl import CFGRLConfig
from utils.datasets import Dataset, GCDataset, HGCDataset
from utils.env_utils import make_gc_env_and_datasets
from utils.evaluation import evaluate
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_wandb_video, setup_wandb
from utils.time_utils import spec


@dataclass
class Config:
    agent_name: str = 'cfgrl'  # Agent name.

    run_group: str = 'Debug'  # Run group for wandb.
    seed: int = 0  # Random seed
    env_name: str = 'antmaze-large-navigate-oraclerep-v0'  # Environment (dataset) name.
    save_dir: str = 'exp/'
    restore_path: str = None
    restore_epoch: int = None

    train_steps: int = 1000000
    log_interval: int = 10000
    eval_interval: int = 500000
    save_interval: int = 1000000

    eval_tasks: int = None  # Number of tasks to evaluate (None for all).
    eval_episodes: int = 15  # Number of episodes for each task.
    eval_temperature: float = 0.0  # Actor temperature for evaluation.
    eval_gaussian: float = None  # Action Gaussian noise for evaluation.
    video_episodes: int = 1  # Number of video episodes for each task.
    video_frame_skip: int = 3  # Frame skip for videos.
    eval_on_cpu: bool = False  # Whether to evaluate on CPU.

    agent: CFGRLConfig = field(default_factory=CFGRLConfig)


def main(cfg: Config):
    # Set up logger.
    exp_name = get_exp_name(cfg.seed)
    setup_wandb(asdict(cfg), project='cfgrl', group=cfg.run_group, name=exp_name)

    cfg.save_dir = os.path.join(cfg.save_dir, wandb.run.project, cfg.run_group, exp_name)
    os.makedirs(cfg.save_dir, exist_ok=True)
    flag_dict = asdict(cfg)
    with open(os.path.join(cfg.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    # Set up environment and dataset.
    config = cfg.agent
    env, train_dataset, val_dataset = make_gc_env_and_datasets(cfg.env_name, frame_stack=config.frame_stack)

    dataset_class = {
        'GCDataset': GCDataset,
        'HGCDataset': HGCDataset,
    }[config['dataset_class']]
    train_dataset = dataset_class(Dataset.create(**train_dataset), config)
    if val_dataset is not None:
        val_dataset = dataset_class(Dataset.create(**val_dataset), config)

    # Initialize agent.
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    example_batch = train_dataset.sample(1)
    pprint(spec(example_batch))

    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        cfg.seed,
        example_batch,
        config,
    )
    pprint(f'Agent: {agent_class.__name__}')

    # Restore agent.
    if cfg.restore_path is not None:
        agent = restore_agent(agent, cfg.restore_path, cfg.restore_epoch)

    # Train agent.
    train_logger = CsvLogger(os.path.join(cfg.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(cfg.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()
    for i in tqdm.tqdm(range(1, cfg.train_steps + 1), smoothing=0.1, dynamic_ncols=True):
        # Update agent.
        # batch = train_dataset.sample(config['batch_size'])
        overfit = train_dataset.sample(
            config['batch_size'], idxs=np.random.choice(range(1000), size=config['batch_size'])
        )
        batch = overfit
        agent, update_info = agent.update(batch)

        # Log metrics.
        if i % cfg.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            if val_dataset is not None:
                val_batch = val_dataset.sample(config['batch_size'])
                _, val_info = agent.total_loss(val_batch, grad_params=None)
                train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
            train_metrics['time/epoch_time'] = (time.time() - last_time) / cfg.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        # Evaluate agent.
        if i % cfg.eval_interval == 0:
            eval_metrics = {}
            if cfg.eval_on_cpu:
                eval_agent = jax.device_put(agent, device=jax.devices('cpu')[0])
            else:
                eval_agent = agent
            renders = []
            overall_metrics = defaultdict(list)
            task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos
            num_tasks = cfg.eval_tasks if cfg.eval_tasks is not None else len(task_infos)
            for task_id in tqdm.trange(1, num_tasks + 1):
                task_name = task_infos[task_id - 1]['task_name']
                eval_info, trajs, cur_renders = evaluate(
                    agent=eval_agent,
                    env=env,
                    task_id=task_id,
                    config=config,
                    num_eval_episodes=cfg.eval_episodes,
                    num_video_episodes=cfg.video_episodes,
                    video_frame_skip=cfg.video_frame_skip,
                    eval_temperature=cfg.eval_temperature,
                    eval_gaussian=cfg.eval_gaussian,
                )
                renders.extend(cur_renders)
                metric_names = ['success']
                eval_metrics.update(
                    {f'evaluation/{task_name}_{k}': v for k, v in eval_info.items() if k in metric_names}
                )
                for k, v in eval_info.items():
                    if k in metric_names:
                        overall_metrics[k].append(v)

            for k, v in overall_metrics.items():
                eval_metrics[f'evaluation/overall_{k}'] = np.mean(v)

            if cfg.video_episodes > 0:
                video = get_wandb_video(renders=renders)
                eval_metrics['evaluation/video'] = video

            wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)

        # Save agent.
        if i % cfg.save_interval == 0:
            save_agent(agent, cfg.save_dir, i)

    train_logger.close()
    eval_logger.close()


if __name__ == '__main__':
    main(tyro.cli(Config))
