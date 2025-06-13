import numpy as np
from tqdm import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from utils.datasets import Dataset, GCDataset, HGCDataset
from utils.env_utils import make_gc_env_and_datasets
from utils.log_utils import get_wandb_video, setup_wandb

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'antmaze-large-navigate-oraclerep-v0', 'Environment (dataset) name.')
flags.DEFINE_string('run_group', 'DatasetInspect', 'Run group for Weights & Biases.')
flags.DEFINE_integer('num_videos', 100, 'Number of episodes to visualize.')
config_flags.DEFINE_config_file('agent', 'agents/cfgrl.py', lock_config=False)


def main(_):
    setup_wandb(project='cfgrl', group=FLAGS.run_group)

    config = FLAGS.agent
    env, train_dataset, _ = make_gc_env_and_datasets(FLAGS.env_name, frame_stack=config['frame_stack'])
    dataset_class = {
        'GCDataset': GCDataset,
        'HGCDataset': HGCDataset,
    }[config['dataset_class']]
    dataset = dataset_class(Dataset.create(**train_dataset), config)

    example = dataset.sample(1)
    spec = {k: v.shape for k, v in example.items()}
    print('Example spec:', spec)

    num_eps = min(len(dataset.initial_locs), FLAGS.num_videos)
    print(dataset.initial_locs[:num_eps])
    print(dataset.terminal_locs[:num_eps])
    renders = []
    idxs = zip(dataset.initial_locs[:num_eps], dataset.terminal_locs[:num_eps])
    for start, end in tqdm(idxs, total=FLAGS.num_videos):
        frames = dataset.dataset['observations'][start:end]
        if len(frames):
            renders.append(frames)
    if renders:
        video = get_wandb_video(renders)
        wandb.log({'dataset/videos': video})

    for k, arr in tqdm(dataset.dataset.items()):
        if arr.ndim == 4 and arr.shape[-1] == 3:
            print(f'Skipping {k} with shape {arr.shape}')
            continue

        if arr.ndim > 1:
            for i in tqdm(range(arr.shape[-1]), leave=False):
                wandb.log({f'dataset/{k}_{i}': wandb.Histogram(arr[...,i])})
        else:
            wandb.log({f'dataset/{k}': wandb.Histogram(arr.flatten())})


if __name__ == '__main__':
    app.run(main)
