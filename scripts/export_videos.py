import os
import time
import hydra
import wandb
import logging
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import adapt3r.utils.utils as utils
from pyinstrument import Profiler
from moviepy import ImageSequenceClip
import json
from PIL import Image
import numpy as np

OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(config_path="../config", config_name='export_videos', version_base=None)
def main(cfg):
    device = cfg.device
    seed = cfg.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_cfg = cfg.training
    OmegaConf.resolve(cfg)
    
    # create model
    save_dir, experiment_name = utils.get_experiment_dir(cfg, evaluate=False, allow_overlap=True)
    os.makedirs(save_dir)

    # Add experiment-specific logging
    logger = utils.setup_logger('export_videos.log', save_dir)

    logger.info(f"Starting video export for experiment: {experiment_name}")
    logger.info(f"Export directory: {save_dir}")

    try:
        state_dict = utils.load_checkpoint(cfg.checkpoint_path, logger=logger)
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {str(e)}")
        raise e
    
    logger.info("Autoloading based on saved parameters")
    policy_cfg = state_dict['config']['algo']['policy']
    if 'overrides' in cfg:
        overrides = OmegaConf.to_container(cfg.overrides, resolve=True)
        utils.recursive_update(policy_cfg, overrides)
    abs_action = policy_cfg['abs_action']
    model = instantiate(policy_cfg)
    model.to(device)
    model.eval()

    model.load_state_dict(state_dict['model'])
    model.normalizer.fit(state_dict['norm_stats'])
    logger.info("Loaded model state and normalization statistics")

    env_runner = instantiate(
        cfg.task.env_runner,
        env_factory={'abs_action': abs_action}
    )

    if 'video_mode' in cfg and cfg.video_mode == 'images':
        def save_video_fn(video_chw, env_name, idx):
            save_path = os.path.join(save_dir, f'{env_name}_{idx}')
            os.makedirs(save_path)
            logger.debug(f"Saving video frames to: {save_path}")

            video_hwc = video_chw.transpose(0, 2, 3, 1)
            for i, frame in enumerate(video_hwc):
                image = Image.fromarray(frame)
                image.save(os.path.join(save_path, f'frame_{i}.png'))
    else:
        def save_video_fn(video_chw, env_name, idx):
            save_path = os.path.join(save_dir, f'{env_name}_{idx}.mp4')
            logger.debug(f"Saving video to: {save_path}")
            clip = ImageSequenceClip(list(video_chw.transpose(0, 2, 3, 1)), fps=24)
            clip.write_videofile(save_path, fps=24, logger=None)

    if 'env_names' in cfg:
        env_names = cfg.env_names
        logger.info(f"Running evaluation on specific environments: {env_names}")
    else:
        env_names = None
        logger.info("Running evaluation on all environments")

    if train_cfg.do_profile:
        profiler = Profiler()
        profiler.start()

    logger.info("Starting rollout evaluation")
    rollout_results = env_runner.run(model, 
                                     n_video=cfg.rollout.n_video, 
                                     do_tqdm=train_cfg.use_tqdm,
                                     env_names=env_names,
                                     save_dir=save_dir,
                                     save_video_fn=save_video_fn,
                                     save_hdf5=cfg.save_hdf5,
                                     fault_tolerant=False)
    
    if train_cfg.do_profile:
        profiler.stop()
        profiler.print()

    success_rate = rollout_results['rollout']['overall_success_rate']
    envs_solved = rollout_results['rollout']['environments_solved']
    logger.info(
        f"Evaluation results - success rate: {success_rate:1.3f} | environments solved: {envs_solved}"
    )

    results_file = os.path.join(save_dir, 'data.json')
    with open(results_file, 'w') as f:
        json.dump(rollout_results, f)
    logger.info(f"Saved evaluation results to: {results_file}")

if __name__ == "__main__":
    main()
