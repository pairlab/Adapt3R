import os
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

import torch
import adapt3r.utils.utils as utils
from pyinstrument import Profiler
from moviepy import ImageSequenceClip
import json
from PIL import Image

OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(config_path="../config", config_name='export_videos', version_base=None)
def main(cfg):
    device = cfg.device
    seed = cfg.seed
    torch.manual_seed(seed)
    train_cfg = cfg.training
    OmegaConf.resolve(cfg)
    
    # create model

    save_dir, _ = utils.get_experiment_dir(cfg, evaluate=False)
    os.makedirs(save_dir)

    try:
        checkpoint_path = utils.get_latest_checkpoint(cfg.checkpoint_path)
        state_dict = utils.load_state(checkpoint_path)
    except Exception as e:
        if cfg.allow_no_ckpt:
            state_dict = None
            print('no checkpoint found, running with untrained model')
        else:
            raise e
        
    if state_dict is not None and 'config' in state_dict:
        print('autoloading based on saved parameters')
        policy_cfg = state_dict['config']['algo']['policy']
        if 'overrides' in cfg:
            utils.recursive_update(policy_cfg, cfg.overrides)
        model = instantiate(policy_cfg, 
                            shape_meta=cfg.task.shape_meta)
    else:
        model = instantiate(cfg.algo.policy,
                            shape_meta=cfg.task.shape_meta)
    model.to(device)
    model.eval()

    if state_dict is not None:
        model.load_state_dict(state_dict['model'])
        model.normalizer.fit(state_dict['norm_stats'])
    else:
        model.normalizer.fit(None)

    env_runner = instantiate(cfg.task.env_runner)
    
    print(save_dir)

    if 'video_mode' in cfg and cfg.video_mode == 'images':
        def save_video_fn(video_chw, env_name, idx):
            save_path = os.path.join(save_dir, f'{env_name}_{idx}')
            os.makedirs(save_path)

            video_hwc = video_chw.transpose(0, 2, 3, 1)
            for i, frame in enumerate(video_hwc):
                image = Image.fromarray(frame)
                image.save(os.path.join(save_path, f'frame_{i}.png'))
    else:
        def save_video_fn(video_chw, env_name, idx):
            save_path = os.path.join(save_dir, f'{env_name}_{idx}.mp4')
            clip = ImageSequenceClip(list(video_chw.transpose(0, 2, 3, 1)), fps=24)
            clip.write_videofile(save_path, fps=24, logger=None)

    if 'env_names' in cfg:
        env_names = cfg.env_names
    else:
        env_names = None

    if train_cfg.do_profile:
        profiler = Profiler()
        profiler.start()
    rollout_results = env_runner.run(model, 
                                     n_video=cfg.rollout.n_video, 
                                     do_tqdm=train_cfg.use_tqdm, 
                                     save_video_fn=save_video_fn,
                                     env_names=env_names)
    if train_cfg.do_profile:
        profiler.stop()
        profiler.print()
    print(
        f"[info]     success rate: {rollout_results['rollout']['overall_success_rate']:1.3f} \
            | environments solved: {rollout_results['rollout']['environments_solved']}")

    with open(os.path.join(save_dir, 'data.json'), 'w') as f:
        json.dump(rollout_results, f)

    


if __name__ == "__main__":
    main()
