import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm, trange
import os

import torch
import matplotlib.pyplot as plt

OmegaConf.register_new_resolver("eval", eval, replace=True)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

@hydra.main(config_path="../../config", config_name='export_videos', version_base=None)
def main(cfg):
    env_name = f'{cfg.task.task_name}_d1'
    env_factory = instantiate(cfg.task.env_factory)
    env = env_factory(env_name=f'{cfg.task.task_name}_d1')

    init_states = []
    for _ in trange(300):
        env.reset()
        init_states.append(env.get_sim_state())
    save_path = os.path.join(SCRIPT_DIR, f'{env_name}.init')
    torch.save(init_states, save_path)


if __name__ == "__main__":
    main()
