import os
import torch
import numpy as np
from tqdm import tqdm, trange

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from libero.libero.utils.bddl_generation_utils import *

OmegaConf.register_new_resolver("eval", eval, replace=True)
    
@hydra.main(config_path="../../config", 
            config_name='train_debug', 
            version_base=None)
def main(cfg):
    OmegaConf.resolve(cfg)

    env_factory = instantiate(cfg.task.env_factory)

    benchmark_instance = instantiate(cfg.task.benchmark_instance)
    task_names = benchmark_instance.get_task_names()
    for task_no, task_name in enumerate(tqdm(task_names)):
        if 'late_start' in cfg and task_no < cfg.late_start:
            continue

        # If the init states file is already there, don't waste time regenerating it
        try:
            benchmark_instance.get_task_init_states(task_no)
            continue
        except FileNotFoundError:
            pass
        
        bddl_file = benchmark_instance.get_task_bddl_file_path(task_no)
        env = env_factory(bddl_file=bddl_file)
        init_states = []

        for _ in trange(50):
            env.reset()
            init_state = env.env.env.sim.get_state().flatten()
            init_states.append(init_state)
        
        init_states = np.array(init_states)
        path = benchmark_instance.get_task_init_states_path(task_no)
        init_states_folder = os.path.dirname(path)
        os.makedirs(init_states_folder, exist_ok=True)
        torch.save(init_states, path)


if __name__ == "__main__":
    main()


