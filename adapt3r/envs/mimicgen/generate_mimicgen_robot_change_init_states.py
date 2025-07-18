import os

import h5py
import hydra
import numpy as np
np.set_printoptions(suppress=True)
from natsort import natsorted
# from IPython.core import ultratb
from tqdm import tqdm, trange
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import cv2

from adapt3r.algos.utils.rotation_transformer import RotationTransformer

from hydra.utils import instantiate, call

# Do not remove this import. It is necessary to register env classes
import torch
import einops
from multiprocessing import Pool
from copy import deepcopy


OmegaConf.register_new_resolver("eval", eval, replace=True)
    


@hydra.main(config_path="../../config", 
            config_name='train_debug', 
            version_base=None)
def main(cfg):
    OmegaConf.resolve(cfg)

    if 'robot' in cfg:
        robot = cfg.robot
    else:
        assert False, 'add "+robot=x" to your command. Dont use task.robot or that will break the script'


    env_factory = instantiate(cfg.task.env_factory)
    env_factory_changed = instantiate(cfg.task.env_factory,
                                      robot=robot,
                                      abs_action=True)


    fpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'{cfg.task.task_name}.init')
    fpath_new = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'{cfg.task.task_name}_{robot}.init')


    # If the init states file is already there, don't waste time regenerating it
    if os.path.exists(fpath_new):
        exit(0)
    
    env = env_factory()
    env_changed = env_factory_changed()
    old_init_states = torch.load(fpath, weights_only=False)
    new_init_states = []

    old_init_state = old_init_states[0]
    old_obs, _ = env.reset(old_init_state)
    
    controller = env.env.robots[0].controller
    goal_pos = controller.goal_pos
    goal_ori = Rotation.from_matrix(
        controller.goal_ori).as_rotvec()
    abs_action = np.concatenate((goal_pos, goal_ori, [-1]))
    for i in trange(100):

        start_obs, _ = env_changed.reset()
        old_state = env_changed.get_sim_state()
        ims = []
        for _ in range(50):
            obs, _, _, _, _ = env_changed.step(abs_action)
            ims.append(obs['agentview_image'][..., ::-1])

        new_init_state = env_changed.get_sim_state()

        new_init_states.append(new_init_state)

    torch.save(new_init_states, fpath_new)


if __name__ == "__main__":
    main()

