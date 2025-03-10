import os

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

import torch
import numpy as np
from tqdm import tqdm, trange
from scipy.spatial.transform import Rotation

from libero.libero.utils.bddl_generation_utils import *

OmegaConf.register_new_resolver("eval", eval, replace=True)
    

@hydra.main(config_path="../../config", 
            config_name='train_debug', 
            version_base=None)
def main(cfg):
    OmegaConf.resolve(cfg)

    if 'robot' in cfg:
        robot = cfg.robot
    else:
        robot = 'Sawyer'

    env_factory = instantiate(cfg.task.env_factory)
    env_factory_changed = instantiate(cfg.task.env_factory,
                                      robot=robot,
                                      abs_action=True)

    # process each task
    benchmark_instance = instantiate(cfg.task.benchmark_instance)
    benchmark_instance_changed = instantiate(cfg.task.benchmark_instance, robot=robot)
    task_names = benchmark_instance.get_task_names()

    for task_no, task_name in enumerate(tqdm(task_names)):
        
        if 'late_start' in cfg and task_no < cfg.late_start:
            continue

        # If the init states file is already there, don't waste time regenerating it
        try:
            benchmark_instance_changed.get_task_init_states(task_no)
            continue
        except FileNotFoundError:
            pass
        
        bddl_file = benchmark_instance.get_task_bddl_file_path(task_no)
        env = env_factory(bddl_file=bddl_file, skip_render=False)
        env_changed = env_factory_changed(bddl_file=bddl_file, skip_render=False)
        old_init_states = benchmark_instance.get_task_init_states(task_no)
        new_init_states = []

        for i in trange(50):
            old_init_state = old_init_states[i]
            old_obs, _ = env.reset(old_init_state)
            
            controller = env.env.robots[0].controller
            goal_pos = controller.goal_pos
            goal_ori = Rotation.from_matrix(
                controller.goal_ori).as_rotvec()
            abs_action = np.concatenate((goal_pos, goal_ori, [-1]))

            # The purpose behind this block of code is to move the new robot such that its
            # starting pose is the same as the starting pose in the original 
            obs, _ = env_changed.reset()
            old_state = env_changed.env.env.sim.get_state().flatten()
            for _ in range(15):
                obs, _, _, _, _ = env_changed.step(abs_action)
            new_init_state = env_changed.env.env.sim.get_state().flatten()
            
            n_joint_pos = 6 if robot == 'UR5e' else 7
            n_scene = (len(new_init_state) - 1) // 2 - n_joint_pos
            # The order of these states is:
            # [
            #   0: timestep,
            #   1-n_joint_pos: robot joint poses
            #   a bunch of info about all the objects in the scene
            #   and then repeat everything except the timestep for velocities
            # ]
            init_state = np.concatenate([
                old_init_state[:1], # copy the timestep from the original init state
                new_init_state[1:1+n_joint_pos], # new joint angles
                old_init_state[8:8+n_scene], # same object positions as the original init state
                np.zeros(n_joint_pos), # zero joint velocity
                old_init_state[15+n_scene:]], axis=0) # same object velocities as original 
            new_init_states.append(init_state)

        new_init_states = np.array(new_init_states)
        path = benchmark_instance_changed.get_task_init_states_path(task_no)
        init_states_folder = os.path.dirname(path)
        os.makedirs(init_states_folder, exist_ok=True)
        torch.save(new_init_states, path)


if __name__ == "__main__":
    main()

