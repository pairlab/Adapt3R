import os

import h5py
import hydra
import numpy as np
from natsort import natsorted
# from IPython.core import ultratb
from libero.libero import benchmark
from tqdm import tqdm, trange
import pytorch3d.transforms as pt
import torch

import adapt3r.env.libero.utils as lu


from hydra.utils import instantiate

compress_keys = {
    'robot0_eye_in_hand_pointcloud_full', 
    'agentview_pointcloud_full'
}


def dump_demo(demo, file_path, demo_i, attrs=None):
    with h5py.File(file_path, 'a') as f:
        group_data = f['data']
        group = group_data.create_group(demo_i)

        if attrs is not None:
            for key in attrs:
                group.attrs[key] = attrs[key]

        demo_length = len(demo['actions'])
        group_data.attrs['total'] = group_data.attrs['total'] + demo_length
        group.attrs['num_samples'] = demo_length
        non_obs_keys = ('actions', 'abs_actions', 'terminated', 'truncated', 'reward', 'success')
        group.create_dataset('states', data=())
        for key in demo:
            if key in non_obs_keys:
                group.create_dataset(key, data=demo[key])
            else:
                if key in compress_keys:
                    data = np.array(demo[key]).astype(np.float16)
                else:
                    data = demo[key]
                group.create_dataset(f'obs/{key}', data=data)


def process_demo(old_demo, env):
    """
    Collect gt depths in simulation by replaying demos
    """

    actions = old_demo['actions']
    states = old_demo['states']
    init_state = states[0]

    obs, info = env.reset(init_state)

    new_demo = {
        'actions': [],
        'abs_actions': [],
        'reward': [],
        'terminated': [],
        'truncated': [],
        'success': []
    }
    for key in obs:
        new_demo[key] = []

    success = False
    ims = []
    for t in trange(len(actions), disable=True):
        for key in obs:
            new_demo[key].append(obs[key])

        obs, reward, terminated, truncated, info = env.step(actions[t])

        # read pos and ori from robots
        controller = env.env.robots[0].controller
        goal_pos = controller.goal_pos
        goal_ori = pt.matrix_to_rotation_6d(torch.tensor(controller.goal_ori)).numpy()

        abs_action = np.concatenate((goal_pos, goal_ori, actions[t][..., -1:]))
        
        success = success or info['success']

        new_demo['actions'].append(actions[t])
        new_demo['abs_actions'].append(abs_action)
        new_demo['reward'].append(reward)
        new_demo['terminated'].append(terminated)
        new_demo['truncated'].append(truncated)
        new_demo['success'].append(info['success'])
    return new_demo


def process_task_dataset(task_no, source_h5_path, dest_h5_path, benchmark, env_factory):
    """
    Generate preprocessed data for a task.
    """
    demos = h5py.File(source_h5_path, 'r')['data']
    demo_keys = natsorted(list(demos.keys()))

    with h5py.File(dest_h5_path, 'a') as f:
        already_processed = 0
        if 'data' not in f:
            group_data = f.create_group('data')

            for attr in demos.attrs:
                group_data.attrs[attr] = demos.attrs[attr]
        else:
            already_processed = len(set(f['data']))
        
    env = env_factory(task_id=task_no, benchmark=benchmark)

    for idx in trange(already_processed, len(demo_keys), disable=0):

        demo_k = demo_keys[idx]
        
        demo = process_demo(demos[demo_k], env)

        dump_demo(demo, dest_h5_path, demo_k)


@hydra.main(config_path="../config", 
            config_name='collect_data', 
            version_base=None)
def main(cfg):

    # define source and destination directories
    source_dir = os.path.join(cfg.data_prefix, cfg.task.suite_name, cfg.task.benchmark_name + '_unprocessed')
    save_dir = os.path.join(cfg.data_prefix, cfg.task.suite_name, cfg.task.benchmark_name + '_processed')
    os.makedirs(save_dir, exist_ok=True)

    print(source_dir)
    print(save_dir)

    # process each task
    benchmark_dict = benchmark.get_benchmark_dict()
    benchmark_instance = benchmark_dict[cfg.task.benchmark_name]()

    num_tasks = benchmark_instance.get_num_tasks()
    task_files = [os.path.join(source_dir, benchmark_instance.get_task_demonstration(i).split('/')[1]) for i in range(num_tasks)]
    task_names = benchmark_instance.get_task_names()

    env_factory = instantiate(cfg.task.env_factory)

    for task_no, task_name in enumerate(task_names):
        task_name = task_names[task_no]
        setting, number, _ = lu.deconstruct_task_name(task_name)
        
        if 'setting_filter' in cfg and setting != cfg.setting_filter:
            continue
        
        if 'late_start' in cfg and task_no < cfg.late_start:
            continue
            
        print(task_name)
        source_h5_path = task_files[task_no]
        dest_h5_path = os.path.join(save_dir, f"{task_names[task_no]}_demo.hdf5")
        process_task_dataset(task_no, source_h5_path, dest_h5_path, benchmark_instance, env_factory)


if __name__ == "__main__":
    main()