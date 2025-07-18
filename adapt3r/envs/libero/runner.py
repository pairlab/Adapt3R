# import gym
# import gym.wrappers
# import gym.wrappers.frame_stack
import numpy as np

import adapt3r.envs.libero.utils as lu
import adapt3r.envs.libero.wrappers as lw
import adapt3r.utils.obs_utils as ObsUtils
from adapt3r.envs.utils import FrameStackObservationFixed
import wandb
from tqdm import tqdm, trange
import multiprocessing
import os
import json
import matplotlib.pyplot as plt

class LiberoRunner():
    def __init__(self,
                 env_factory,
                 benchmark,
                 mode, # all or few
                #  obs_modality,
                 rollouts_per_env,
                 num_parallel_envs,
                 max_episode_length,
                 frame_stack=1,
                 fps=10,
                 debug=False,
                 task_embedding_format='clip',
                 test_inference_time=False,
                 ):
        self.env_factory = env_factory
        self.benchmark = benchmark
        descriptions = [self.benchmark.get_task(i).language for i in range(self.benchmark.n_tasks)]
        task_embs = lu.get_task_embs(task_embedding_format, descriptions)
        self.benchmark.set_task_embs(task_embs)
        self.env_names = self.benchmark.get_task_names()
        self.test_inference_time = test_inference_time
        # ObsUtils.initialize_obs_utils_with_obs_specs({"obs": obs_modality})

        self.mode = mode
        self.rollouts_per_env = rollouts_per_env
        self.num_parallel_envs = num_parallel_envs
        self.frame_stack = frame_stack
        if num_parallel_envs>1:
            if multiprocessing.get_start_method(allow_none=True) != "spawn":  
                multiprocessing.set_start_method("spawn", force=True)
        self.max_episode_length = max_episode_length
        self.fps = fps
        
    def run(self, 
            policy, 
            n_video=0, 
            do_tqdm=False, 
            save_video_fn=None, 
            save_dir=None, 
            save_progress=False,
            env_names=None,
            fault_tolerant=False,
            save_hdf5=False,
        ):
        if env_names is None:
            env_names = self.env_names
        if save_progress and os.path.exists(os.path.join(save_dir, 'progress.json')):
            progress_file = os.path.join(save_dir, 'progress.json')
            with open(progress_file, 'r') as f:
                progress = json.load(f)
        else:
            progress = {
                'successes': [],
                'per_env_any_success': [],
                'rewards': [],
                'per_env_success_rates': {},
                'per_env_rewards': {},
            }
        # successes, per_env_any_success, rewards = [], [], []
        # per_env_success_rates, per_env_rewards = {}, {}
        videos = {}
        for env_name in tqdm(env_names, disable=not do_tqdm):
            if env_name in progress['per_env_success_rates']:
                continue

            any_success = False
            env_succs, env_rews, env_video = [], [], []
            rollouts = self.run_policy_in_env(env_name, 
                                              policy, 
                                              render=n_video > 0, 
                                              fault_tolerant=fault_tolerant)
            for i, (success, total_reward, episode) in enumerate(rollouts):
                any_success = any_success or success
                progress['successes'].append(bool(success))
                env_succs.append(success)
                env_rews.append(total_reward)
                progress['rewards'].append(total_reward)

                if i < n_video:
                    if save_video_fn is not None:
                        video_hwc = np.array(episode['render'])
                        video_chw = video_hwc.transpose((0, 3, 1, 2))
                        save_video_fn(video_chw, env_name, i)
                    else:
                        env_video.extend(episode['render'])
                    
            progress['per_env_success_rates'][env_name] = np.mean(env_succs)
            progress['per_env_rewards'][env_name] = np.mean(env_rews)
            progress['per_env_any_success'].append(bool(any_success))

            if len(env_video) > 0:
                video_hwc = np.array(env_video)
                video_chw = video_hwc.transpose((0, 3, 1, 2))
                videos[env_name] = wandb.Video(video_chw, fps=self.fps)
                
            if save_progress:
                progress_file = os.path.join(save_dir, 'progress.json')
                with open(progress_file, 'w') as f:
                    json.dump(progress, f)
            # break

        output = {}
        output['rollout'] = {
            'overall_success_rate': np.mean(progress['successes']),
            'overall_average_reward': np.mean(progress['rewards']),
            'environments_solved': int(np.sum(progress['per_env_any_success'])),
        }
        output['rollout_success_rate'] = {}
        for env_name in env_names:
            output['rollout_success_rate'][env_name] = progress['per_env_success_rates'][env_name]
            # This metric isn't that useful
            # output[f'rollout_detail/average_reward_{env_name}'] = per_env_rewards[env_name]
        if len(videos) > 0:
            output['rollout_videos'] = {}
        for env_name in videos:

            output['rollout_videos'][env_name] = videos[env_name]
        
        return output

    def run_policy_in_env(self, env_name, policy, render=False, fault_tolerant=False):
        env_id = self.env_names.index(env_name)
        env_num = min(self.num_parallel_envs, self.rollouts_per_env)
        # env = self.env_factory(env_id, self.benchmark)
        env_fn = lambda: lw.LiberoFrameStack(self.env_factory(task_id=env_id, benchmark=self.benchmark), self.frame_stack)
        env = lw.LiberoVectorWrapper(env_fn, self.num_parallel_envs)

        all_init_states = self.benchmark.get_task_init_states(env_id)
        count = 0
        eval_loop_num = (self.rollouts_per_env+self.num_parallel_envs - 1) // self.num_parallel_envs

        while count < eval_loop_num:
            indices = np.arange(count * env_num, (count + 1) * env_num) % all_init_states.shape[0]
            init_states_ = all_init_states[indices]
            # breakpoint()
            try:
                success, total_reward, episode = self.run_episode(env, 
                                                                    env_name, 
                                                                    policy,
                                                                    init_states_,
                                                                    env_num,
                                                                    render)
            except Exception as e:
                if fault_tolerant:
                    print('WARNING: rollout failed. Recording a failure')
                    print(e)
                    success = [False] * self.num_parallel_envs
                    total_reward = [0] * self.num_parallel_envs
                    episode = {}
                else:
                    raise e

            count += 1
            for k in range(env_num):
                episode_k = {key: value[:,k] for key, value in episode.items()}
                yield success[k], total_reward[k], episode_k
    
    def run_episode(self, env, env_name, policy, init_states_, env_num, render=False):
        obs, info = env.reset(init_states=init_states_)

        if hasattr(policy, 'get_action'):
            policy.reset()
            policy_object = policy
            policy = lambda obs, task_id, **kwargs: policy_object.get_action(obs, task_id, **kwargs)
        
        success, total_reward = [False]*env_num, [0]*env_num

        episode = {key: [value[:,-1]] for key, value in obs.items()}
        episode['actions'] = []
        if render:
            episode['render'] = [env.render()]

        task_id = self.env_names.index(env_name)
        task_emb = self.benchmark.get_task_emb(task_id)
        task_emb = {key: value.repeat(env_num, 1) for key, value in task_emb.items()}
        if self.test_inference_time:
            import time
            import copy
            
            t_start = time.time()
            for i in trange(1000):
                obs_clone = copy.deepcopy(obs)
                action = policy(obs_clone, task_id, **task_emb)
            t_end = time.time()
            s_per_iter = (t_end - t_start) / 1000
            hz = 1 / s_per_iter
            print('Seconds per iter:', s_per_iter)
            print('Hz:', hz)
            exit()
        for _ in trange(self.max_episode_length, disable=False):
            action = policy(obs, task_id, **task_emb)
            # TODO: fix bounds in the libero env and then uncomment
            # action = np.clip(action, env.action_space.low, env.action_space.high)
            next_obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            obs = next_obs
            for key, value in obs.items():
                episode[key].append(value[:,-1])
            episode['actions'].append(action)
            if render:
                episode['render'].append(env.render())
        
            for k in range(env_num):
                success[k] = success[k] or info[k]['success']
            
            if all(success):
                break

        episode = {key: np.array(value) for key, value in episode.items()}
        return success, total_reward, episode