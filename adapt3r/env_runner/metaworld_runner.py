import numpy as np

import adapt3r.env.metaworld.utils as mu
import wandb
from tqdm import tqdm, trange
import os
from tqdm import tqdm
import json


class MetaWorldRunner():
    def __init__(self,
                 env_factory,
                 benchmark_name,
                 mode, # train or test
                 rollouts_per_env,
                 fps=10,
                 debug=False,
                 random_task=False,
                 max_episode_length=500,
                 ):
        self.env_factory = env_factory
        self.benchmark_name = benchmark_name
        self.benchmark = mu.get_benchmark(benchmark_name) if not debug else None
        self.mode = mode
        self.rollouts_per_env = rollouts_per_env
        self.fps = fps
        self.random_task = random_task
        self.max_episode_length = max_episode_length
        

    def run(self, 
            policy, 
            n_video=0, 
            do_tqdm=False, 
            save_video_fn=None, 
            save_dir=None,
            fault_tolerant=False):
        # print
        env_names = mu.get_env_names(self.benchmark_name, self.mode)
        
        if save_dir is not None and os.path.exists(os.path.join(save_dir, 'progress.json')):
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
        
        videos = {}
        for env_name in tqdm(env_names, disable=not do_tqdm):
            if env_name in progress['per_env_success_rates']:
                continue

            any_success = False
            env_succs, env_rews, env_video = [], [], []
            rollouts = self.run_policy_in_env(env_name, policy, render=n_video > 0)
            for i, (success, total_reward, episode) in enumerate(rollouts):
                print(success)
                any_success = any_success or success
                progress['successes'].append(success)
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
            progress['per_env_any_success'].append(any_success)

            if len(env_video) > 0:
                video_hwc = np.array(env_video)
                video_chw = video_hwc.transpose((0, 3, 1, 2))
                videos[env_name] = wandb.Video(video_chw, fps=self.fps)
                
            if save_dir is not None:
                progress_file = os.path.join(save_dir, 'progress.json')
                with open(progress_file, 'w') as f:
                    json.dump(progress, f)
            
        output = {}
        output['rollout'] = {
            'overall_success_rate': np.mean(progress['successes']),
            'overall_average_reward': np.mean(progress['rewards']),
            'environments_solved': int(np.sum(progress['per_env_any_success'])),
        }
        output['rollout_success_rate'] = {}
        for env_name in env_names:
            output['rollout_success_rate'][env_name] = progress['per_env_success_rates'][env_name]
        if len(videos) > 0:
            output['rollout_videos'] = {}
        for env_name in videos:

            output['rollout_videos'][env_name] = videos[env_name]
        
        return output


    def run_policy_in_env(self, env_name, policy, render=False):
        env = self.env_factory(env_name=env_name)
        tasks = mu.get_tasks(self.benchmark, self.mode)
        
        env_tasks = [task for task in tasks if task.env_name == env_name]
        count = 0
        while count < self.rollouts_per_env:
            if len(env_tasks) > 0:
                if self.random_task:
                    task_ind = np.random.randint(len(env_tasks))
                    task = env_tasks[task_ind]
                else:
                    task = env_tasks[count % len(env_tasks)]
                env.set_task(task)

            success, total_reward, episode = self.run_episode(env, 
                                                              env_name, 
                                                              policy,
                                                              render)
            count += 1
            yield success, total_reward, episode
        
        env.close()
        del env


    def run_episode(self, env, env_name, policy, render=False):
        obs, _ = env.reset()
        if hasattr(policy, 'get_action'):
            policy.reset()
            policy_object = policy
            policy = lambda obs, task_id: policy_object.get_action(obs, task_id)
        
        done, success, total_reward = False, False, 0

        episode = {key: [value[-1]] for key, value in obs.items()}
        episode['actions'] = []
        episode['terminated'] = []
        episode['truncated'] = []
        episode['reward'] = []
        episode['success'] = []
        if render:
            episode['render'] = [env.render()]

        task_id = mu.get_index(env_name)

        count = 0

        # while not done:
        for _ in trange(self.max_episode_length):
            obs = {k: np.expand_dims(v, 0) for k, v in obs.items()}
            action = policy(obs, task_id).squeeze()
            # action = env.action_space.sample()
            action = np.clip(action, env.action_space.low, env.action_space.high)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            obs = next_obs

            for key, value in obs.items():
                episode[key].append(value[-1])
            episode['actions'].append(action)
            episode['terminated'].append(terminated)
            episode['truncated'].append(truncated)
            episode['reward'].append(reward)
            episode['success'].append(info['success'])
            if int(info["success"]) == 1:
                success = True
            if render:
                episode['render'].append(env.render())

            count += 1
            if done:
                break

        episode = {key: np.array(value) for key, value in episode.items()}
        return success, total_reward, episode
    