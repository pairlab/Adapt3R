from .wrappers import MimicGenFrameStack
import wandb
import numpy as np
from tqdm import tqdm, trange
import os
import json
import torch
import h5py

# _eval_envs = {
#     "square": ["square_d1"],
#     "stack": ["stack_d1"],
#     "threading": ["threading_d1"],
#     "three_piece_assembly": [ "three_piece_assembly_d1"],
#     "coffee": ["coffee_d1"],
#     "coffee_preparation": ["coffee_preparation_d1"],
#     "mug_cleanup": ["mug_cleanup_d1"],
#     "kitchen": ["kitchen_d1"],
#     "hammer_cleanup": ["hammer_cleanup_d1"],
#     "stack_three": ["stack_three_d1"],
# }


class MimicGenRunner():
    def __init__(self,
                 env_factory,
                 benchmark_name,
                 mode,  # train or test
                 rollouts_per_env,
                 fps=10,
                 frame_stack=1,
                 max_episode_length=500,
                 render_interval=5,
                 debug=False,
                 deterministic=True,
                 save_freq=10,
                 ):
        
        self.frame_stack = frame_stack
        self.render_interval = render_interval

        self.env_factory = env_factory
        self.benchmark_name = benchmark_name
        self.mode = mode
        self.rollouts_per_env = rollouts_per_env
        self.fps = fps
        self.max_episode_length = max_episode_length
        self.deterministic = deterministic
        self.robot = env_factory.keywords['robot']
        self.counter = 0
        # self.eval_envs_names = _eval_envs[benchmark_name]
        self.eval_envs_names = [benchmark_name]
        self.save_freq = save_freq
        print()
        print("Mode: ", mode)
        print("Benchmark name: ", benchmark_name)
        print("Eval envs: ", self.eval_envs_names)
        print()

    def get_init_states(self, env_name):
        if self.deterministic:
            fpath = os.path.dirname(os.path.abspath(__file__))
            if self.robot == 'Panda':
                fpath = os.path.join(fpath, 'init_states', f'{env_name}.init')
            else:
                # return [None] * self.rollouts_per_env
                fpath = os.path.join(fpath, 'init_states', f'{env_name}_{self.robot}.init')
            try:
                return torch.load(fpath, weights_only=False)
            except:
                print(f"No init states found for {env_name}")
                return [None] * self.rollouts_per_env
        else:
            return [None] * self.rollouts_per_env

    def run(
            self, 
            policy, 
            n_video=0, 
            save_dir=None, 
            save_progress=False,
            save_video_fn=None, 
            save_hdf5=False,
            do_tqdm=False,
            env_names=None,
            fault_tolerant=False,
            ):
        if save_progress and os.path.exists(os.path.join(save_dir, 'progress.json')):
            progress_file = os.path.join(save_dir, 'progress.json')
            with open(progress_file, 'r') as f:
                progress = json.load(f)
        else:
            progress = {
                'successes': [],
                'rewards': [],
                'per_env_success': {},
                'per_env_rewards': {},
                'per_env_solved': [],
                'per_env_metrics': {}  # Add field for storing metrics
            }

        self.counter = len(progress['successes'])

        if save_hdf5:
            hdf5_dir = os.path.join(save_dir, 'rollouts.hdf5')
            f_out = h5py.File(hdf5_dir, 'w')
            data_grp = f_out.create_group("data")

        videos = {}
        for j, env_name in enumerate(self.eval_envs_names):
            if env_name in progress['per_env_success']:
                continue

            print(f"\nRunning rollouts for environment: {env_name}")
            solved = False
            env_succeses, env_rewards, env_video = [], [], []
            env_metrics = []
            env_fn = lambda: MimicGenFrameStack(self.env_factory(env_name=env_name), self.frame_stack)
            init_states = self.get_init_states(env_name=env_name)
            start = len(progress['successes']) - j * self.rollouts_per_env
            rollouts = self.run_policy_in_env(
                policy, 
                env_fn=env_fn, 
                render=n_video > 0, 
                init_states=init_states,
                start=start,
                do_tqdm=do_tqdm,
                fault_tolerant=fault_tolerant
            )
            for i, (success, total_reward, episode) in enumerate(rollouts):
                success = bool(success)
                total_reward = float(total_reward)
                solved = solved or success
                progress['successes'].append(success)
                progress['rewards'].append(total_reward)

                env_succeses.append(success)
                env_rewards.append(total_reward)
                env_metrics.append(episode['metrics'])
            
                if i % self.save_freq == 0:
                    if save_progress:
                        progress_file = os.path.join(save_dir, 'progress.json')
                        with open(progress_file, 'w') as f:
                            json.dump(progress, f)

                if i < n_video:
                    if save_video_fn is not None:
                        video_hwc = np.array(episode['render'])
                        video_chw = video_hwc.transpose((0, 3, 1, 2))
                        save_video_fn(video_chw, env_name, i)
                    else:
                        env_video.extend(episode['render'])

                if save_hdf5:
                    ep_data_grp = data_grp.create_group(f'demo_{i}')
                    episode.pop('render')
                    ep_data_grp.create_dataset('actions', data=episode.pop('actions'))
                    for k, v in episode.items():
                        if type(v) == np.ndarray:
                            ep_data_grp.create_dataset(f'obs/{k}', data=v)

                    
            progress['per_env_success'][env_name] = float(np.mean(env_succeses))
            progress['per_env_rewards'][env_name] = float(np.mean(env_rewards))
            progress['per_env_solved'].append(bool(solved))
            
            avg_metrics = {}
            metric_keys = env_metrics[0].keys()
            for key in metric_keys:
                avg_metrics[key] = float(np.mean([float(m[key]) for m in env_metrics]))
            progress['per_env_metrics'][env_name] = avg_metrics

            if len(env_video) > 0:
                video_hwc = np.array(env_video)
                video_chw = video_hwc.transpose((0, 3, 1, 2))
                videos[env_name] = wandb.Video(video_chw, fps=self.fps)

            if save_progress:
                progress_file = os.path.join(save_dir, 'progress.json')
                with open(progress_file, 'w') as f:
                    json.dump(progress, f)

        if save_hdf5:
            f_out.close()

        output = {}
        output['rollout'] = {
            'overall_success_rate': float(np.mean(progress['successes'])),
            'overall_average_reward': float(np.mean(progress['rewards'])),
            'environments_solved': int(np.sum(progress['per_env_solved'])),
        }

        # Add metrics to the rollout dictionary
        if len(progress['per_env_metrics']) > 0:
            # Get the first (and only) environment metrics
            env_name = list(progress['per_env_metrics'].keys())[0]
            metrics = progress['per_env_metrics'][env_name]
            for metric_key, metric_value in metrics.items():
                output['rollout'][metric_key] = metric_value

        output['rollout_success_rate'] = {}
        for env_name in self.eval_envs_names:
            output['rollout_success_rate'][env_name] = float(progress['per_env_success'][env_name])
        
        if len(videos) > 0:
            output['rollout_videos'] = {}

        for env_name in videos:
            output['rollout_videos'][env_name] = videos[env_name]
        
        return output

    def run_policy_in_env(self, policy, env_fn, render=False, init_states=None, start=0, do_tqdm=False, fault_tolerant=False):
        for i in range(start, self.rollouts_per_env):
            env = env_fn()
            try:
                success, total_reward, episode = self.run_episode(env, policy, render, init_states[i], do_tqdm=do_tqdm)
            except Exception as e:
                if fault_tolerant:
                    print('WARNING: rollout failed. Recording a failure')
                    print(e)
                    success = False
                    total_reward = 0
                    episode = {}
                else:
                    raise e
            yield success, total_reward, episode

    def run_episode(self, env, policy, render=False, init_state=None, do_tqdm=False):
        obs, info = env.reset(init_state=init_state)

        if hasattr(policy, 'get_action'):
            policy.reset()
            policy_object = policy
            policy = lambda obs: policy_object.get_action(obs, 0)
        
        success = False
        total_reward = 0

        episode = {key: [value[-1]] for key, value in obs.items()}
        episode['actions'] = []
        episode['render'] = []

        steps = 0
        done = False
        print("Running rollout: ", self.counter)
        self.counter += 1
        
        tracked_metrics = None

        for steps in trange(self.max_episode_length, desc="Episode Progress", disable=not do_tqdm):
            if done:
                break
            for key in obs.keys():
                obs[key] = obs[key][np.newaxis, :]

            action = policy(obs).squeeze()
            # action = np.clip(action, env.action_space.low, env.action_space.high)
            next_obs, reward, terminated, truncated, info = env.step(action)

            try:
                current_metrics = env.env.env._get_partial_task_metrics()
            except:
                current_metrics = {}
            
            if tracked_metrics is None:
                tracked_metrics = {k: False for k in current_metrics.keys()}
            
            for key in tracked_metrics:
                tracked_metrics[key] = tracked_metrics[key] or current_metrics[key]
            
            done = terminated or truncated
            total_reward += reward
            obs = next_obs

            for key, value in obs.items():
                episode[key].append(value[-1])

            episode['actions'].append(action)

            # env.env.env.render()

            if render:
                episode['render'].append(env.render())
            
            success = success or info['success']
            
            if success:
                break

            steps += 1

        episode = {key: np.array(value) for key, value in episode.items()}
        
        episode['metrics'] = tracked_metrics if tracked_metrics is not None else {}
        
        return success, total_reward, episode