from abc import ABC, abstractmethod
from collections import deque

import einops
import numpy as np
import torch
import torch.nn as nn

import adapt3r.utils.obs_utils as ObsUtils

# from adapt3r.modules.v1 import *
import adapt3r.utils.tensor_utils as TensorUtils
from adapt3r.algos.utils.encoder import BaseEncoder
from adapt3r.algos.utils.normalizer import Normalizer
from adapt3r.algos.utils.rotation_transformer import RotationTransformer
from adapt3r.utils.utils import map_tensor_to_device


class Policy(nn.Module, ABC):
    """
    Super class with some basic functionality and functions we expect
    from all policy classes in our training loop
    """

    def __init__(
        self,
        encoder: BaseEncoder,
        aug_factory,
        optimizer_factory,
        scheduler_factory,
        shape_meta,
        abs_action,
        device,
        normalizer: Normalizer = None,
    ):
        super().__init__()

        self.encoder = encoder
        self.use_augmentation = aug_factory is not None
        self.shape_meta = shape_meta
        self.optimizer_factory = optimizer_factory
        self.scheduler_factory = scheduler_factory
        if normalizer is None:
            normalizer = Normalizer(mode="identity")
        self.normalizer = normalizer
        self.abs_action = abs_action
        self.action_key = 'abs_actions' if abs_action else 'actions'
        self.device = device

        # Use 6D actions if we are predicting abs actions, else axis angle
        self.network_action_dim = 10 if abs_action else 7
        if abs_action:
            self.rotation_transformer = RotationTransformer(from_rep="axis_angle", to_rep="rotation_6d")

        if self.use_augmentation:
            self.aug = aug_factory(shape_meta=shape_meta)

        self.device = device

    @abstractmethod
    def compute_loss(self, data):
        raise NotImplementedError("Implement in subclass")

    def get_optimizers(self):
        decay, no_decay = TensorUtils.separate_no_decay(self)
        optimizers = [
            self.optimizer_factory(params=decay),
            self.optimizer_factory(params=no_decay, weight_decay=0.0),
        ]
        return optimizers

    def get_schedulers(self, optimizers):
        if self.scheduler_factory is None:
            return []
        else:
            return [self.scheduler_factory(optimizer=optimizer) for optimizer in optimizers]

    def preprocess_input(self, data, train_mode=True):
        if train_mode and self.use_augmentation:
            data = self.aug(data)
        for key in self.shape_meta["observation"]["rgb"]:
            for obs_key in ("obs", "next_obs"):
                if obs_key in data:
                    x = TensorUtils.to_float(data[obs_key][key])
                    x = x / 255.0
                    x = torch.clip(x, 0, 1)
                    data[obs_key][key] = x
        
        action_norm_keys = ("abs_actions" if self.abs_action else "actions",)
        norm_keys = action_norm_keys + tuple(self.shape_meta["observation"]["lowdim"])
        data = self.normalizer.normalize(data, keys=norm_keys)

        return data

    def obs_encode(self, data, obs_key="obs"):
        return self.encoder(data, obs_key)

    def reset(self):
        return

    def get_task_emb(self, data):
        return self.encoder.get_task_emb(data)

    def get_action(self, obs, task_id, task_emb=None):
        self.eval()
        for key, value in obs.items():
            if key in self.shape_meta["rgb"]:
                value = ObsUtils.process_frame(value, channel_dim=3)
            obs[key] = torch.tensor(value)
        batch = {}
        batch["obs"] = obs
        if task_emb is not None:
            batch["task_emb"] = task_emb
        else:
            batch["task_id"] = torch.tensor([task_id], dtype=torch.long)
        batch = map_tensor_to_device(batch, self.device)
        with torch.no_grad():
            action = self.sample_actions(batch)
        action = self.normalizer.unnormalize({self.action_key: action})[self.action_key]
        return action

    def postprocess_action(self, action):
        if self.abs_action:
            pos, rot_raw, gripper = torch.split(action, [3, action.shape[-1] - 4, 1], dim=-1)
            rot = self.rotation_transformer.inverse(rot_raw)
            action = torch.cat([pos, rot, gripper], dim=-1)
        return action

    def preprocess_dataset(self, dataset, use_tqdm=True):
        return

    @abstractmethod
    def sample_actions(self, obs):
        raise NotImplementedError('Implement in subclass')
    


class ChunkPolicy(Policy):
    """
    Super class for policies which predict chunks of actions
    """

    def __init__(self, action_horizon, chunk_size, temporal_agg=False, **kwargs):
        super().__init__(**kwargs)

        self.action_horizon = action_horizon
        self.chunk_size = chunk_size
        self.temporal_agg = temporal_agg
        self.action_queue = None
        self.action_history = None
        self.batch_size = None
        self.actions_in_queue = 0

    def reset(self):
        if self.temporal_agg:
            if self.batch_size is not None:
                self.action_history = np.zeros(
                    (
                        self.batch_size,
                        self.chunk_size,
                        self.chunk_size,
                        self.network_action_dim,
                    )
                )
                self.actions_in_queue = 0
        else:
            self.action_queue = deque(maxlen=self.action_horizon)

    def get_action(self, obs, task_id, task_emb):
        if self.temporal_agg:
            actions = self._get_action_agg(obs, task_id, task_emb)
        else:
            actions = self._get_action_no_agg(obs, task_id, task_emb)
        
        actions = self.postprocess_action(actions)
        return actions

    def _get_action_agg(self, obs, task_id, task_emb):
        self.eval()
        if self.batch_size is None:
            self.batch_size = obs[list(obs.keys())[0]].shape[0]
            self.reset()

        batch = self._make_batch(obs, task_id, task_emb)
        with torch.no_grad():
            actions = self.sample_actions(batch)
            action_key = "abs_actions" if self.abs_action else "actions"
            actions = self.normalizer.unnormalize({action_key: actions})[action_key]
            actions = actions

        # Chop off the actions corresponding to the last timestep
        # and the oldest action in the history
        self.action_history = self.action_history[:, :-1, 1:]
        self.actions_in_queue = min(self.chunk_size, self.actions_in_queue + 1)
        self.action_history = np.concatenate(
            (
                self.action_history,
                np.zeros((self.batch_size, self.chunk_size - 1, 1, self.network_action_dim)),
            ),
            axis=2,
        )
        actions = einops.rearrange(actions, "b sbs d_act -> b 1 sbs d_act")
        self.action_history = np.concatenate((actions, self.action_history), axis=1)

        k = 0.01
        action_weights = np.concatenate((
            np.exp(-k * np.arange(self.actions_in_queue)),
            np.zeros(self.chunk_size - self.actions_in_queue)
            ))
        action_weights = action_weights / np.sum(action_weights)
        action_weights = action_weights.reshape((1, -1, 1))
        out_action = np.sum(self.action_history[:, 0] * action_weights, axis=1)

        return out_action

    def _get_action_no_agg(self, obs, task_id, task_emb=None):
        assert self.action_queue is not None, "you need to call policy.reset() before getting actions"

        if len(self.action_queue) == 0:
            batch = self._make_batch(obs, task_id, task_emb)
            with torch.no_grad():
                actions = self.sample_actions(batch)
                actions = self.normalizer.unnormalize({self.action_key: actions})[self.action_key]
                actions = np.transpose(actions, (1, 0, 2))
                self.action_queue.extend(actions[: self.action_horizon])
        action = self.action_queue.popleft()
        return action

    def _make_batch(self, obs, task_id, task_emb):
        for key, value in obs.items():
            if key in self.shape_meta["observation"]["rgb"]:
                value = ObsUtils.process_frame(value, channel_dim=3)
            elif key in self.shape_meta["observation"]["lowdim"]:
                value = TensorUtils.to_float(value)  # from double to float
            elif "depth" in key:
                value = ObsUtils.process_frame(value, channel_dim=1)
            obs[key] = torch.tensor(value)
        batch = {}
        batch["obs"] = obs
        if task_emb is not None:
            batch["task_emb"] = task_emb
        batch["task_id"] = torch.tensor([task_id], dtype=torch.long)
        batch = map_tensor_to_device(batch, self.device)
        return batch

    @abstractmethod
    def sample_actions(self, obs):
        raise NotImplementedError("Implement in subclass")
