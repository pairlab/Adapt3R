from abc import ABC, abstractmethod
from collections import deque

import einops
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR, ConstantLR

import adapt3r.utils.obs_utils as ObsUtils

# from adapt3r.modules.v1 import *
import adapt3r.utils.tensor_utils as TensorUtils
from adapt3r.algos.encoders.base import BaseEncoder
from adapt3r.algos.utils.normalizer import Normalizer
from adapt3r.algos.utils.rotation_transformer import RotationTransformer
from adapt3r.utils.utils import map_tensor_to_device
from adapt3r.utils.geometry import posRotMat2Mat, quat2mat
import adapt3r.utils.point_cloud_utils as pcu
import adapt3r.utils.pytorch3d_transforms as p3d
import adapt3r.envs.utils as eu


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
        shape_meta,
        abs_action,
        device,
        eecf=False,
        bimanual=False,
        normalizer: Normalizer = None,
        rot_rep=None
    ):
        super().__init__()

        self.encoder = encoder
        self.use_augmentation = aug_factory is not None
        self.shape_meta = shape_meta

        self.optimizer_factory = optimizer_factory
        if normalizer is None:
            normalizer = Normalizer(mode="identity")
        self.normalizer = normalizer
        self.abs_action = abs_action
        self.eecf = eecf
        self.bimanual = bimanual
        self.device = device

        # Use 6D actions if we are predicting abs actions, else axis angle
        if rot_rep is None:
            self.rot_rep = "rotation_6d" if abs_action else "axis_angle"
        else:
            self.rot_rep = rot_rep
        rot_rep_in = shape_meta["rotation_rep_in_abs"] if abs_action else shape_meta["rotation_rep_in"]
        self.rotation_transformer = RotationTransformer(
            rep_in=rot_rep_in,
            rep_network=self.rot_rep,
            rep_out=shape_meta["rotation_rep_out"]
        )

        # Note: for unimanual tasks, the order is pos, rot, gripper
        # For bimanual tasks, the order is right_pos, right_rot, left_pos, 
        # left_rot, right_gripper, left_gripper. Therefore the following works
        # Convention note: "network" refers to the representation used in the network
        action_meta = self.shape_meta["actions"] if not abs_action else self.shape_meta["abs_actions"]
        self.network_action_dim = sum(action_meta.values())
        adjustment = self.rotation_transformer.get_network_size() - self.rotation_transformer.get_input_size()
        if self.bimanual:
            adjustment *= 2
        self.network_action_dim += adjustment
        self.pos_dim = 3
        self.rot_dim_in = self.rotation_transformer.get_input_size()
        self.rot_dim_network = self.rotation_transformer.get_network_size()
        self.rot_dim_out = self.rotation_transformer.get_output_size()
        if self.bimanual:
            per_hand_dim = self.network_action_dim // 2
            self.gripper_dim = per_hand_dim - self.pos_dim - self.rot_dim_network
        else:
            self.gripper_dim = self.network_action_dim - self.pos_dim - self.rot_dim_network

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

    def get_schedulers(self, optimizers, total_steps, schedule_type, warmup_steps, lr, end_factor=0.01):
        return [
            create_scheduler(optimizer, total_steps, schedule_type, warmup_steps, lr, end_factor) 
            for optimizer in optimizers
        ]

    @torch.no_grad()
    def preprocess_input(self, data, train_mode=True):
        camera_names = eu.list_cameras(self.shape_meta)
        for camera_name in camera_names:
            image_name = eu.camera_name_to_image_key(camera_name)
            if image_name in data["obs"]:
                data["obs"][image_name] = data["obs"][image_name] / 255.0
                data["obs"][image_name] = torch.clip(data["obs"][image_name], 0, 1)
            depth_name = eu.camera_name_to_depth_key(camera_name)
            if depth_name in data["obs"]:
                data["obs"][depth_name] = data["obs"][depth_name] / 1000

        if train_mode and self.use_augmentation:
            data = self.aug(data)

        if train_mode:
            self.preprocess_actions(data)

        norm_keys = ('actions',) + tuple(self.shape_meta["observation"]["lowdim"])
        data = self.normalizer.normalize(data, keys=norm_keys)

        return data
    
    def decompose_actions(self, actions):
        if actions.shape[-1] == self.network_action_dim:
            rot_dim = self.rot_dim_network
        else:
            rot_dim = self.rot_dim_in
        
        if self.bimanual:
            right_pos, right_rot, left_pos, left_rot, right_gripper, left_gripper = \
                  torch.split(actions, [
                      self.pos_dim, 
                      rot_dim, 
                      self.pos_dim, 
                      rot_dim, 
                      self.gripper_dim, 
                      self.gripper_dim], dim=-1)
            return [[right_pos, right_rot, right_gripper], [left_pos, left_rot, left_gripper]]
        else:
            pos, rot, gripper = torch.split(actions, [self.pos_dim, rot_dim, self.gripper_dim], dim=-1)
            return [[pos, rot, gripper]]
        
    def reassemble_actions(self, actions_decomp):
        final_action = []
        for hand in actions_decomp:
            final_action.append(torch.cat([hand[0], hand[1]], dim=-1))
        for hand in actions_decomp:
            final_action.append(hand[2])
        
        return torch.cat(final_action, dim=-1)

    def preprocess_actions(self, data):
        actions = data["actions"]

        actions_decomp = self.decompose_actions(actions)

        if self.bimanual:
            hand_mat_invs = [data[f"obs"][f"robot0_{hand}_eef_mat_inv"][:, -1] for hand in ["right", "left"]]
        else:
            hand_mat_invs = [data[f"obs"][f"hand_mat_inv"][:, -1]]
        
        for i in range(len(actions_decomp)):
            pos, rot, gripper = actions_decomp[i]
            rot_network = self.rotation_transformer.preprocess(rot)
            if self.eecf:
                hand_mat_inv = hand_mat_invs[i]
                if self.abs_action:
                    rot_mat = self.rotation_transformer.network_to_matrix(rot_network)
                    mat = pcu.pos_rot_mat_to_mat(pos, rot_mat)
                    mat_eecf = torch.einsum('bij,bnjk->bnik', hand_mat_inv, mat)
                    pos, rot_network_eecf = pcu.matrix_to_pos_rot_matrix(mat_eecf)
                    rot_network = self.rotation_transformer.matrix_to_network(rot_network_eecf)
                else:
                    pos = torch.einsum("...ij,...nj->...ni", hand_mat_inv[..., :3, :3], pos)
            actions_decomp[i] = [pos, rot_network, gripper]
        
        actions = self.reassemble_actions(actions_decomp)

        data["actions"] = actions
        return data

    def postprocess_actions(self, data):
        actions = data['actions']
        actions_decomp = self.decompose_actions(actions)

        if self.bimanual:
            hand_mats = [data[f"obs"][f"robot0_{hand}_eef_mat"][:, -1] for hand in ["right", "left"]]
        else:
            hand_mats = [data[f"obs"][f"hand_mat"][:, -1]]

        for i in range(len(actions_decomp)):
            pos, rot_network, gripper = actions_decomp[i]
            if self.eecf:
                hand_mat = hand_mats[i]
                if self.abs_action:
                    rot_mat = self.rotation_transformer.network_to_matrix(rot_network)
                    mat_eecf = pcu.pos_rot_mat_to_mat(pos, rot_mat)
                    mat = torch.einsum('bij,bnjk->bnik', hand_mat, mat_eecf)
                    pos, rot_network_eecf = pcu.matrix_to_pos_rot_matrix(mat)
                    rot_network = self.rotation_transformer.matrix_to_network(rot_network_eecf)
                else:
                    pos = torch.einsum("...ij,...j->...i", hand_mat[..., :3, :3], pos)
            actions_decomp[i] = [pos, rot_network, gripper]
        actions = self.reassemble_actions(actions_decomp)
        data["actions"] = actions
        return data

    # This needs to be separate because if we have temporal aggregation and abs_actions, it is 
    # important that the aggregation is done with 6D rotations
    def final_postprocess_actions(self, action):
        action_decomp = self.decompose_actions(action)

        for i in range(len(action_decomp)):
            pos, rot, gripper = action_decomp[i]
            rot = self.rotation_transformer.postprocess(rot)
            action_decomp[i] = [pos, rot, gripper]
        action = self.reassemble_actions(action_decomp)

        return action
    
    def obs_encode(self, data, obs_key="obs"):
        return self.encoder(data, obs_key)

    def reset(self):
        return

    def get_task_emb(self, data):
        return self.encoder.get_task_emb(data)

    # TODO: this is out of date and not used since we exclusively use the chunk policy which overrides this
    def get_action(self, obs, task_id, **kwargs):
        self.eval()
        batch = self._make_batch(obs, task_id, **kwargs)
        with torch.no_grad():
            action = self.sample_actions(batch)
        action = self.normalizer.unnormalize({self.action_key: action})[self.action_key]
        return action
    
    def _make_batch(self, obs, task_id, **kwargs):
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
        if kwargs is not None:
            batch.update(kwargs)
        batch["task_id"] = torch.tensor([task_id], dtype=torch.long)
        batch = map_tensor_to_device(batch, self.device)
        return batch


    def preprocess_dataset(self, dataset, use_tqdm=True):
        return

    @abstractmethod
    def sample_actions(self, obs):
        raise NotImplementedError("Implement in subclass")


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

    def get_action(self, obs, task_id, **kwargs):
        if self.temporal_agg:
            actions = self._get_action_agg(obs, task_id, **kwargs)
        else:
            actions = self._get_action_no_agg(obs, task_id, **kwargs)
        actions = self.final_postprocess_actions(actions)
        return actions.to(torch.float32).cpu().numpy()

    def _get_action_agg(self, obs, task_id, **kwargs):  # obs, task_id, task_emb=None):
        self.eval()
        if self.batch_size is None:
            self.batch_size = obs[list(obs.keys())[0]].shape[0]
            self.reset()

        batch = self._make_batch(obs, task_id, **kwargs)
        with torch.no_grad():
            actions = self.sample_actions(batch)
            actions = self.normalizer.unnormalize({"actions": actions})["actions"]
            batch['actions'] = torch.tensor(actions, device=self.device)
            actions = self.postprocess_actions(batch)['actions']
            actions = actions.cpu().numpy()

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

        action_sums = np.sum(self.action_history, axis=1)
        action_denoms = self.chunk_size - np.arange(self.chunk_size)
        action_denoms = np.minimum(action_denoms, self.actions_in_queue)
        action_denoms = einops.repeat(action_denoms, "sbs -> B sbs 1", B=self.batch_size)
        out_actions = action_sums / action_denoms
        out_actions = torch.tensor(out_actions)
        return out_actions[:, 0]

    def _get_action_no_agg(self, obs, task_id, **kwargs):
        assert (
            self.action_queue is not None
        ), "you need to call policy.reset() before getting actions"

        if len(self.action_queue) == 0:
            batch = self._make_batch(obs, task_id, **kwargs)
            with torch.no_grad():
                actions = self.sample_actions(batch)
                actions = self.normalizer.unnormalize({"actions": actions})["actions"]
                batch['actions'] = torch.tensor(actions, device=self.device)
                actions = self.postprocess_actions(batch)['actions']
                actions = actions.cpu().numpy()
                actions = np.transpose(actions, (1, 0, 2))
                self.action_queue.extend(actions[: self.action_horizon])
        action = self.action_queue.popleft()
        return torch.tensor(action)
        

    @abstractmethod
    def sample_actions(self, obs):
        raise NotImplementedError("Implement in subclass")


def create_scheduler(optimizer, total_steps, schedule_type, warmup_steps, lr, end_factor=0.01):
    if schedule_type is None:
        return []
    eta_min = end_factor * lr
    # If no warmup is requested, just return the main scheduler
    if warmup_steps <= 0:
        if schedule_type == 'cosine':
            return CosineAnnealingLR(optimizer, eta_min=eta_min, T_max=total_steps)
        elif schedule_type == 'linear':
            return LinearLR(optimizer, start_factor=1.0, end_factor=end_factor, total_iters=total_steps)
        elif schedule_type == 'constant':
            return ConstantLR(optimizer, factor=1.0)
        else:
            raise ValueError(f"Unknown scheduler type: {schedule_type}")
    
    # Create warmup scheduler
    warmup_scheduler = LinearLR(
        optimizer, 
        start_factor=0.001,
        end_factor=1.0,
        total_iters=warmup_steps
    )
    
    # Create main scheduler based on the specified type
    if schedule_type == 'cosine':
        main_scheduler = CosineAnnealingLR(optimizer, eta_min=eta_min, T_max=total_steps - warmup_steps)
    elif schedule_type == 'linear':
        main_scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=end_factor,
            total_iters=total_steps - warmup_steps
        )
    elif schedule_type == 'constant':
        main_scheduler = ConstantLR(optimizer, factor=1.0)
    else:
        raise ValueError(f"Unknown scheduler type: {schedule_type}")
    
    # Combine schedulers
    return SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_steps]
    )