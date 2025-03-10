import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from adapt3r.algos.utils.obs_core import CropRandomizer
from adapt3r.utils.point_cloud_utils import show_point_cloud
import pytorch3d.transforms as pt
import einops



class IdentityAug(nn.Module):
    def __init__(self, shape_meta=None, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class TranslationAug(nn.Module):
    """
    Utilize the random crop from robomimic.
    """

    def __init__(
        self,
        shape_meta,
        translation,
    ):
        super().__init__()

        # shapes = set()
        #     shapes.add(tuple(shape))
        
        # if len(shapes)
        self.randomizers = {}
        self.shape_meta = shape_meta


        for name, input_shape in shape_meta['observation']['rgb'].items():
            depth_name = name[:-4] + "_depth"
            if depth_name in shape_meta['observation']['depth']:
                input_shape[0] += 1
            pc_full_name = name[:-4] + '_pointcloud_full'
            if pc_full_name in self.shape_meta['observation']['pointcloud']:
                input_shape[0] += 3

            input_shape = tuple(input_shape)

            self.pad_translation = translation // 2
            pad_output_shape = (
                input_shape[0],
                input_shape[1] + translation,
                input_shape[2] + translation,
            )

            crop_randomizer = CropRandomizer(
                input_shape=pad_output_shape,
                crop_height=input_shape[1],
                crop_width=input_shape[2],
            )
            self.randomizers[input_shape] = crop_randomizer

    def forward(self, data):
        if self.training:

            for name in self.shape_meta['observation']['rgb']:
                obs_data = data['obs']
                x = obs_data[name]

                depth_name = name[:-4] + "_depth"
                if depth_name in self.shape_meta['observation']['depth']:
                    x = torch.cat((x, obs_data[depth_name]), dim=2)
                
                pc_full_name = name[:-4] + '_pointcloud_full'
                if pc_full_name in self.shape_meta['observation']['pointcloud']:
                    pc_full = einops.rearrange(obs_data[pc_full_name], 'b f h w d -> b f d h w')
                    x = torch.cat((x, pc_full), dim=2)

                batch_size, temporal_len, img_c, img_h, img_w = x.shape

                input_shape = (img_c, img_h, img_w)
                crop_randomizer = self.randomizers[input_shape]

                x = x.reshape(batch_size, temporal_len * img_c, img_h, img_w)
                out = F.pad(x, pad=(self.pad_translation,) * 4, mode="replicate")
                out = crop_randomizer.forward_in(out)
                out = out.reshape(batch_size, temporal_len, img_c, img_h, img_w)

                if depth_name in self.shape_meta['observation']['depth']:
                    depth = out[:, :, 3:4]
                    obs_data[depth_name] = depth
                if pc_full_name in self.shape_meta['observation']['pointcloud']:
                    pc_full = einops.rearrange(out[:, :, -3:], 'b f d h w -> b f h w d')
                    obs_data[pc_full_name] = pc_full
                out = out[:, :, :3]
                obs_data[name] = out
        return data


class ImgColorJitterAug(torch.nn.Module):
    """
    Conduct color jittering augmentation outside of proposal boxes
    """

    def __init__(
        self,
        shape_meta,
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.3,
        epsilon=0.05,
    ):
        super().__init__()
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )
        self.epsilon = epsilon
        self.shape_meta = shape_meta

    def forward(self, data):
        if self.training and np.random.rand() > self.epsilon:
            for name in self.shape_meta['observation']['rgb']:
                data['obs'][name] = self.color_jitter(data['obs'][name])
        return data


class BatchWiseImgColorJitterAug(torch.nn.Module):
    """
    Color jittering augmentation to individual batch.
    This is to create variation in training data to combat
    BatchNorm in convolution network.
    """

    def __init__(
        self,
        shape_meta,
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.3,
        epsilon=0.1,
    ):
        super().__init__()
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )
        self.epsilon = epsilon
        self.shape_meta = shape_meta

    def forward(self, data):
        if self.training:
            for name in self.shape_meta['observation']['rgb']:
                x = data['obs'][name]
                mask = torch.rand((x.shape[0], *(1,)*(len(x.shape)-1)), device=x.device) > self.epsilon
                
                jittered = self.color_jitter(x)

                out = mask * jittered + torch.logical_not(mask) * x
                data['obs'][name] = out
                
        return data


class DataAugGroup(nn.Module):
    """
    Add augmentation to multiple inputs
    """

    def __init__(self, aug_list, shape_meta):
        super().__init__()
        aug_list = [aug(shape_meta) for aug in aug_list]
        self.aug_layer = nn.Sequential(*aug_list)

    def forward(self, data):
        return self.aug_layer(data)
    