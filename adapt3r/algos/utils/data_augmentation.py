import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from adapt3r.algos.utils.obs_core import CropRandomizer
import adapt3r.utils.camera_utils as cu

import adapt3r.envs.utils as eu


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
        use_image=True,
        use_depth=False,
    ):
        super().__init__()

        self.randomizers = {}
        self.shape_meta = shape_meta
        self.use_image = use_image
        self.use_depth = use_depth

        obs_meta = shape_meta['observation']

        # for camera_name in shape_meta['observation']['rgb'].items():
        for camera_name in eu.list_cameras(shape_meta):
            channels = 0
            rgb_name = eu.camera_name_to_image_key(camera_name)
            depth_name = eu.camera_name_to_depth_key(camera_name)
            if use_image and rgb_name in obs_meta['rgb']:
                channels += 3
                size = obs_meta['rgb'][rgb_name][1:]
            if use_depth and depth_name in obs_meta['depth']:
                channels += 1
                size = obs_meta['depth'][depth_name][1:]
            
            input_shape = [channels] + size

            # pc_full_name = camera_name + '_pointcloud_full'
            # if pc_full_name in self.shape_meta['observation']['pointcloud']:
            #     input_shape[0] += 3

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

            obs_data = data['obs']
            for camera_name in eu.list_cameras(self.shape_meta):
                x = []
                rgb_name = eu.camera_name_to_image_key(camera_name)
                depth_name = eu.camera_name_to_depth_key(camera_name)
                if rgb_name in obs_data:
                    x.append(obs_data[rgb_name])
                if depth_name in obs_data and self.use_depth:
                    x.append(obs_data[depth_name])

                x = torch.cat(x, dim=2)
                
                batch_size, temporal_len, img_c, img_h, img_w = x.shape

                input_shape = (img_c, img_h, img_w)
                crop_randomizer = self.randomizers[input_shape]

                intrinsic_name = eu.camera_name_to_intrinsic_key(camera_name)
                if intrinsic_name in obs_data:
                    intrinsics = obs_data[intrinsic_name]
                    # intrinsics = einops.rearrange(intrinsics, 'b t i j -> (b t) i j')
                    intrinsics = cu.pad_update_intrinsics(intrinsics, self.pad_translation)
                else:
                    intrinsics = None

                x = x.reshape(batch_size, temporal_len * img_c, img_h, img_w)
                out = F.pad(x, pad=(self.pad_translation,) * 4, mode="replicate")
                out, intrinsics = crop_randomizer.forward_in((out, intrinsics))
                out = out.reshape(batch_size, temporal_len, img_c, img_h, img_w)

                if rgb_name in obs_data:
                    rgb = out[:, :, :3]
                    out = out[:, :, 3:]
                    obs_data[rgb_name] = rgb
                if depth_name in obs_data and self.use_depth:
                    depth = (out)
                    obs_data[depth_name] = depth

                if intrinsics is not None:
                    obs_data[intrinsic_name] = intrinsics
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
            obs_data = data['obs']
            for camera_name in eu.list_cameras(self.shape_meta):
                image_name = eu.camera_name_to_image_key(camera_name)
                if image_name not in obs_data:
                    continue

                x = obs_data[image_name]
                mask = torch.rand((x.shape[0], *(1,)*(len(x.shape)-1)), device=x.device) > self.epsilon
                
                jittered = self.color_jitter(x)

                out = mask * jittered + torch.logical_not(mask) * x

                obs_data[image_name] = out
        
        # self.count += 1
        return data


class EfficientBatchWiseImgColorJitterAug(torch.nn.Module):
    """
    This version is slower than the other one with no upsides.
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
        import kornia.augmentation as K
        self.color_jitter = K.ColorJiggle(
            brightness=brightness, 
            contrast=contrast, 
            saturation=saturation, 
            hue=hue, 
            p=epsilon,
        )
        self.epsilon = epsilon
        self.shape_meta = shape_meta

    def forward(self, data):
        if self.training:
            obs_data = data['obs']
            camera_names = eu.list_cameras(self.shape_meta)

            ims = []
            for camera_name in camera_names:
                image_name = eu.camera_name_to_image_key(camera_name)
                if image_name not in obs_data:
                    continue
                
                ims.append(obs_data[image_name])

            ims = torch.stack(ims, dim=1)
            B, N, T, C, H, W = ims.shape

            ims = ims.reshape(B * N * T, C, H, W)
            jittered = self.color_jitter(ims)
            jittered = jittered.reshape(B, N, T, C, H, W)
            jittered_ims = torch.unbind(jittered, dim=1)
            for i, camera_name in enumerate(camera_names):
                image_name = eu.camera_name_to_image_key(camera_name)
                if image_name in obs_data:
                    obs_data[image_name] = jittered_ims[i]

            # for camera_name in eu.list_cameras(self.shape_meta):
            #     image_name = eu.camera_name_to_image_key(camera_name)
            #     if image_name not in obs_data:
            #         continue

            #     x = obs_data[image_name]
            #     B, T, C, H, W = x.shape

            #     x = x.reshape(B * T, C, H, W)
            #     jittered = self.color_jitter(x)
            #     jittered = jittered.reshape(B, T, C, H, W)

            #     obs_data[image_name] = jittered
        
        # self.count += 1
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
    