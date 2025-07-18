"""
This file contains all neural modules related to encoding the spatial
information of obs_t, i.e., the abstracted knowledge of the current visual
input conditioned on the language.
"""

import einops

import torch
import torch.nn as nn

from adapt3r.algos.utils.misc import weight_init
from adapt3r.algos.encoders.base import BaseEncoder
import adapt3r.envs.utils as eu



class RGBEncoder(BaseEncoder):
    def __init__(
        self,
        image_encoder_factory,
        lowdim_encoder_factory,
        share_image_encoder=True,
        share_lowdim_encoder=True,
        language_fusion=False,
        load_depth=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.language_fusion = language_fusion
        language_fusion_input = "film" if language_fusion else None
        self.load_depth = load_depth

        do_lowdim = lowdim_encoder_factory is not None

        # observation encoders
        self.image_encoders = {}
        obs_meta = self.shape_meta["observation"]
        if share_image_encoder:
            assert (
                len(set([tuple(shape) for _, shape in obs_meta["rgb"].items()])) == 1
            ), "all images must have the same shape"
            assert len(obs_meta["depth"]) in (
                0,
                len(obs_meta["rgb"]),
            ), "must either have no depth input or a depth input for each image"
            shape = list(list(obs_meta["rgb"].items())[0][1])
            if load_depth:
                shape[0] += 1
            self.image_encoders = image_encoder_factory(
                shape, 
                language_fusion=language_fusion_input, 
                language_dim=self.lang_embed_dim
            )
            self.n_out_perception = self.frame_stack * len(eu.list_cameras(self.shape_meta)) * self.image_encoders.n_out
            self.d_out_perception = self.image_encoders.out_channels
        else:

            for camera_name in eu.list_cameras(self.shape_meta):
                # depth_name = camera_name + "_depth"
                depth_name = eu.camera_name_to_depth_key(camera_name)
                rgb_name = eu.camera_name_to_image_key(camera_name)
                shape_in = list(self.shape_meta["observation"]["rgb"][rgb_name])
                if load_depth:
                    assert depth_name in obs_meta["depth"], f"depth {depth_name} not found in obs_meta"
                    shape_in[0] += 1
                encoder = image_encoder_factory(
                    shape_in,
                    language_fusion=language_fusion_input,
                    language_dim=self.lang_embed_dim,
                )

                self.image_encoders[camera_name] = encoder
                self.n_out_perception += self.frame_stack * encoder.n_out
                self.d_out_perception = encoder.out_channels

            self.image_encoders = nn.ModuleDict(self.image_encoders)

        self.lowdim_encoders = {}
        if do_lowdim and len(obs_meta["lowdim"]) > 0:
            if share_lowdim_encoder:
                total_lowdim = 0
                for name, shape in obs_meta["lowdim"].items():
                    total_lowdim += shape
                encoder = lowdim_encoder_factory(total_lowdim)
                encoder.apply(weight_init)
                self.lowdim_encoders = encoder
                self.n_out_lowdim += 1
                self.d_out_lowdim = encoder.out_channels
            else:
                for name, shape in obs_meta["lowdim"].items():
                    encoder = lowdim_encoder_factory(shape)
                    encoder.apply(weight_init)
                    self.lowdim_encoders[name] = encoder
                    self.n_out_lowdim += 1
                    self.d_out_lowdim = encoder.out_channels
                self.lowdim_encoders = nn.ModuleDict(self.lowdim_encoders)

    def forward(self, data, obs_key):
        obs_data = data[obs_key]

        ### 1. encode image
        img_encodings, lowdim_encodings = [], []
        langs = self.get_task_emb(data) if self.language_fusion else None

        for camera_name in eu.list_cameras(self.shape_meta):
            x = obs_data[eu.camera_name_to_image_key(camera_name)]
            x = torch.clip(x, 0, 1)
            obs_data[eu.camera_name_to_image_key(camera_name)] = x

        if type(self.image_encoders) in (dict, nn.ModuleDict):
            for camera_name in eu.list_cameras(self.shape_meta):
                img_name = eu.camera_name_to_image_key(camera_name)
                depth_name = eu.camera_name_to_depth_key(camera_name)
                
                x = obs_data[img_name]
                if self.load_depth:
                    depth = torch.clamp(obs_data[depth_name], 0.001, 5) - 2.5
                    x = torch.cat((x, depth), dim=2)

                B, T, C, H, W = x.shape
                e = self.image_encoders[img_name](
                    x.reshape(B * T, C, H, W),
                    langs=langs
                    )
                e = e.view(B, T, *e.shape[1:])
                e = list(einops.rearrange(e, "b t m d -> (t m) b d"))
                img_encodings.extend(e)
        else:
            imgs = []
            for camera_name in eu.list_cameras(self.shape_meta):
                img_name = eu.camera_name_to_image_key(camera_name)
                img = obs_data[img_name]
                
                camera_name = eu.image_key_to_camera_name(img_name)
                depth_name = camera_name + "_depth"
                if self.load_depth:
                    depth = torch.clamp(obs_data[depth_name], 0.001, 5) - 2.5
                    img = torch.cat((img, depth), dim=2)

                imgs.append(img)
            x = torch.stack(imgs, dim=1)
            B, N, T, C, H, W = x.shape
            x = einops.rearrange(x, 'b n t c h w -> (b n t) c h w')
            langs = einops.repeat(langs, 'b d -> (b n t) d', n=N, t=T)
            x = self.image_encoders(x, langs=langs)
            x = x.view(B, N, T, *x.shape[1:])
            img_encodings = list(einops.rearrange(x, "b ncam t m d -> (ncam t m) b d"))

        # 2. add proprio info
        if type(self.lowdim_encoders) in (dict, nn.ModuleDict):
            for lowdim_name in self.lowdim_encoders.keys():
                x = self.lowdim_encoders[lowdim_name](obs_data[lowdim_name])  # (B, T, H_extra)
                x = list(einops.rearrange(x, "b t d -> t b d"))
                lowdim_encodings.extend(x)
        else:
            lowdims = []
            for lowdim_name in self.shape_meta["observation"]["lowdim"].keys():
                lowdims.append(obs_data[lowdim_name])
            lowdim_input = torch.cat(lowdims, dim=-1)
            x = self.lowdim_encoders(lowdim_input)
            x = list(einops.rearrange(x, "b t d -> t b d"))
            lowdim_encodings.extend(x)

        return img_encodings, lowdim_encodings
