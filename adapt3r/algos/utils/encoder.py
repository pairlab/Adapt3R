"""
This file contains all neural modules related to encoding the spatial
information of obs_t, i.e., the abstracted knowledge of the current visual
input conditioned on the language.
"""

import einops

import dgl.geometry as dgl_geo
import pytorch3d.ops as torch3d_ops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import FeaturePyramidNetwork

# import adapt3r.utils.libero_utils as lu
from adapt3r.algos.utils.misc import weight_init
from adapt3r.algos.utils.rgb_modules import ResnetEncoder

from .clip import load_clip
from .resnet import load_resnet18, load_resnet50

from adapt3r.algos.utils.resnet import load_resnet50, load_resnet18
from adapt3r.algos.utils.clip import load_clip
from adapt3r.algos.utils.position_encodings import NeRFSinusoidalPosEmb
from adapt3r.algos.utils.misc import weight_init
import adapt3r.env.libero.utils as lu
from adapt3r.utils.point_cloud_utils import show_point_cloud, batch_transform_point_cloud, crop_point_cloud
import matplotlib.pyplot as plt



class BaseEncoder(nn.Module):
    def __init__(
        self,
        shape_meta,
        lang_embed_dim,
        **kwargs
    ):
        super().__init__()
        self.shape_meta = shape_meta
        self.lang_embed_dim = lang_embed_dim
        if len(kwargs) > 0:
            print('warning: unused kwargs:', set(kwargs))

        # These need to be populated by subclasses
        self.n_out_perception = 0
        self.d_out_perception = 0
        self.n_out_proprio = 0
        self.d_out_proprio = 0

        if shape_meta.task.type == "onehot":
            self.task_encoder = nn.Embedding(
                num_embeddings=shape_meta.task.n_tasks, embedding_dim=lang_embed_dim
            )
        else:
            self.task_encoder = nn.Linear(shape_meta.task.dim, lang_embed_dim)

    def get_task_emb(self, data):
        if "task_emb" in data:
            return self.task_encoder(data["task_emb"])
        else:
            return self.task_encoder(data["task_id"])


class DefaultEncoder(BaseEncoder):
    def __init__(
        self,
        image_encoder_factory,
        lowdim_encoder_factory,
        separate_depth=False,
        share_image_encoder=True,
        share_lowdim_encoder=True,
        language_fusion=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.language_fusion = language_fusion
        language_fusion_input = "film" if language_fusion else None

        do_image = image_encoder_factory is not None
        do_lowdim = lowdim_encoder_factory is not None
        total_obs_channels = 0

        # observation encoders
        self.image_encoders = {}
        obs_meta = self.shape_meta["observation"]
        if do_image and len(obs_meta["rgb"]) > 0:
            if share_image_encoder:
                assert (
                    len(set([tuple(shape) for _, shape in obs_meta["rgb"].items()])) == 1
                ), "all images must have the same shape"
                assert len(obs_meta["depth"]) in (
                    0,
                    len(obs_meta["rgb"]),
                ), "must either have no depth input or a depth input for each image"
                shape = list(list(obs_meta["rgb"].items())[0][1])
                if len(obs_meta["depth"]):
                    shape[0] += 1
                shared_encoder = image_encoder_factory(
                    shape, language_fusion=language_fusion_input, language_dim=self.lang_embed_dim
                )
            else:
                shared_encoder = None

            for name, shape in obs_meta["rgb"].items():
                if shared_encoder is None:
                    depth_name = name[:-4] + "_depth"
                    shape_in = list(shape)
                    if depth_name in obs_meta["depth"]:
                        assert (
                            not separate_depth
                        ), "separating depth into its own encoder is not yet supported"
                        shape_in[0] += 1
                    encoder = image_encoder_factory(
                        shape_in,
                        language_fusion=language_fusion_input,
                        language_dim=self.lang_embed_dim,
                    )
                else:
                    encoder = shared_encoder

                total_obs_channels += encoder.out_channels
                self.image_encoders[name] = encoder
                self.n_out_perception += 1
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
                self.n_out_proprio += 1
                self.d_out_proprio = encoder.out_channels
            else:
                for name, shape in obs_meta["lowdim"].items():
                    encoder = lowdim_encoder_factory(shape)
                    encoder.apply(weight_init)
                    total_obs_channels += encoder.out_channels
                    self.lowdim_encoders[name] = encoder
                    self.n_out_proprio += 1
                    self.d_out_proprio = encoder.out_channels
                self.lowdim_encoders = nn.ModuleDict(self.lowdim_encoders)

    def forward(self, data, obs_key):
        obs_data = data[obs_key]

        ### 1. encode image
        perception_encodings, proprio_encodings = [], []
        langs = self.get_task_emb(data) if self.language_fusion else None
        for img_name in self.image_encoders.keys():
            x = obs_data[img_name]

            depth_name = img_name[:-4] + "_depth"
            if depth_name in obs_data:
                depth = obs_data[depth_name]
                x = torch.cat((x, depth), dim=2)

            B, T, C, H, W = x.shape
            e = self.image_encoders[img_name](x.reshape(B * T, C, H, W), langs=langs)
            e = e.view(B, T, *e.shape[1:])
            perception_encodings.append(e)

        # 2. add proprio info
        if type(self.lowdim_encoders) in (dict, nn.ModuleDict):
            for lowdim_name in self.lowdim_encoders.keys():
                proprio_encodings.append(
                    self.lowdim_encoders[lowdim_name](obs_data[lowdim_name])
                )  # add (B, T, H_extra)
        else:
            lowdims = []
            for lowdim_name in self.shape_meta["observation"]["lowdim"].keys():
                lowdims.append(obs_data[lowdim_name])
            lowdim_input = torch.cat(lowdims, dim=-1)
            proprio_encodings.append(self.lowdim_encoders(lowdim_input))
        return perception_encodings, proprio_encodings


class HybridEncoder(BaseEncoder):
    def __init__(
        self,
        backbone_type,
        pointcloud_extractor_factory,
        lowdim_encoder_factory,
        num_points,
        hidden_dim,
        do_image=True,
        do_pos=True,
        do_lang=True,
        do_rgb=False,
        do_crop=True,
        do_hand_crop=True,
        tight_crop=False,
        hand_frame=True,
        do_rot_aug=False,
        finetune=False,
        task_suite_name=None,  # I don't like having this here but for now it's necessary to get boundary info
        task_benchmark_name=None,
        xyz_proj_type="nerf",
        downsample_mode="pos",
        vis_attn=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        pc_in = (
            (do_image + do_lang) * hidden_dim
            + do_pos * (3 if xyz_proj_type == "none" else hidden_dim)
            + (3 if do_rgb else 0)
        )
        self.pointcloud_extractor = pointcloud_extractor_factory(in_shape=pc_in)
        self.pointcloud_extractor.apply(weight_init)
        self.num_points = num_points
        self.do_crop = do_crop
        self.do_hand_crop = do_hand_crop
        self.hand_frame = hand_frame
        self.do_rot_aug = do_rot_aug
        self.shape_meta = self.shape_meta
        self.do_image = do_image
        self.do_pos = do_pos
        self.do_lang = do_lang
        self.do_rgb = do_rgb
        self.n_out_perception = 1
        self.d_out_perception = self.pointcloud_extractor.out_channels
        self.downsample_mode = downsample_mode
        self.vis_attn = vis_attn

        do_lowdim = lowdim_encoder_factory is not None

        if do_lowdim and len(self.shape_meta["observation"]["lowdim"]) > 0:
            d_lowdim = 0
            for name, shape in self.shape_meta["observation"]["lowdim"].items():
                d_lowdim += shape
            encoder = lowdim_encoder_factory(d_lowdim)
            encoder.apply(weight_init)
            self.lowdim_encoder = encoder
            self.n_out_proprio += 1
            self.d_out_proprio = encoder.out_channels
        else:
            self.lowdim_encoder = None

        image_keys = self.shape_meta["observation"]["rgb"].keys()
        pointcloud_keys = self.shape_meta["observation"]["pointcloud"].keys()
        assert len(image_keys) == len(pointcloud_keys)
        for key in image_keys:
            assert image_key_to_pointcloud_key(key) in pointcloud_keys

        image_size = tuple(self.shape_meta["observation"]["rgb"][next(iter(image_keys))])

        self.backbone_type = backbone_type
        if backbone_type == "resnet50":
            self.backbone, self.normalize = load_resnet50()
        elif backbone_type == "resnet18":
            self.backbone, self.normalize = load_resnet18()
        elif backbone_type == "fusion":
            self.backbone = ResnetEncoder(
                input_shape=image_size,
                language_fusion="film",
                language_dim=self.lang_embed_dim,
                do_projection=False,
                return_all_feats=True,
            )
            # The libero backbone already normalizes
            self.normalize = nn.Identity()
        elif backbone_type == "clip":
            self.backbone, self.normalize = load_clip()
        else:
            raise NotImplementedError(f"backbone type {backbone_type} not supported")

        self.finetune = finetune
        if not finetune:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.feature_pyramid = FeaturePyramidNetwork(
            [256], hidden_dim
        )

        if xyz_proj_type == "nerf":
            self.xyz_proj = NeRFSinusoidalPosEmb(hidden_dim)
        elif xyz_proj_type == "learned":
            self.xyz_proj = nn.Sequential(
                nn.Linear(3, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, hidden_dim),
            )
            self.xyz_proj.apply(weight_init)
        elif xyz_proj_type == "none":
            self.xyz_proj = nn.Identity()
        else:
            raise ValueError()

        if task_suite_name == "libero":
            boundaries = lu.get_boundaries(benchmark_name=task_benchmark_name, tight=tight_crop)
            hand_frame_boundaries = torch.tensor(((0, -1, -1), (1, 1, 1)), dtype=torch.float32)
        elif task_suite_name == "metaworld":
            boundaries = torch.tensor(((-1, -1, -1), (1, 1, 1)))
            boundaries = einops.repeat(boundaries, "i j -> 50 i j")
            hand_frame_boundaries = torch.tensor(((0, -1, -1), (1, 1, 1)), dtype=torch.float32)
        self.register_buffer("boundaries", torch.tensor(boundaries, dtype=torch.float32))
        self.register_buffer('hand_frame_boundaries', hand_frame_boundaries)

        if self.lang_embed_dim != hidden_dim:
            self.lang_proj = nn.Linear(self.lang_embed_dim, hidden_dim)
        else:
            self.lang_proj = nn.Identity()

    def forward(self, data, obs_key):
        obs_data = data[obs_key]

        rgb = []
        pcd = []
        for rgb_key in self.shape_meta["observation"]["rgb"]:
            rgb.append(obs_data[rgb_key])
            pcd.append(obs_data[image_key_to_pointcloud_key(rgb_key)])

        rgb = torch.stack(rgb)
        pcd = torch.stack(pcd)

        device = rgb.device

        n_cam, B, fs, _, _, _ = rgb.shape

        rgb = einops.rearrange(rgb, "ncam b fs c h w -> (b fs ncam) c h w")
        pcd = einops.rearrange(pcd, "ncam b fs h w c -> (b fs ncam) c h w")

        # Pass each view independently through backbone
        if self.do_image:
            rgb_normalized = self.normalize(rgb)
            if self.finetune:
                if self.backbone_type == "fusion":
                    task_emb = einops.repeat(
                        data["task_emb"], "b d -> (b fs ncam) d", fs=fs, ncam=n_cam
                    )
                    rgb_features = self.backbone(rgb_normalized, langs=task_emb)
                else:
                    rgb_features = self.backbone(rgb_normalized)
            else:
                with torch.no_grad():
                    if self.backbone_type == "fusion":
                        task_emb = einops.repeat(
                            data["task_emb"], "b d -> (b fs ncam) d", fs=fs, ncam=n_cam
                        )
                        rgb_features = self.backbone(rgb_normalized, langs=task_emb)
                    else:
                        rgb_features = self.backbone(rgb_normalized)

            # Pass visual features through feature pyramid network
            rgb_features = self.feature_pyramid(rgb_features)
        else:
            rgb_features = {"out": torch.zeros((B * n_cam, 60, 32, 32), device="cuda"),}

        # Isolate level's visual features
        rgb_features = rgb_features['out']

        # Interpolate xy-depth to get the locations for this level
        feat_h, feat_w = rgb_features.shape[-2:]
        pcd = F.interpolate(pcd, (feat_h, feat_w), mode="nearest")
        rgb = F.interpolate(rgb, (feat_h, feat_w), mode="bilinear")

        # Merge different cameras for clouds, separate for rgb features
        pcd = einops.rearrange(
            pcd, "(bt fs ncam) c h w -> (bt fs) (ncam h w) c", ncam=n_cam, fs=fs
        )
        rgb_features = einops.rearrange(
            rgb_features, "(bt fs ncam) c h w -> (bt fs) (ncam h w) c", ncam=n_cam, fs=fs
        )
        rgb = einops.rearrange(
            rgb, "(bt fs ncam) c h w -> (bt fs) (ncam h w) c", ncam=n_cam, fs=fs
        )

        _, n_pts, d_feat = rgb_features.shape
        mask = torch.ones(pcd.shape[:-1], device=device)
        if self.do_crop:
            boundaries = einops.repeat(self.boundaries[data["task_id"]], 'b n d -> (b fs) n d', fs=fs)
            mask = crop_point_cloud(pcd, boundaries)


        if self.do_hand_crop:
            hand_mat_inv = einops.rearrange(data["obs"]["hand_mat_inv"], "b fs i j -> (b fs) i j")
            pcd_hand = batch_transform_point_cloud(pcd, hand_mat_inv)

            boundaries = einops.repeat(self.hand_frame_boundaries, 'n d -> (b fs) n d', b=B, fs=fs)
            hand_mask = crop_point_cloud(pcd_hand, boundaries)
            mask = torch.logical_and(mask, hand_mask)

        if self.hand_frame:
            hand_mat_inv = einops.rearrange(data["obs"]["hand_mat_inv"], "b fs i j -> (b fs) i j")
            pcd = batch_transform_point_cloud(pcd, hand_mat_inv)

        pcd = pcd * mask.unsqueeze(-1)
        rgb = rgb * mask.unsqueeze(-1)
        rgb_features = rgb_features * mask.unsqueeze(-1)

        if self.downsample_mode == "pos":
            downsampled_pcd, downsample_indices = torch3d_ops.sample_farthest_points(
                pcd, lengths=torch.sum(mask, dim=1), K=self.num_points
            )
            downsample_mask = ~(downsample_indices == -1)
            downsample_indices_clipped = torch.clamp(downsample_indices, min=0)

            downsampled_rgb = torch.gather(
                rgb, 1, einops.repeat(downsample_indices_clipped, "b n -> b n 3")
            )
            downsampled_feats = torch.gather(
                rgb_features,
                1,
                einops.repeat(downsample_indices_clipped, "b n -> b n k", k=rgb_features.shape[-1]),
            )
            num_points = self.num_points

        elif self.downsample_mode == "feat":
            # We found this hack (clipping off most of the features) to dramatically 
            # speed things up and not have too much effect on perf
            downsample_indices = dgl_geo.farthest_point_sampler(
                rgb_features[..., :30], self.num_points, 0
            )
            downsample_mask = ~(downsample_indices == -1)
            downsample_indices_clipped = torch.clamp(downsample_indices, min=0)

            downsampled_pcd = torch.gather(
                pcd, 1, einops.repeat(downsample_indices_clipped, "b n -> b n 3")
            )
            downsampled_rgb = torch.gather(
                rgb, 1, einops.repeat(downsample_indices_clipped, "b n -> b n 3")
            )
            downsampled_feats = torch.gather(
                rgb_features,
                1,
                einops.repeat(downsample_indices_clipped, "b n -> b n k", k=rgb_features.shape[-1]),
            )
            num_points = self.num_points

        elif self.downsample_mode == "none":
            downsampled_pcd = pcd.float()
            downsampled_feats = rgb_features
            downsampled_rgb = rgb
            num_points = pcd.shape[1]
            downsample_mask = torch.ones(B, num_points, device=device).bool()

        pcd_pos_emb = self.xyz_proj(downsampled_pcd)

        cat_cloud = []
        if self.do_pos:
            cat_cloud.append(pcd_pos_emb)
        if self.do_image:
            cat_cloud.append(downsampled_feats)
        if self.do_lang:
            lang_emb = self.get_task_emb(data)
            lang_emb = self.lang_proj(lang_emb)
            lang_emb = einops.repeat(lang_emb, "b d -> (b fs) n d", fs=fs, n=num_points)
            cat_cloud.append(lang_emb)
        if self.do_rgb:
            cat_cloud.append(downsampled_rgb)
        cat_cloud = torch.cat(cat_cloud, dim=-1)

        cat_cloud = einops.rearrange(cat_cloud, "(b fs) n d -> b fs n d", b=B)
        downsample_mask = einops.rearrange(downsample_mask, "(b fs) n -> b fs n", b=B)
        xyz = einops.rearrange(downsampled_pcd, "(b fs) n c -> b fs n c", b=B)
        out = self.pointcloud_extractor(
            xyz,
            cat_cloud,
            mask=downsample_mask,
            pc_pos=downsampled_pcd,
            pc_rgb=downsampled_rgb,
            vis_attn=False,
        )

        if self.lowdim_encoder is not None:
            lowdim = []
            for name, shape in self.shape_meta["observation"]["lowdim"].items():
                lowdim.append(obs_data[name])
            lowdim = torch.cat(lowdim, dim=-1)
            proprio_out = [self.lowdim_encoder(lowdim)]
        else:
            proprio_out = []

        return [out], proprio_out


def image_key_to_pointcloud_key(image_key):
    return f"{image_key[:-4]}_pointcloud_full"
