import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import FeaturePyramidNetwork

from adapt3r.algos.utils.misc import weight_init
from adapt3r.algos.encoders.rgb_modules import ResnetEncoder
from adapt3r.algos.encoders.point_cloud_base import PointCloudBaseEncoder
import adapt3r.envs.utils as eu
import adapt3r.utils.point_cloud_utils as pcu

from .clip import load_clip
from adapt3r.algos.utils.position_encodings import NeRFSinusoidalPosEmb
from .resnet import load_resnet18, load_resnet50


class Adapt3REncoder(PointCloudBaseEncoder):
    """Adapt3R Encoder for processing point clouds with RGB images and language embeddings.
    
    This encoder combines point cloud data with RGB images and optional language embeddings
    to create a rich representation of the scene. It supports multiple camera views
    
    Args:
        backbone_type (str): Type of backbone to use ('resnet50', 'resnet18', 'fusion', or 'clip')
        pointcloud_extractor_factory: Factory function to create point cloud extractor
        hidden_dim (int): Dimension of hidden features
        do_image (bool): Whether to process RGB images
        do_pos (bool): Whether to include position encoding
        do_lang (bool): Whether to include language embeddings
        do_rgb (bool): Whether to include raw RGB values
        hand_frame (bool): Whether to transform points to hand frame
        do_rot_aug (bool): Whether to apply rotation augmentation
        finetune (bool): Whether to fine-tune the backbone
        xyz_proj_type (str): Type of position encoding ('nerf' or 'none')
    """
    
    def __init__(
        self,
        backbone_type: str,
        pointcloud_extractor_factory,
        hidden_dim: int,
        do_image: bool = True,
        do_pos: bool = True,
        do_lang: bool = True,
        do_rgb: bool = False,
        hand_frame: bool = True,
        do_rot_aug: bool = False,
        finetune: bool = False,
        xyz_proj_type: str = "nerf",
        clip_model: str = "RN50",
        **kwargs,
    ) -> None:
        # Initialize parent class
        super().__init__(**kwargs)

        # Set additional flags not in parent class
        self.hand_frame = hand_frame
        self.do_rot_aug = do_rot_aug
        self.do_image = do_image
        self.do_pos = do_pos
        self.do_lang = do_lang
        self.do_rgb = do_rgb
        self.xyz_proj_type = xyz_proj_type
        self.hidden_dim = hidden_dim
        
        
        # Initialize pointcloud extractor
        self._init_pointcloud_extractor(pointcloud_extractor_factory)
        
        self.n_out_perception = self.frame_stack
        self.d_out_perception = self.pointcloud_extractor.out_channels

        # Setup backbone based on type
        self._init_backbone(backbone_type, finetune, clip_model)

        # Initialize feature pyramid network
        self._init_feature_pyramid()

        # Setup position encoding
        self._init_position_encoding()

        # Setup language projection
        self._init_language_projection()

    def _init_pointcloud_extractor(self, factory) -> None:
        """Initialize the point cloud extractor."""

        # Calculate point cloud input dimension
        pc_in = (
            (self.do_image + self.do_lang) * self.hidden_dim
            + self.do_pos * (3 if self.xyz_proj_type == "none" else self.hidden_dim)
            + (3 if self.do_rgb else 0)
        )
        self.pointcloud_extractor = factory(in_shape=pc_in)
        self.pointcloud_extractor.apply(weight_init)

    def _init_backbone(self, backbone_type: str, finetune: bool, clip_model: str) -> None:
        """Initialize the backbone network."""
        self.backbone_type = backbone_type
        if backbone_type == "resnet50":
            self.backbone, self.normalize = load_resnet50()
        elif backbone_type == "resnet18":
            self.backbone, self.normalize = load_resnet18()
        elif backbone_type == "fusion":
            self.backbone = ResnetEncoder(
                input_shape=tuple(self.shape_meta["observation"]["rgb"][next(iter(self.shape_meta["observation"]["rgb"]))]),
                language_fusion="film",
                language_dim=self.lang_embed_dim,
                do_projection=False,
                return_all_feats=True,
            )
            self.normalize = nn.Identity()
        elif backbone_type == "clip":
            self.backbone, self.normalize = load_clip(clip_model)
        else:
            raise NotImplementedError(f"backbone type {backbone_type} not supported")

        self.finetune = finetune
        if not finetune:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def _init_feature_pyramid(self) -> None:
        """Initialize the feature pyramid network."""
        self.feature_pyramid = FeaturePyramidNetwork([256], self.hidden_dim)

    def _init_position_encoding(self) -> None:
        """Initialize position encoding."""
        if self.xyz_proj_type == "nerf":
            self.xyz_proj = NeRFSinusoidalPosEmb(self.hidden_dim)
        elif self.xyz_proj_type == "none":
            self.xyz_proj = nn.Identity()
        else:
            raise ValueError(f"Unsupported xyz_proj_type: {self.xyz_proj_type}")

    def _init_language_projection(self) -> None:
        """Initialize language projection layer."""
        if self.lang_embed_dim != self.hidden_dim:
            self.lang_proj = nn.Linear(self.lang_embed_dim, self.hidden_dim)
        else:
            self.lang_proj = nn.Identity()

    def forward(self, data, obs_key):
        obs_data = data[obs_key]
        
        rgb = []
        pcd = []

        pcds = self._build_point_cloud(obs_data)
        for camera_name in eu.list_cameras(self.shape_meta):
            rgb.append(obs_data[eu.camera_name_to_image_key(camera_name)])
            pcd.append(pcds[camera_name])


        assert len(rgb) == len(pcd)

        rgb = torch.stack(rgb).to(dtype=torch.float32)
        pcd = torch.stack(pcd).to(dtype=torch.float32)

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

        rgb_features = rgb_features['out']

        # Interpolate xy-depth to get the locations for this level
        feat_h, feat_w = rgb_features.shape[-2:]
        pcd = F.interpolate(pcd, (feat_h, feat_w), mode="nearest")
        rgb = F.interpolate(rgb, (feat_h, feat_w), mode="bilinear")

        # Merge different cameras for clouds, separate for rgb features
        pcd = einops.rearrange(
            pcd, "(bt fs ncam) c h w -> bt fs (ncam h w) c", ncam=n_cam, fs=fs
        )
        rgb_features = einops.rearrange(
            rgb_features, "(bt fs ncam) c h w -> bt fs (ncam h w) c", ncam=n_cam, fs=fs
        )
        rgb = einops.rearrange(
            rgb, "(bt fs ncam) c h w -> bt fs (ncam h w) c", ncam=n_cam, fs=fs
        )

        # Apply cropping
        mask = self._crop_point_cloud(pcd=pcd, task_id=data["task_id"], hand_mat_inv=data["obs"]["hand_mat_inv"])

        pcd = pcd * mask.unsqueeze(-1)
        rgb = rgb * mask.unsqueeze(-1)
        rgb_features = rgb_features * mask.unsqueeze(-1)

        if self.hand_frame:
            pcd = pcu.batch_transform_point_cloud(pcd, data["obs"]["hand_mat_inv"])
        
        pcd, rgb_features, rgb, mask = self._downsample_point_cloud(pcd=pcd, rgb_features=rgb_features, rgb=rgb, mask=mask)

        pcd_pos_emb = self.xyz_proj(pcd)

        cat_cloud = []
        if self.do_pos:
            cat_cloud.append(pcd_pos_emb)
        if self.do_image:
            cat_cloud.append(rgb_features)
        if self.do_lang:
            lang_emb = self.get_task_emb(data)
            lang_emb = self.lang_proj(lang_emb)
            lang_emb = einops.repeat(lang_emb, "b d -> b fs n d", fs=fs, n=self.num_points)
            cat_cloud.append(lang_emb)
        if self.do_rgb:
            cat_cloud.append(rgb)
        cat_cloud = torch.cat(cat_cloud, dim=-1)

        out = self.pointcloud_extractor(
            cat_cloud,
            mask=mask,
        )
        out = list(einops.rearrange(out, "b fs d -> fs b d"))

        lowdim_out = self._encode_lowdim(obs_data)

        return out, lowdim_out



