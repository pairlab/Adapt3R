import dgl.geometry as dgl_geo
import einops
# import pytorch3d.ops as torch3d_ops
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import FeaturePyramidNetwork

from .clip import load_clip
from .layers import FFWRelativeCrossAttentionModule, ParallelAttention
from .position_encodings import RotaryPositionEncoding3D
from .resnet import load_resnet18, load_resnet50
from adapt3r.algos.encoders.point_cloud_base import PointCloudBaseEncoder


class Encoder(PointCloudBaseEncoder):

    def __init__(
        self,
        backbone="clip",
        image_size=(256, 256),
        embedding_dim=60,
        num_sampling_level=3,
        num_attn_heads=8,
        num_vis_ins_attn_layers=2,
        fps_subsampling_factor=5,
        beefy=False,
        **kwargs,
    ):
        super().__init__(
            downsample_mode='feat',
            do_hand_crop=False,
            **kwargs,
        )
        assert backbone in ["resnet50", "resnet18", "clip"]
        assert image_size in [(128, 128), (256, 256)]
        assert num_sampling_level in [1, 2, 3, 4]

        self.image_size = image_size
        self.num_sampling_level = num_sampling_level
        self.fps_subsampling_factor = fps_subsampling_factor

        # Initialize backbone
        if backbone == "resnet50":
            self.backbone, self.normalize = load_resnet50()
        elif backbone == "resnet18":
            self.backbone, self.normalize = load_resnet18()
        elif backbone == "clip":
            self.backbone, self.normalize = load_clip()
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Initialize feature pyramid network
        self.feature_pyramid = FeaturePyramidNetwork([256, 512], embedding_dim)

        # Set feature map pyramid based on image size
        if self.image_size == (128, 128):
            self.feature_map_pyramid = ["res3", "res1", "res1", "res1"] if not beefy else ["res2", "res1", "res1", "res1"]
            self.downscaling_factor_pyramid = [4, 2, 2, 2]
        elif self.image_size == (256, 256):
            self.feature_map_pyramid = ["res3", "res1", "res1", "res1"]
            self.downscaling_factor_pyramid = [8, 2, 2, 2]

        # Initialize position encoding and attention modules
        self.relative_pe_layer = RotaryPositionEncoding3D(embedding_dim)
        self.curr_gripper_embed = nn.Embedding(self.frame_stack, embedding_dim)
        self.goal_gripper_embed = nn.Embedding(1, embedding_dim)
        self.instruction_encoder = nn.Linear(512, embedding_dim)
        
        # Initialize vision-language attention
        layer = ParallelAttention(
            num_layers=num_vis_ins_attn_layers,
            d_model=embedding_dim,
            n_heads=num_attn_heads,
            self_attention1=False,
            self_attention2=False,
            cross_attention1=True,
            cross_attention2=False,
        )
        self.vl_attention = nn.ModuleList([layer for _ in range(1)])
        
        # Initialize gripper context head
        self.gripper_context_head = FFWRelativeCrossAttentionModule(
            embedding_dim, num_attn_heads, num_layers=3, use_adaln=False
        )

    def encode_images(self, obs):
        """
        Compute visual features and point cloud embeddings at different scales.

        Args:
            obs: Full observation dictionary containing:
                - rgb: (B, ncam, 3, H, W), pixel intensities
                - depth: (B, ncam, 1, H, W), depth values
                - camera parameters and other metadata

        Returns:
            rgb_feats_pyramid: List of RGB features at different scales
            pcd_pyramid: List of point cloud data at different scales
        """
        # Extract and stack RGB and depth observations
        rgb_obs = []
        depth_obs = []
        for key in obs.keys():
            if "rgb" in key or "image" in key:
                rgb_obs.append(obs[key])
            elif "depth" in key:
                depth_obs.append(obs[key])

        rgb = torch.stack(rgb_obs, dim=1)[:, :, 0]
        depth = torch.stack(depth_obs, dim=1)[:, :, 0]
        num_cameras = rgb.shape[1]

        # Process RGB through backbone
        rgb = einops.rearrange(rgb, "bt ncam c h w -> (bt ncam) c h w")
        rgb = rgb / 255.
        rgb = self.normalize(rgb)
        rgb_features = self.backbone(rgb)
        rgb_features = self.feature_pyramid(rgb_features)

        # Build and process point cloud
        pcd = self._build_point_cloud(obs)
        pcd = torch.stack([pcd[cam] for cam in pcd.keys()], dim=2)  # [B, fs, ncam, H, W, 3]
        pcd = einops.rearrange(pcd, "b fs ncam h w c -> (b fs ncam) c h w")

        # Process features at different scales
        rgb_feats_pyramid = []
        pcd_pyramid = []
        for i in range(self.num_sampling_level):
            # Get features for current scale
            rgb_features_i = rgb_features[self.feature_map_pyramid[i]]
            feat_h, feat_w = rgb_features_i.shape[-2:]
            
            # Interpolate point cloud to match feature dimensions
            pcd_i = F.interpolate(pcd, (feat_h, feat_w), mode="bilinear")
            
            # Reshape for multi-camera setup
            h, w = pcd_i.shape[-2:]
            pcd_i = einops.rearrange(pcd_i, "(bt ncam) c h w -> bt (ncam h w) c", ncam=num_cameras)
            rgb_features_i = einops.rearrange(
                rgb_features_i, "(bt ncam) c h w -> bt ncam c h w", ncam=num_cameras
            )

            rgb_feats_pyramid.append(rgb_features_i)
            pcd_pyramid.append(pcd_i)

        return rgb_feats_pyramid, pcd_pyramid

    def crop_point_cloud(self, pcd):
        """
        Crop point cloud to valid region.

        Args:
            pcd: Point cloud data of shape (B, N, 3)
            
        Returns:
            mask: Boolean mask of shape (B, N) indicating valid points
        """
        boundaries = torch.tensor(((-1, -1, -1), (1, 1, 1)), dtype=torch.float32, device=pcd.device)
        boundaries = einops.repeat(boundaries, "i j -> b i j", b=pcd.shape[0])
        pcd = einops.rearrange(pcd, "b n d -> b 1 n d")
        return self._crop_point_cloud(pcd, boundaries=boundaries).squeeze(1)

    def run_fps(self, pcd, rgb_features):
        """
        Run farthest point sampling on point cloud and features.

        Args:
            pcd: Point cloud data of shape (B, N, F, 2)
            rgb_features: RGB features of shape (B, N, F)
            
        Returns:
            Tuple of (downsampled_pcd, downsampled_features)
        """
        _, npts, _ = rgb_features.shape
        num_points = max(npts // self.fps_subsampling_factor, 1)
        
        # Reshape for downsampling
        pcd = einops.rearrange(pcd, "b n d i -> b 1 n (d i)")
        rgb_features = einops.rearrange(rgb_features, "b n d -> b 1 n d")
        
        # Apply downsampling
        downsampled_pcd, downsampled_features, _, _ = self._downsample_point_cloud(
            pcd=pcd,
            rgb_features=rgb_features,
            num_points=num_points,
        )
        
        # Reshape back to original format
        downsampled_pcd = einops.rearrange(downsampled_pcd, "b 1 n (d i) -> b n d i", i=2)
        downsampled_features = einops.rearrange(downsampled_features, "b 1 n d -> n b d")
        
        return downsampled_pcd, downsampled_features

    def encode_instruction(self, instruction):
        """
        Encode instruction text into features.

        Args:
            instruction: Instruction embeddings of shape (B, max_instruction_length, 512)

        Returns:
            Tuple of (instruction features, dummy positional embeddings)
        """
        instr_feats = self.instruction_encoder(instruction)
        instr_dummy_pos = torch.zeros(
            len(instruction), instr_feats.shape[1], 3, device=instruction.device
        )
        instr_dummy_pos = self.relative_pe_layer(instr_dummy_pos)
        return instr_feats, instr_dummy_pos

    def encode_curr_gripper(self, curr_gripper, context_feats, context):
        """
        Encode current gripper state.

        Args:
            curr_gripper: Current gripper state of shape (B, nhist, 3+)
            context_feats: Context features
            context: Context point cloud

        Returns:
            Tuple of (gripper features, gripper positional embeddings)
        """
        if curr_gripper is not None:
            return self._encode_gripper(curr_gripper, self.curr_gripper_embed, context_feats, context)
        return None, None

    def encode_goal_gripper(self, goal_gripper, context_feats, context):
        """
        Encode goal gripper state.

        Args:
            goal_gripper: Goal gripper state of shape (B, 3+)
            context_feats: Context features
            context: Context point cloud

        Returns:
            Tuple of (gripper features, gripper positional embeddings)
        """
        goal_gripper_feats, goal_gripper_pos = self._encode_gripper(
            goal_gripper[:, None], self.goal_gripper_embed, context_feats, context
        )
        return goal_gripper_feats, goal_gripper_pos

    def _encode_gripper(self, gripper, gripper_embed, context_feats, context):
        """
        Encode gripper state with context.

        Args:
            gripper: Gripper state of shape (B, npt, 3+)
            gripper_embed: Gripper embedding layer
            context_feats: Context features
            context: Context point cloud

        Returns:
            Tuple of (gripper features, gripper positional embeddings)
        """
        gripper_feats = gripper_embed.weight.unsqueeze(0).repeat(len(gripper), 1, 1)
        gripper_pos = self.relative_pe_layer(gripper[..., :3])
        context_pos = self.relative_pe_layer(context)

        gripper_feats = einops.rearrange(gripper_feats, "b npt c -> npt b c")
        context_feats = einops.rearrange(context_feats, "b npt c -> npt b c")
        gripper_feats = self.gripper_context_head(
            query=gripper_feats, value=context_feats, query_pos=gripper_pos, value_pos=context_pos
        )[-1]
        gripper_feats = einops.rearrange(gripper_feats, "nhist b c -> b nhist c")

        return gripper_feats, gripper_pos

    def vision_language_attention(self, feats, instr_feats, feats_mask=None, instr_mask=None):
        """
        Apply vision-language attention.

        Args:
            feats: Visual features
            instr_feats: Instruction features
            feats_mask: Optional mask for visual features
            instr_mask: Optional mask for instruction features

        Returns:
            Attended visual features
        """
        feats, _ = self.vl_attention[0](
            seq1=feats,
            seq1_key_padding_mask=feats_mask,
            seq2=instr_feats,
            seq2_key_padding_mask=instr_mask,
            seq1_pos=None,
            seq2_pos=None,
            seq1_sem_pos=None,
            seq2_sem_pos=None,
        )
        return feats
