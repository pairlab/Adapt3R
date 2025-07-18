import einops

import dgl.geometry as dgl_geo
import torch

from adapt3r.algos.utils.misc import weight_init
from adapt3r.algos.encoders.base import BaseEncoder
import adapt3r.envs.utils as eu
import adapt3r.utils.point_cloud_utils as pcu
from adapt3r.utils.point_cloud_utils import lift_point_cloud_batch


class PointCloudBaseEncoder(BaseEncoder):
    def __init__(
        self,
        num_points=None,
        lowdim_encoder_factory=None,
        do_crop=True,
        do_hand_crop=True,
        tight_crop=True,
        task_suite_name=None,  # I don't like having this here but for now it's necessary to get boundary info
        task_benchmark_name=None,
        downsample_mode="pos",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_points = num_points
        self.shape_meta = self.shape_meta
        self.do_crop = do_crop
        self.do_hand_crop = do_hand_crop
        self.downsample_mode = downsample_mode
        do_lowdim = lowdim_encoder_factory is not None

        if do_lowdim and len(self.shape_meta["observation"]["lowdim"]) > 0:
            d_lowdim = 0
            for name, shape in self.shape_meta["observation"]["lowdim"].items():
                d_lowdim += shape
            encoder = lowdim_encoder_factory(d_lowdim)
            encoder.apply(weight_init)
            self.lowdim_encoder = encoder
            self.n_out_lowdim += 1
            self.d_out_lowdim = encoder.out_channels
        else:
            self.lowdim_encoder = None

        if task_suite_name == "libero":
            import adapt3r.envs.libero.utils as lu
            boundaries = lu.get_boundaries(benchmark_name=task_benchmark_name, tight=tight_crop)
            boundaries = torch.tensor(boundaries, dtype=torch.float32)
        elif task_suite_name == "mimicgen":
            boundaries = torch.tensor(((-1, -1, 0), (1, 1, 2)))
            boundaries = einops.repeat(boundaries, "i j -> 1 i j")
        else:
            boundaries = torch.tensor(((-1, -1, -1), (1, 1, 1)))
            boundaries = einops.repeat(boundaries, "i j -> 1 i j")
        self.register_buffer("boundaries", torch.tensor(boundaries, dtype=torch.float32))
        self.boundaries: torch.Tensor = self.boundaries # To help with type checking
        
        hand_frame_boundaries = torch.tensor(((-1, -1, 0), (1, 1, 1)), dtype=torch.float32)
        self.register_buffer('hand_frame_boundaries', hand_frame_boundaries)
        self.hand_frame_boundaries: torch.Tensor = self.hand_frame_boundaries # To help with type checking
        

    def forward(self, *args, **kwargs):
        raise NotImplementedError("PointCloudBaseEncoder is an abstract class")

    def _downsample_point_cloud(self, pcd, mask=None, rgb=None, rgb_features=None, num_points=None):
        if self.downsample_mode == "none":
            # No downsampling - return inputs as is
            return pcd, rgb_features, rgb, mask
        
        B, fs, N, D = pcd.shape
        device = pcd.device
        # Reshape inputs for batch processing
        pcd = einops.rearrange(pcd, "b fs n d -> (b fs) n d")
        if mask is not None:
            mask = einops.rearrange(mask, "b fs n -> (b fs) n")
        if rgb is not None:
            rgb = einops.rearrange(rgb, "b fs n d -> (b fs) n d")
        if rgb_features is not None:
            rgb_features = einops.rearrange(rgb_features, "b fs n d -> (b fs) n d")

        num_points = num_points or self.num_points

        # Get indices for sampling points
        if self.downsample_mode == "pos":
            # Sample points with FPS based on point cloud positions
            downsample_indices = dgl_geo.farthest_point_sampler(pcd, num_points, 0)
            downsample_indices_clipped = torch.clamp(downsample_indices, min=0)

        elif self.downsample_mode == "feat":
            assert rgb_features is not None
            # Sample points with FPS if features available, otherwise uniform
            downsample_indices = dgl_geo.farthest_point_sampler(
                rgb_features[..., :30], num_points, 0
            )
            downsample_indices_clipped = downsample_indices

        # Gather points and features using indices
        downsampled_pcd = torch.gather(
            pcd, 1, einops.repeat(downsample_indices_clipped, "b n -> b n d", d=pcd.shape[-1])
        )
        downsampled_pcd = einops.rearrange(downsampled_pcd, "(b fs) n d -> b fs n d", b=B)

        if mask is not None:
            downsample_mask = torch.gather(mask, 1, downsample_indices_clipped)
            downsample_mask = einops.rearrange(downsample_mask, "(b fs) n -> b fs n", b=B)
        else:
            downsample_mask = None
        
        if rgb is not None:
            downsampled_rgb = torch.gather(
                rgb, 1, einops.repeat(downsample_indices_clipped, "b n -> b n 3")
            )
            downsampled_rgb = einops.rearrange(downsampled_rgb, "(b fs) n d -> b fs n d", b=B)
        else:
            downsampled_rgb = None
            
        if rgb_features is not None:
            downsampled_feats = torch.gather(
                rgb_features,
                1,
                einops.repeat(downsample_indices_clipped, "b n -> b n k", k=rgb_features.shape[-1]),
            )
            downsampled_feats = einops.rearrange(downsampled_feats, "(b fs) n d -> b fs n d", b=B)
        else:
            downsampled_feats = None


        return downsampled_pcd, downsampled_feats, downsampled_rgb, downsample_mask

    def _crop_point_cloud(self, pcd, task_id=None, hand_mat_inv=None, boundaries=None):
        """
        Apply cropping to point cloud data based on boundaries and hand frame.
        
        Args:
            pcd: Point cloud data
            rgb_features: RGB features
            rgb: RGB data
            data: Input data dictionary
            device: Target device
            
        Returns:
            Tuple of (cropped_pcd, cropped_rgb_features, cropped_rgb, mask)
        """
        device = pcd.device
        B, fs, _, _ = pcd.shape
        pcd = einops.rearrange(pcd, "b fs n d -> (b fs) n d")
        mask = torch.ones(pcd.shape[:-1], device=device).bool()
        
        if self.do_crop:
            if boundaries is None:
                boundaries = self.boundaries[task_id]
            boundaries = einops.repeat(boundaries, 'b n d -> (b fs) n d', fs=fs, b=B)
            mask = pcu.crop_point_cloud(pcd, boundaries)

        if self.do_hand_crop:
            assert hand_mat_inv is not None

            hand_mat_invs = hand_mat_inv if type(hand_mat_inv) == list else [hand_mat_inv]
            
            for hand_mat_inv in hand_mat_invs:
                hand_mat_inv = hand_mat_inv[:, -1] # Take last hand mat along frame stack dimension
                pcd_hand = pcu.batch_transform_point_cloud(pcd, hand_mat_inv)

                boundaries = einops.repeat(self.hand_frame_boundaries, 'n d -> (b fs) n d', b=B, fs=fs)
                hand_mask = pcu.crop_point_cloud(pcd_hand, boundaries)
                mask = torch.logical_and(mask, hand_mask)

        mask = einops.rearrange(mask, "(b fs) n -> b fs n", b=B)
        
        return mask

    def _encode_lowdim(self, obs_data):
        if self.lowdim_encoder is None:
            return []
        
        lowdim = []
        for name, shape in self.shape_meta["observation"]["lowdim"].items():
            lowdim.append(obs_data[name])
        lowdim = torch.cat(lowdim, dim=-1)
        encodings = self.lowdim_encoder(lowdim)
        encodings = list(einops.rearrange(encodings, "b fs d -> fs b d"))
        return encodings

    def _build_point_cloud(self, obs_data):
        out = {}
        for camera_name in eu.list_cameras(self.shape_meta):
            intrinsic_key = eu.camera_name_to_intrinsic_key(camera_name)
            extrinsic_key = eu.camera_name_to_extrinsic_key(camera_name)
            depth_key = eu.camera_name_to_depth_key(camera_name)
            
            depths = obs_data[depth_key].squeeze(2).to(torch.float32)
            intrinsics = obs_data[intrinsic_key]
            extrinsics = obs_data[extrinsic_key]
            
            B, T, H, W = depths.shape
            depths = depths.reshape(B, 1, H, W)
            intrinsics = intrinsics.reshape(B, 1, 3, 3)
            extrinsics = extrinsics.reshape(B, 1, 4, 4)
            
            pcd = lift_point_cloud_batch(
                depths,        # [B, 1, H, W]
                intrinsics,    # [B, 1, 3, 3]
                extrinsics,    # [B, 1, 4, 4]
                keepdims=True
            )
            out[camera_name] = pcd
        
        return out


