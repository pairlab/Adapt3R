import einops

import torch

from adapt3r.algos.utils.misc import weight_init
from adapt3r.algos.encoders.point_cloud_base import PointCloudBaseEncoder
import adapt3r.envs.utils as eu
import adapt3r.utils.point_cloud_utils as pcu


class DP3Encoder(PointCloudBaseEncoder):
    def __init__(
        self,
        pointcloud_extractor_factory,
        do_rgb=False,
        **kwargs,
    ):
        # Initialize parent class
        super().__init__(**kwargs)

        # Calculate point cloud input dimension
        pc_in = 3 + (3 if do_rgb else 0)
        
        # Initialize pointcloud extractor
        self.pointcloud_extractor = pointcloud_extractor_factory(in_shape=pc_in)
        self.pointcloud_extractor.apply(weight_init)
        
        # Set additional flags not in parent class
        self.do_rgb = do_rgb
        self.n_out_perception = 1
        self.d_out_perception = self.pointcloud_extractor.out_channels


    def forward(self, data, obs_key):
        obs_data = data[obs_key]
        
        pcd = []

        pcds = self._build_point_cloud(obs_data)
        for camera_name in eu.list_cameras(self.shape_meta):
            pcd.append(pcds[camera_name])

        pcd = torch.stack(pcd).to(dtype=torch.float32)

        n_cam, B, fs, _, _, _ = pcd.shape

        pcd = einops.rearrange(pcd, "ncam b fs h w c -> b fs (ncam h w) c")

        # Apply cropping
        hand_mat_inv = obs_data["hand_mat_inv"] if 'hand_mat_inv' in obs_data else None
        mask = self._crop_point_cloud(pcd, data["task_id"], hand_mat_inv)

        pcd = pcd * mask.unsqueeze(-1)

        if self.do_rgb:
            rgb = []
            for camera_name in eu.list_cameras(self.shape_meta):
                rgb.append(obs_data[eu.camera_name_to_image_key(camera_name)])
            rgb = torch.stack(rgb).to(dtype=torch.float32)
            rgb = einops.rearrange(rgb, "ncam b fs c h w -> b fs (ncam h w) c")
            rgb = rgb * mask.unsqueeze(-1)
        else:
            rgb = torch.zeros_like(pcd)

        pcd, _, rgb, mask = self._downsample_point_cloud(pcd, mask, rgb)

        cat_cloud = []
        cat_cloud.append(pcd)
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



