import open3d as o3d
import torch
import einops


"""
Generates Open3D camera intrinsic matrix object from numpy camera intrinsic
    matrix and image width and height

@param cam_mat: 3x3 numpy array representing camera intrinsic matrix
@param width:   image width in pixels
@param height:  image height in pixels

@return t_mat:  4x4 transformation matrix as numpy array
"""
def cammat2o3d(cam_mat, width, height):
    cx = cam_mat[0,2]
    fx = cam_mat[0,0]
    cy = cam_mat[1,2]
    fy = cam_mat[1,1]

    return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

def depth2fgpcd_batch(depth, cam_params):
    # depth: (B, h, w)
    # fgpcd: (B, n, 3)
    # mask: (B, h, w)

    B, h, w = depth.shape

    fgpcd = torch.zeros((B, h * w, 3), device=depth.device, dtype=torch.float16)
    fx, fy, cx, cy = cam_params
    pos_x, pos_y = torch.meshgrid(
        torch.arange(w, device=depth.device), 
        torch.arange(h, device=depth.device), indexing='ij')
    
    pos_x = einops.rearrange(pos_x, 'w h -> h w')
    pos_y = einops.rearrange(pos_y, 'w h -> h w')
    
    pos_x = pos_x.repeat((B, 1, 1))
    pos_y = pos_y.repeat((B, 1, 1))

    fx = fx.reshape((-1, 1, 1))
    fy = fy.reshape((-1, 1, 1))
    cx = cx.reshape((-1, 1, 1))
    cy = cy.reshape((-1, 1, 1))

    fgpcd[:, :, 0] = einops.rearrange((pos_x - cx) * depth / fx, 'b h w -> b (h w)')
    fgpcd[:, :, 1] = einops.rearrange((pos_y - cy) * depth / fy, 'b h w -> b (h w)')
    fgpcd[:, :, 2] = einops.rearrange(depth, 'b h w -> b (h w)')
    return fgpcd


def lift_point_cloud_batch(depths, Ks, poses, max_depth=1.5, maintain_dims=False) -> torch.Tensor:
    # depths: [B, H, W] numpy array in meters
    # Ks: [B, 3, 3] numpy array
    # poses: [B, 4, 4] numpy array
    # masks: [B, H, W] numpy array in bool
    B, H, W = depths.shape

    cam_param = [Ks[:, 0, 0], Ks[:, 1, 1], Ks[:, 0, 2], Ks[:, 1, 2]]  # fx, fy, cx, cy
    pcd = depth2fgpcd_batch(depths, cam_param)
    trans_pcd = batch_transform_point_cloud(pcd, poses)

    if maintain_dims:
        trans_pcd = einops.rearrange(trans_pcd, 'b (h w) c -> b h w c', h=H)

    return trans_pcd

def batch_transform_point_cloud(pcd, transform):
    B, N, _ = pcd.shape

    pcd_1 = torch.cat([pcd, torch.ones((B, N, 1), device=pcd.device, dtype=pcd.dtype)], dim=-1)
    transform = transform.to(dtype=pcd.dtype)

    trans_pcd_1 = torch.einsum('bnd,bid->bni', pcd_1, transform)
    trans_pcd = trans_pcd_1[:, :, :-1]
    return trans_pcd


def mask_sort_point_cloud(pcd, mask):
    """
    Rearranges the point cloud so that the non-masked out points are at the beginning
    and the rest is zero
    """
    B, N, D = pcd.shape
    indices = torch.masked_fill(
        input=torch.cumsum(mask.int(), dim=1), 
        mask=~mask, 
        value=0)
    indices_repeat = einops.repeat(indices, "b n -> b n k", k=D)
    masked_pcd = torch.scatter(
        input=torch.zeros(
            (B, N + 1, 3), 
            device=pcd.device, 
            dtype=pcd.dtype),
        index=indices_repeat,
        src=pcd,
        dim=1,
    )[:, 1:]
    return masked_pcd
    

def crop_point_cloud(pcd, boundaries):
    
    boundaries = einops.rearrange(boundaries, 'b n d -> b n 1 d')
    boundaries_low = boundaries[:, 0]
    boundaries_high = boundaries[:, 1]

    above_lower = torch.all(pcd > boundaries_low, dim=-1)
    below_upper = torch.all(pcd < boundaries_high, dim=-1)
    mask = torch.logical_and(above_lower, below_upper)
    return mask


def show_point_cloud(pcd, pcd_colors=None, vectors=None):
    if type(pcd) != o3d.geometry.PointCloud:
        if type(pcd) == torch.Tensor:
            pcd = pcd.detach().cpu().numpy()
        
        if type(pcd_colors) == torch.Tensor:
            pcd_colors = pcd_colors.detach().cpu().numpy()
        elif pcd_colors is None and pcd.shape[1] == 6:
            pcd_colors = pcd[:, 3:]
            pcd = pcd[:, :3]
        
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(pcd)
        cloud.colors = o3d.utility.Vector3dVector(pcd_colors)

    else:
        cloud = pcd

    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    o3d.visualization.draw_geometries([cloud, origin])