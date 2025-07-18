import torch
import einops
import adapt3r.utils.pytorch3d_transforms as p3d
import matplotlib.pyplot as plt

import open3d as o3d
import numpy as np


# """
# Generates Open3D camera intrinsic matrix object from numpy camera intrinsic
#     matrix and image width and height

# @param cam_mat: 3x3 numpy array representing camera intrinsic matrix
# @param width:   image width in pixels
# @param height:  image height in pixels

# @return t_mat:  4x4 transformation matrix as numpy array
# """
# def cammat2o3d(cam_mat, width, height):
#     cx = cam_mat[0,2]
#     fx = cam_mat[0,0]
#     cy = cam_mat[1,2]
#     fy = cam_mat[1,1]

#     return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

def depth2fgpcd_batch(depth, cam_params, keepdims=False):
    # depth: (B, ncam, h, w)
    # fgpcd: (B, ncam, n, 3)
    # mask: (B, ncam, h, w)

    B, ncam, h, w = depth.shape

    fgpcd = torch.zeros((B, ncam * h * w, 3), device=depth.device, dtype=torch.float16)
    # fx, fy, cx, cy = cam_params
    cx = cam_params[..., 0, 2]
    fx = cam_params[..., 0, 0]
    cy = cam_params[..., 1, 2]
    fy = cam_params[..., 1, 1]
    pos_x, pos_y = torch.meshgrid(
        torch.arange(w, device=depth.device), 
        torch.arange(h, device=depth.device), indexing='ij')
    
    pos_x = einops.rearrange(pos_x, 'w h -> h w')
    pos_y = einops.rearrange(pos_y, 'w h -> h w')
    
    pos_x = pos_x.repeat((B, ncam, 1, 1))
    pos_y = pos_y.repeat((B, ncam, 1, 1))

    fx = fx.reshape((B, ncam, 1, 1))
    fy = fy.reshape((B, ncam, 1, 1))
    cx = cx.reshape((B, ncam, 1, 1))
    cy = cy.reshape((B, ncam, 1, 1))

    fgpcd[:, :, 0] = einops.rearrange((pos_x - cx) * depth / fx, 'b ncam h w -> b (ncam h w)')
    fgpcd[:, :, 1] = einops.rearrange((pos_y - cy) * depth / fy, 'b ncam h w -> b (ncam h w)')
    fgpcd[:, :, 2] = einops.rearrange(depth, 'b ncam h w -> b (ncam h w)')
    
    if keepdims:
        fgpcd = einops.rearrange(fgpcd, 'b (ncam h w) c -> b ncam h w c', ncam=ncam, h=h)
    return fgpcd


def lift_point_cloud_batch(depths, intrinsics, extrinsics, keepdims=False) -> torch.Tensor:
    # depths: [B, ncam, H, W] numpy array in meters
    # Ks: [B, ncam, 3, 3] numpy array
    # poses: [B, ncam, 4, 4] numpy array
    # masks: [B, ncam, H, W] numpy array in bool
    B, ncam, H, W = depths.shape

    pcd = depth2fgpcd_batch(depths, intrinsics, keepdims=False)
    pcd = einops.rearrange(pcd, 'b (ncam hw) c -> b ncam hw c', ncam=ncam)

    trans_pcd = batch_transform_point_cloud(pcd, extrinsics)

    if keepdims:
        trans_pcd = einops.rearrange(trans_pcd, 'b ncam (h w) c -> b ncam h w c', ncam=ncam, h=H)
    else:
        trans_pcd = einops.rearrange(trans_pcd, 'b ncam (h w) c -> b (ncam h w) c')
        

    return trans_pcd

def batch_transform_point_cloud(pcd, transform):
    pcd_homo = torch.cat([pcd, torch.ones((*pcd.shape[:-1], 1), 
                                          device=pcd.device, 
                                          dtype=pcd.dtype)], 
                         dim=-1)
    transform = transform.to(dtype=pcd.dtype)

    trans_pcd_homo = torch.einsum('...nd,...id->...ni', pcd_homo, transform)
    trans_pcd = trans_pcd_homo[..., :-1]
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
            (B, N + 1, D), 
            device=pcd.device, 
            dtype=pcd.dtype),
        index=indices_repeat,
        src=pcd,
        dim=1,
    )[:, 1:]
    return masked_pcd

def matrix_to_pos_rot_matrix(matrix):
    rot_mat = matrix[..., :3, :3]
    pos = matrix[..., :3, 3]
    return pos, rot_mat

def matrix_to_pos_6d(matrix):
    pos, rot_mat = matrix_to_pos_rot_matrix(matrix)
    rot_6d = p3d.matrix_to_rotation_6d(rot_mat)
    return pos, rot_6d


def pos_euler_to_matrix(pos_euler):
    pos, euler_angles = torch.split(pos_euler, 3, dim=-1)
    # This is necessary because DROID are saved as extrinsic rotations but 
    # pytorch3d only supports intrinsic
    euler_angles = torch.flip(euler_angles, dims=(-1,))
    rot_mat = p3d.euler_angles_to_matrix(euler_angles, convention="ZYX")
    mat = pos_rot_mat_to_mat(pos, rot_mat)
    return mat


def pos_rot_mat_to_mat(pos, rot_mat):
    batch_dims = pos.shape[:-1]
    mat = torch.zeros((*batch_dims, 4, 4), device=pos.device)
    mat[..., :3,:3] = rot_mat
    mat[..., :3, 3] = pos
    mat[...,  3, 3] = 1.0
    # invert the matrix to represent standard definition of extrinsics: from world to cam
    # mat = torch.linalg.inv(mat)
    return mat
    
    

def crop_point_cloud(pcd, boundaries):
    
    boundaries = einops.rearrange(boundaries, 'b n d -> b n 1 d')
    boundaries_low = boundaries[:, 0]
    boundaries_high = boundaries[:, 1]

    above_lower = torch.all(pcd > boundaries_low, dim=-1)
    below_upper = torch.all(pcd < boundaries_high, dim=-1)
    mask = torch.logical_and(above_lower, below_upper)
    return mask


def show_point_cloud(
        pcd, 
        pcd_colors=None, 
        vectors=None, 
        extra_points=None,
        show_axes=False,
        extra_frames=None,
        ):
    if type(pcd) != o3d.geometry.PointCloud:
        if type(pcd) == torch.Tensor:
            pcd = pcd.to(dtype=torch.float32).detach().cpu().numpy()
        
        if type(pcd_colors) == torch.Tensor:
            pcd_colors = pcd_colors.detach().cpu().numpy()
        elif pcd_colors is None and pcd.shape[1] == 6:
            pcd_colors = pcd[:, 3:]
            pcd = pcd[:, :3]
        
        if extra_points is not None:
            if type(extra_points) == torch.Tensor:
                extra_points = extra_points.detach().cpu().numpy()
            pcd = np.concatenate([pcd, extra_points])
            pcd_colors = np.concatenate([pcd_colors, [[1, 0, 0]] * len(extra_points)])
        
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pcd))
        if pcd_colors is not None:
            cloud.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(pcd_colors))

    else:
        cloud = pcd

    geometries = [cloud]

    # Add coordinate frames if specified
    if extra_frames is not None:
        if isinstance(extra_frames, torch.Tensor):
            extra_frames = extra_frames.detach().cpu().numpy()
        if extra_frames.ndim == 2:
            extra_frames = extra_frames[None]  # Add batch dimension if single frame
        
        for i, frame in enumerate(extra_frames):
            # Create coordinate frame
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.1,  # Size of the coordinate frame
                origin=frame[:3, 3]  # Translation from the transformation matrix
            )
            
            # Extract rotation matrix and apply it
            rotation = frame[:3, :3]
            coord_frame.rotate(rotation, center=frame[:3, 3])
            
            geometries.append(coord_frame)

    if show_axes:
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        geometries.append(origin)

    o3d.visualization.draw_geometries(geometries)

def show_point_cloud_viser(pcd, pcd_colors=None, vectors=None, extra_points=None, port=8080):
    """
    Display a point cloud using viser in 3D, with optional colors and direction vectors.
    
    Args:
        pcd: [N, 3] or [N, 6] torch.Tensor or np.ndarray of point cloud coordinates (with optional RGB)
        pcd_colors: [N, 3] torch.Tensor or np.ndarray of RGB values, optional
        vectors: [N, 3] torch.Tensor or np.ndarray of vectors to draw as arrows, optional
        extra_points: [M, 3] additional points to visualize in red, optional
        port: Port number for viser server, defaults to 8080
    """
    import viser
    import viser.transforms as tf
    
    # Convert inputs to numpy arrays
    if isinstance(pcd, torch.Tensor):
        pcd = pcd.detach().cpu().numpy()
    
    if isinstance(pcd_colors, torch.Tensor):
        pcd_colors = pcd_colors.detach().cpu().numpy()
    elif pcd_colors is None and pcd.shape[1] == 6:
        pcd_colors = pcd[:, 3:]
        pcd = pcd[:, :3]
    
    if extra_points is not None:
        if isinstance(extra_points, torch.Tensor):
            extra_points = extra_points.detach().cpu().numpy()
        pcd = np.concatenate([pcd, extra_points])
        if pcd_colors is not None:
            pcd_colors = np.concatenate([pcd_colors, np.tile([1,0,0], (len(extra_points), 1))])
    
    if pcd_colors is None:
        pcd_colors = np.ones_like(pcd) * 0.5

    server = viser.ViserServer(port=port)
    
    # Add point cloud
    server.add_point_cloud(
        name="point_cloud",
        points=pcd,
        colors=pcd_colors,
        point_size=0.01
    )
    
    # Add coordinate axes
    length = 0.1
    origin = np.zeros(3)
    axes = {
        "x": ([length,0,0], [1,0,0]),
        "y": ([0,length,0], [0,1,0]), 
        "z": ([0,0,length], [0,0,1])
    }
    
    print(f"Viser visualization server running at: http://localhost:{port}")
    return server


def show_point_cloud_plt(pcd, pcd_colors=None, vectors=None):
    """
    Display a point cloud using matplotlib in 3D, with optional colors and direction vectors.

    Args:
        pcd: [N, 3] or [N, 6] torch.Tensor or np.ndarray of point cloud coordinates (with optional RGB)
        pcd_colors: [N, 3] torch.Tensor or np.ndarray of RGB values, optional
        vectors: [N, 3] torch.Tensor or np.ndarray of vectors to draw as arrows, optional
    """
    if isinstance(pcd, torch.Tensor):
        pcd = pcd.detach().cpu().numpy()

    if pcd_colors is None and pcd.shape[1] == 6:
        pcd_colors = pcd[:, 3:]
        pcd = pcd[:, :3]

    if isinstance(pcd_colors, torch.Tensor):
        pcd_colors = pcd_colors.detach().cpu().numpy()

    if pcd_colors is None:
        pcd_colors = np.full_like(pcd, 0.5)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')

    ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], c=pcd_colors, s=5)

    if vectors is not None:
        if isinstance(vectors, torch.Tensor):
            vectors = vectors.detach().cpu().numpy()
        ax.quiver(
            pcd[:, 0], pcd[:, 1], pcd[:, 2],
            vectors[:, 0], vectors[:, 1], vectors[:, 2],
            length=0.05, normalize=True, color='red'
        )

    # Draw coordinate axes at origin
    origin = np.zeros((3,))
    axis = np.eye(3) * 0.1
    ax.quiver(*origin, *axis[0], color='r', linewidth=2, label='X-axis')
    ax.quiver(*origin, *axis[1], color='g', linewidth=2, label='Y-axis')
    ax.quiver(*origin, *axis[2], color='b', linewidth=2, label='Z-axis')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 1])
    ax.legend()
    plt.tight_layout()
    plt.show()