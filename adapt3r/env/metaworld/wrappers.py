import numpy as np
import gymnasium
import mujoco
import torch
import open3d as o3d
import robosuite.utils.transform_utils as T
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE

from adapt3r.env.utils.frame_stack import FrameStackObservationFixed
from adapt3r.env.utils.efficient_offscreen_viewer import EfficientOffScreenViewer
from adapt3r.utils.point_cloud_utils import cammat2o3d
from adapt3r.utils.geometry import posRotMat2Mat, quat2mat

class MetaWorldFrameStack(FrameStackObservationFixed):
    def __init__(self, env_name, env_factory, num_stack):
        self.num_stack = num_stack
        env = env_factory(env_name)
        super().__init__(env, num_stack)

    def set_task(self, task):
        self.env.set_task(task)


class MetaWorldWrapper(gymnasium.Wrapper):
    def __init__(self, 
                 env_name: str,
                 shape_meta: dict,
                 img_height: int = 128,
                 img_width: int = 128,
                 num_points: int = 512,
                 cameras: tuple = ('corner2',),
                 camera_pose_variations: bool = False,
                 boundaries = None,
                 env_kwargs: dict = None):
        
        # Initialize environment
        if env_kwargs is None:
            env_kwargs = {}
        env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[f'{env_name}-goal-observable'](**env_kwargs)
        env._freeze_rand_vec = False
        super().__init__(env)

        # I don't know why
        self.env.model.cam_pos[2] = [0.75, 0.075, 0.7]

        self.img_width = img_width
        self.img_height = img_height
        self.num_points = num_points
        self.boundaries = boundaries
        self.cameras = cameras

        # Extract observation types from shape_meta
        self.obs_meta = shape_meta['observation']
        self.rgb_outputs = list(self.obs_meta['rgb'])
        self.depth_outputs = list(self.obs_meta['depth'])
        self.pointcloud_outputs = list(self.obs_meta['pointcloud'])
        self.lowdim_outputs = list(self.obs_meta['lowdim'])

        # Initialize viewer
        self.viewer = EfficientOffScreenViewer(
            env.model,
            env.data,
            img_width,
            img_height,
            env.mujoco_renderer.max_geom,
            env.mujoco_renderer._vopt,
        )

        # Set up observation space
        self._setup_observation_space()

        # Handle camera pose variations if needed
        if camera_pose_variations:
            self._setup_camera_variations()

    def _setup_observation_space(self):
        """Sets up the observation space for the environment."""
        obs_space_dict = {'obs_gt': self.env.observation_space}
        
        # RGB observation spaces
        for key in self.rgb_outputs:
            obs_space_dict[key] = gymnasium.spaces.Box(
                low=0, high=255,
                shape=(self.img_height, self.img_width, 3),
                dtype=np.uint8
            )
        
        # Depth observation spaces
        for key in self.depth_outputs:
            obs_space_dict[key] = gymnasium.spaces.Box(
                low=0, high=1,
                shape=(self.img_height, self.img_width, 1),
                dtype=np.float32
            )
        
        # Point cloud observation spaces
        for key in self.pointcloud_outputs:
            obs_space_dict[key] = gymnasium.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=tuple(self.obs_meta['pointcloud'][key]),
                dtype=np.float32
            )
        
        # Low-dimensional observation spaces
        for key in self.lowdim_outputs:
            obs_space_dict[key] = gymnasium.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.obs_meta['lowdim'][key],),
                dtype=np.float32
            )

        # Add hand matrices if using point clouds
        if self.pointcloud_outputs:
            matrix_space = gymnasium.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(4, 4),
                dtype=np.float32
            )
            obs_space_dict.update({
                'hand_mat': matrix_space,
                'hand_mat_inv': matrix_space
            })

        self.observation_space = gymnasium.spaces.Dict(obs_space_dict)

    def _setup_camera_variations(self):
        """Sets up camera pose variations."""
        cam_id = mujoco.mj_name2id(
            self.env.model, 
            mujoco.mjtObj.mjOBJ_CAMERA, 
            'corner2'
        )
        cam_id2 = mujoco.mj_name2id(
            self.env.model, 
            mujoco.mjtObj.mjOBJ_CAMERA, 
            'corner3'
        )
        self.env.model.cam_pos[cam_id] = self.env.model.cam_pos[cam_id2]
        self.env.model.cam_quat[cam_id] = self.env.model.cam_quat[cam_id2]
        self.env.model.cam_fovy[cam_id] = self.env.model.cam_fovy[cam_id2]

    def step(self, action):
        obs_gt, reward, terminated, truncated, info = super().step(action)
        obs_gt = obs_gt.astype(np.float32)
        info['obs_gt'] = obs_gt
        next_obs = self.make_obs(obs_gt)
        terminated = info['success'] == 1
        return next_obs, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        obs_gt, info = super().reset()
        obs_gt = obs_gt.astype(np.float32)
        info['obs_gt'] = obs_gt
        obs = self.make_obs(obs_gt)
        return obs, info

    def make_obs(self, obs_gt):
        obs = {'obs_gt': obs_gt}
        
        # Get images and depths from all cameras
        image_dict, depth_dict = {}, {}
        for camera_name in self.cameras:
            image_obs, depth_obs = self.render(camera_name=camera_name, mode='all')
            depth_dict[camera_name] = depth_obs
            image_dict[camera_name] = image_obs

        # Process RGB observations
        for key in self.rgb_outputs:
            obs[key] = image_dict[key[:-4]][::-1]

        # Process depth observations
        for key in self.depth_outputs:
            obs[key] = np.expand_dims(depth_dict[key[:-6]][::-1], -1)

        # Process point cloud observations if needed
        if self.pointcloud_outputs:
            ims = np.array([image_dict[camera] for camera in self.cameras])
            depths_m = np.array([self.depth_img_2_meters(depth_dict[camera]) for camera in self.cameras])

            intrinsics = []
            extrinsics = []
            for key in self.cameras:
                intrinsics.append(self.get_camera_intrinsic_matrix(key))
                extrinsics.append(self.get_camera_extrinsic_matrix(key))
            
            o3d_clouds = []
            for i in range(len(self.cameras)):
                rgb_im = o3d.geometry.Image(np.ascontiguousarray(ims[i]))
                depth_im = o3d.geometry.Image(np.clip(np.ascontiguousarray(depths_m[i]), 0, 4))
                im = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    rgb_im, 
                    depth_im, 
                    convert_rgb_to_intensity=False,
                    depth_trunc=5,
                    depth_scale=1
                )
                o3d_cam_mat = cammat2o3d(intrinsics[i], self.img_width, self.img_height)
                cloud = o3d.geometry.PointCloud.create_from_rgbd_image(im, o3d_cam_mat)

                transformed_cloud = cloud.transform(extrinsics[i])

                camera_name = self.cameras[i]
                full_pc_name = f'{camera_name}_pointcloud_full'
                if full_pc_name in self.pointcloud_outputs:
                    obs[full_pc_name] = np.asarray(transformed_cloud.points).reshape(ims[i].shape)[::-1]

                if self.boundaries is not None:
                    bbox = o3d.geometry.AxisAlignedBoundingBox(
                        min_bound=self.boundaries[0], 
                        max_bound=self.boundaries[1]
                    )
                    cropped_cloud = transformed_cloud.crop(bbox)
                else:
                    cropped_cloud = transformed_cloud
                o3d_clouds.append(cropped_cloud)

            combined_cloud = o3d.geometry.PointCloud()
            for cloud in o3d_clouds:
                combined_cloud += cloud

            hand_data = self.env.data.body('hand')
            hand_pos = hand_data.xpos
            hand_quat = hand_data.xquat
            hand_rot_mat = quat2mat(hand_quat)
            hand_mat = posRotMat2Mat(hand_pos, hand_rot_mat)
            hand_mat_inv = np.linalg.inv(hand_mat)

            obs['hand_mat'] = hand_mat
            obs['hand_mat_inv'] = hand_mat_inv

        # Process low-dimensional observations
        if 'agent_pos' in self.lowdim_outputs:
            obs['agent_pos'] = obs_gt[:3]
        if 'ee_open' in self.lowdim_outputs:
            obs['ee_open'] = obs_gt[3:4]

        return obs

    def render(self, camera_name=None, mode='rgb_array'):
        if camera_name is None:
            camera_name = self.cameras[0]
        cam_id = mujoco.mj_name2id(
            self.env.model, 
            mujoco.mjtObj.mjOBJ_CAMERA, 
            camera_name
        )
        return self.viewer.render(render_mode=mode, camera_id=cam_id)

    def set_task(self, task):
        self.env.set_task(task)
        self.env._partially_observable = False

    def seed(self, seed):
        self.env.seed(seed)

    def close(self):
        self.viewer.close()
    
    def get_camera_intrinsic_matrix(self, camera_name):
        """
        Obtains camera intrinsic matrix.
        Args:
            camera_name (str): name of camera
            camera_height (int): height of camera images in pixels
            camera_width (int): width of camera images in pixels
        Return:
            K (np.array): 3x3 camera matrix
        """
        cam_id = mujoco.mj_name2id(
            self.env.model, 
            mujoco.mjtObj.mjOBJ_CAMERA, 
            camera_name
        )
        fovy = self.env.model.cam_fovy[cam_id]
        f = 0.5 * self.img_height / np.tan(fovy * np.pi / 360)
        K = np.array([[f, 0, self.img_width / 2], [0, f, self.img_height / 2], [0, 0, 1]])
        return K

    def get_camera_extrinsic_matrix(self, camera_name):
        """
        Returns a 4x4 homogenous matrix corresponding to the camera pose in the
        world frame. MuJoCo has a weird convention for how it sets up the
        camera body axis, so we also apply a correction so that the x and y
        axis are along the camera view and the z axis points along the
        viewpoint.
        Normal camera convention: https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
        Args:
            camera_name (str): name of camera
        Return:
            R (np.array): 4x4 camera extrinsic matrix
        """
        cam_id = mujoco.mj_name2id(
            self.env.model, 
            mujoco.mjtObj.mjOBJ_CAMERA, 
            camera_name
        )
        camera_pos = self.env.data.cam_xpos[cam_id]
        camera_rot = self.env.data.cam_xmat[cam_id].reshape(3, 3)
        R = T.make_pose(camera_pos, camera_rot)

        # IMPORTANT! This is a correction so that the camera axis is set up along the viewpoint correctly.
        camera_axis_correction = np.array(
            [[1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]]
        )
        R = R @ camera_axis_correction
        return R

    def depth_img_2_meters(self, depth):
        """Converts depth image to meters."""
        extent = self.env.model.stat.extent
        near = self.env.model.vis.map.znear * extent
        far = self.env.model.vis.map.zfar * extent
        return near / (1 - depth * (1 - near / far))

if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    from PIL import Image
    import numpy as np

    # Test configuration
    shape_meta = {
        "action_dim": 4,
        "observation": {
            "rgb": {
                "corner_rgb": (3, 128, 128),
                "corner2_rgb": (3, 128, 128),
            },
            "depth": {
                "corner_depth": (1, 128, 128),
                "corner2_depth": (1, 128, 128),
            },
            "lowdim": {
                "agent_pos": 3,
                "ee_open": 1,
            },
            "pointcloud": {
                "corner_pointcloud_full": (128, 128, 3),
                "corner2_pointcloud_full": (128, 128, 3),
            },
        },
    }

    # Initialize environment
    print("Initializing environment...")
    env = MetaWorldWrapper(
        env_name='button-press-topdown-v2',
        shape_meta=shape_meta,
        img_height=128,
        img_width=128,
        num_points=512,
        cameras=('corner', 'corner2'),
        camera_pose_variations=True,
        boundaries=None,
    )

    print("\nTesting reset and observation space...")
    obs, info = env.reset()

    print(obs.keys())
    for key, value in obs.items():
        if value is not None:
            print(f"{key}: shape {value.shape}, dtype {value.dtype}")
        else:
            print(f"{key}: None")

    print("\nTesting environment interaction...")
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for i in range(5):
        print(f"\nStep {i+1}")
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Reward: {reward}")
        print(f"Terminated: {terminated}")
        print(f"Truncated: {truncated}")
        
        if i == 2:
            axes[0, 0].imshow(obs['corner_rgb'])
            axes[0, 0].set_title('Corner RGB')
            axes[0, 1].imshow(obs['corner2_rgb'])
            axes[0, 1].set_title('Corner2 RGB')
            
            axes[1, 0].imshow(obs['corner_depth'][:, :, 0], cmap='viridis')
            axes[1, 0].set_title('Corner Depth')
            axes[1, 1].imshow(obs['corner2_depth'][:, :, 0], cmap='viridis')
            axes[1, 1].set_title('Corner2 Depth')

    plt.tight_layout()
    plt.show()

    print("\nVisualizing point clouds...")
    fig = plt.figure(figsize=(10, 5))
    
    ax1 = fig.add_subplot(121, projection='3d')
    corner_points = obs['corner_pointcloud_full'].reshape(-1, 3)
    ax1.scatter(
        corner_points[:, 0],
        corner_points[:, 1],
        corner_points[:, 2],
        c=obs['corner_rgb'].reshape(-1, 3) / 255.0,
        s=20
    )
    ax1.set_title('Corner Pointcloud')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.grid(True)

    ax2 = fig.add_subplot(122, projection='3d')
    corner2_points = obs['corner2_pointcloud_full'].reshape(-1, 3)
    ax2.scatter(
        corner2_points[:, 0],
        corner2_points[:, 1],
        corner2_points[:, 2],
        c=obs['corner2_rgb'].reshape(-1, 3) / 255.0,
        s=20
    )
    ax2.set_title('Corner2 Pointcloud')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.grid(True)
    
    ax1.set_box_aspect([1,1,1])
    ax2.set_box_aspect([1,1,1])
    
    plt.tight_layout()
    plt.show()

    print("\nClosing environment...")
    env.close()
    print("Test complete!")