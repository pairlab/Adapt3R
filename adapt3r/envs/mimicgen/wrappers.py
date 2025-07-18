import numpy as np
import gymnasium
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from adapt3r.utils.geometry import posRotMat2Mat, quat2mat
from adapt3r.envs.utils.frame_stack import FrameStackObservationFixed
import adapt3r.envs.utils as eu
from copy import deepcopy
from gymnasium.vector.utils import concatenate

from mimicgen.envs.robosuite.three_piece_assembly import *
from mimicgen.envs.robosuite.stack import *
from mimicgen.envs.robosuite.threading import *
from mimicgen.envs.robosuite.coffee import *
# from mimicgen.envs.robosuite.kitchen import *
# from mimicgen.envs.robosuite.hammer_cleanup import *
from mimicgen.envs.robosuite.mug_cleanup import *
from mimicgen.envs.robosuite.nut_assembly import *

from robosuite import load_controller_config


_envs = {
    "square_d0": Square_D0,
    "square_d1": Square_D1,
    "square_d2": Square_D2,
    "threading_d0": Threading_D0,
    "threading_d1": Threading_D1,
    "threading_d2": Threading_D2,
    "stack_d0": Stack_D0,
    "stack_d1": Stack_D1,
    "stack_three_d0": StackThree_D0,
    "stack_three_d1": StackThree_D1,
    "three_piece_assembly_d0": ThreePieceAssembly_D0,
    "three_piece_assembly_d1": ThreePieceAssembly_D1,
    "three_piece_assembly_d2": ThreePieceAssembly_D2,
    "coffee_d0": Coffee_D0,
    "coffee_d1": Coffee_D1,
    "coffee_d2": Coffee_D2,
    "coffee_preparation_d0": CoffeePreparation_D0,
    "coffee_preparation_d1": CoffeePreparation_D1,
    "mug_cleanup_d0": MugCleanup_D0,
    "mug_cleanup_d1": MugCleanup_D1,
    # "kitchen_d0": Kitchen_D0,
    # "kitchen_d1": Kitchen_D1,
    # "hammer_cleanup_d0": HammerCleanup_D0,
    # "hammer_cleanup_d1": HammerCleanup_D1,
}


class MimicGenFrameStack(FrameStackObservationFixed):
    def set_init_state(self, *args, **kwargs):
        obs = self.env.set_init_state(*args, **kwargs)

        if self.padding_type == "reset":
            self.padding_value = obs
        for _ in range(self.stack_size - 1):
            self.obs_queue.append(self.padding_value)
        self.obs_queue.append(obs)

        updated_obs = deepcopy(
            concatenate(self.env.observation_space, self.obs_queue, self.stacked_obs)
        )
        return updated_obs

class MimicGenWrapper(gymnasium.Env):
    def __init__(self,
                 env_name,
                 shape_meta,
                 img_height=84,
                 img_width=84,
                 num_points=512,
                 cameras=('agentview', 'robot0_eye_in_hand'),
                 hd_rendering=False, # render in HD and then resize according to the above parameters, only enable for when collecting videos
                 abs_action=False,
                 robot='Panda',
                 camera_pose_variations=None,
                ):
        
        self.env_name = env_name
        self.img_width = img_width
        self.img_height = img_height

        obs_meta = shape_meta['observation']
        self.rgb_outputs = list(obs_meta['rgb'])
        self.depth_outputs = list(obs_meta['depth'])
        self.lowdim_outputs = list(obs_meta['lowdim'])

        self.num_points = num_points
        self.cameras = cameras
        self.hd_rendering = hd_rendering
        self.controller_config = load_controller_config(default_controller='OSC_POSE')
        self.controller_config['control_delta'] = not abs_action
        self.controller_config['control_delta'] = not abs_action
        self.camera_pose_variations = camera_pose_variations
        self.abs_action = abs_action

        self.counter = 0

        self.images = None
        self.depths = None
        self.intrinsics = None
        self.extrinsics = None

        self.env = _envs[env_name](
            robots=robot,
            gripper_types='PandaGripper',
            camera_names=self.cameras,
            camera_heights=self.img_height,
            camera_widths=self.img_width,
            camera_depths=True,
            controller_configs=self.controller_config,
            has_renderer=True,
        )

        obs_space_dict = {}
        for key in self.rgb_outputs:
            obs_space_dict[key] = gymnasium.spaces.Box(
                low=0,
                high=255,
                shape=(img_height, img_width, 3),
                dtype=np.uint8
            )
        for key in self.depth_outputs:
            obs_space_dict[key] = gymnasium.spaces.Box(
                low=0,
                high=1,
                shape=(img_height, img_width, 1),
                dtype=np.float32
            )
        for key in self.lowdim_outputs:
            obs_space_dict[key] = gymnasium.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_meta['lowdim'][key],),
                dtype=np.float32
            )

        for cam in self.cameras:
            obs_space_dict[eu.camera_name_to_intrinsic_key(cam)] = gymnasium.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(3, 3),
                dtype=np.float32
            )
            obs_space_dict[eu.camera_name_to_extrinsic_key(cam)] = gymnasium.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(4, 4),
                dtype=np.float32
            )
        
        obs_space_dict['hand_mat_inv'] = gymnasium.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(4, 4),
            dtype=np.float32
        )

        obs_space_dict['hand_mat'] = gymnasium.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(4, 4),
            dtype=np.float32
        )

        self.observation_space = gymnasium.spaces.Dict(obs_space_dict)
        self.action_space = gymnasium.spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)
        self.render_out = None

        if self.hd_rendering:
            new_sensors, new_names = self.env._create_camera_sensors(
                cam_name='agentview',
                cam_w=512,
                cam_h=512,
                cam_d=False,
                cam_segs=None,
            )
            observable = Observable(
                name='agentview_image_hd',
                sensor=new_sensors[0],
                sampling_rate=self.env.control_freq,
            )
            self.env.add_observable(observable)   

        if camera_pose_variations:
          
            camera_name = 'agentview'
            cam_id = self.env.sim.model.camera_name2id(camera_name)
            old_position = self.env.sim.model.cam_pos[cam_id].copy()
            old_rotation = self.env.sim.model.cam_quat[cam_id].copy()

            if type(camera_pose_variations) == str:

                if camera_pose_variations == 'small':
                    self.new_position = old_position + np.array([0, 0.3, -0.1])
                    self.new_rotation = np.array([0.44834694, 0.2579209 , 0.37187116, 0.77082661])
                elif camera_pose_variations == 'medium':
                    self.new_position = old_position + np.array([-0.2, 0.7, -0.2])
                    self.new_rotation = np.array([0.16658396, 0.23584841, 0.47143382, 0.83329194])
                elif camera_pose_variations == 'large':
                    self.new_position = old_position + np.array([-1.2, 1., -0.2])
                    self.new_rotation = np.array([-0.14345217, -0.20207276,  0.57364103,  0.78072021])
                else:
                    raise ValueError(f'invalid camera_pose_variation: {camera_pose_variations}')
                
            elif type(camera_pose_variations) in (int, float):

                # Rotate about the vertical line through the end effector
                base_pos = self.env.sim.data.body('gripper0_eef').xpos

                theta = camera_pose_variations
                # theta = np.pi / 2
                r = old_position[0] - base_pos[0]
                x = r * np.cos(theta) + base_pos[0]
                y = r * np.sin(theta) + base_pos[1]
                z = old_position[2]
                self.new_position = np.array([x, y, z])

                Rz = R.from_euler('z', theta).as_matrix()
                R_ref = R.from_quat(old_rotation, scalar_first=True).as_matrix()
                R_total = Rz @ R_ref
                new_quat = R.from_matrix(R_total).as_quat(scalar_first=True)

                self.new_rotation = new_quat


            camera_name = 'agentview'
            self.cam_id = self.env.sim.model.camera_name2id(camera_name)
            self.env.sim.model.cam_pos[self.cam_id] = self.new_position
            self.env.sim.model.cam_quat[self.cam_id] = self.new_rotation

    def reset(self, init_state=None, **kwargs):
        self.env.reset()
        if init_state is not None:
            self.set_init_state(init_state=init_state)
        if self.abs_action:
            goal_pos = self.env.sim.data.get_site_xpos('gripper0_grip_site')
            goal_ori = R.from_matrix(
                self.env.sim.data.get_site_xmat('gripper0_grip_site')
            ).as_rotvec()
            dummy = np.concatenate((goal_pos, goal_ori, [-1]))
        else:
            dummy = np.zeros((7,))
        if self.abs_action:
            goal_pos = self.env.sim.data.get_site_xpos('gripper0_grip_site')
            goal_ori = R.from_matrix(
                self.env.sim.data.get_site_xmat('gripper0_grip_site')
            ).as_rotvec()
            dummy = np.concatenate((goal_pos, goal_ori, [-1]))
        else:
            dummy = np.zeros((7,))
        for _ in range(5):
            raw_obs, _, _, _ = self.env.step(dummy)
        return self.make_obs(raw_obs), {}

    def step(self, action):
        raw_obs, reward, truncated, info = self.env.step(action)
        obs = self.make_obs(raw_obs)
        info['success'] = self.env._check_success()
        terminated = info['success']
        return obs, reward, terminated, truncated, info
    
    def get_sim_state(self):
        state = {'state': self.env.sim.get_state().flatten()}
        
        return state['state']

    def set_init_state(self, init_state):
        # self.env.set_init_state(*args, **kwargs)
        obs = self.regenerate_obs_from_state(init_state)
        # obs = self.regenerate_obs_from_state(init_state)
        
        # Re-apply camera variations after reset if needed
        if self.camera_pose_variations:
            self.env.sim.model.cam_pos[self.cam_id] = self.new_position
            self.env.sim.model.cam_quat[self.cam_id] = self.new_rotation

    def set_state(self, mujoco_state):
        self.env.sim.set_state_from_flattened(mujoco_state)

    def regenerate_obs_from_state(self, mujoco_state):
        self.set_state(mujoco_state)
        self.env.sim.forward()
        return self.env._get_observations()

    def make_obs(self, raw_obs):

        obs = {}

        if self.hd_rendering:
            self.render_out = raw_obs['agentview_image_hd'][::-1]
        else:
            self.render_out = raw_obs[eu.camera_name_to_image_key(self.cameras[0])][::-1]

        for key in self.rgb_outputs:
            obs[key] = raw_obs[key]

        for key in self.depth_outputs:
            obs[key] = raw_obs[key]

        for key in self.lowdim_outputs:
            obs[key] = raw_obs[key]

        for cam_name in self.cameras:
            K = self.get_camera_intrinsic_matrix(camera_name=cam_name)
            R = self.get_camera_extrinsic_matrix(camera_name=cam_name)

            image = np.ascontiguousarray(raw_obs[eu.camera_name_to_image_key(cam_name)][::-1])
            depth = np.ascontiguousarray((self.get_real_depth_map(raw_obs[eu.camera_name_to_depth_key(cam_name)].squeeze()[::-1]) * 1000).astype(np.uint16))

            obs[eu.camera_name_to_image_key(cam_name)] = image
            obs[eu.camera_name_to_depth_key(cam_name)] = depth[..., np.newaxis]

            obs[eu.camera_name_to_intrinsic_key(cam_name)] = K
            obs[eu.camera_name_to_extrinsic_key(cam_name)] = R

        eef_data = self.env.sim.data.body('gripper0_eef')
        hand_pos = eef_data.xpos
        hand_quat = eef_data.xquat
        hand_rot_mat = quat2mat(hand_quat)
        hand_mat = posRotMat2Mat(hand_pos, hand_rot_mat)
        hand_mat_inv = np.linalg.inv(hand_mat)

        obs["hand_mat"] = hand_mat.astype(np.float32)
        obs["hand_mat_inv"] = hand_mat_inv.astype(np.float32)
        return obs
    
    def render(self, mode='human'):
        return self.render_out
    
    def get_real_depth_map(self, depth_map):
        """
        Reproduced from https://github.com/ARISE-Initiative/robosuite/blob/c57e282553a4f42378f2635b9a3cbc4afba270fd/robosuite/utils/camera_utils.py#L106
        since older versions of robosuite do not have this conversion from normalized depth values returned by MuJoCo
        to real depth values.
        """
        # Make sure that depth values are normalized
        assert np.all(depth_map >= 0.0) and np.all(depth_map <= 1.0)
        extent = self.env.sim.model.stat.extent
        far = self.env.sim.model.vis.map.zfar * extent
        near = self.env.sim.model.vis.map.znear * extent
        return near / (1.0 - depth_map * (1.0 - near / far))
    
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
        cam_id = self.env.sim.model.camera_name2id(camera_name)
        fovy = self.env.sim.model.cam_fovy[cam_id]
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
        cam_id = self.env.sim.model.camera_name2id(camera_name)
        camera_pos = self.env.sim.data.cam_xpos[cam_id]
        camera_rot = self.env.sim.data.cam_xmat[cam_id].reshape(3, 3)
        R = T.make_pose(camera_pos, camera_rot)

        # IMPORTANT! This is a correction so that the camera axis is set up along the viewpoint correctly.
        camera_axis_correction = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        )
        R = R @ camera_axis_correction
        return R

    def visualize_camera_position(self, position, rotation):
        camera_name = 'agentview'
        cam_id = self.env.sim.model.camera_name2id(camera_name)
        self.env.sim.model.cam_pos[cam_id] = position
        self.env.sim.model.cam_quat[cam_id] = rotation

        dummy = np.zeros((7,))
        raw_obs, _, _, _ = self.env.step(dummy)

        plt.imshow(raw_obs['agentview_image'][::-1])
        plt.show()

    
def main():
    # Define shape metadata
    shape_meta = {
        'observation': {
            'rgb': ['agentview_image', 'robot0_eye_in_hand_image'],
            'depth': ['agentview_depth', 'robot0_eye_in_hand_depth'],
            'pointcloud': {},
            'lowdim': {
                'robot0_eef_pos': 3,
                'robot0_eef_quat': 4,
            }
        }
    }

    # Create environment
    env = MimicGenWrapper(
        env_name="stack_d0",
        shape_meta=shape_meta,
        img_height=84,
        img_width=84,
        num_points=512,
        cameras=('agentview', 'robot0_eye_in_hand'),
        device="cuda",
        robot='UR5e',
        # camera_pose_variations=(2 * np.pi / 3) # Enable camera pose variations
    )

    # Reset environment
    obs, _ = env.reset()
    print("\nObservation keys:", obs.keys())

    env.step(np.zeros((7,)))

    camera_name = 'agentview'
    cam_id = env.env.sim.model.camera_name2id(camera_name)
    plt.imshow(obs[f'{camera_name}_image'])
    plt.show()
    
    # Visualize current camera position
    # env.visualize_camera_position(
    #     # env.env.sim.model.cam_pos[env.cam_id],
    #     # env.env.sim.model.cam_quat[env.cam_id]
    # )

    breakpoint()
    
    # # Run a few random actions
    # for i in range(5):
    #     # Take random action
    #     action = env.action_space.sample()
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     env.env.render()
        
    #     print(f"\nStep {i + 1}")
    #     print(f"Reward: {reward}")
    #     print(f"Success: {info['success']}")
        
    #     # Create first figure for images
    #     fig1 = plt.figure(figsize=(10, 5))
        
    #     # RGB Image
    #     ax1 = fig1.add_subplot(121)
    #     ax1.imshow(obs['agentview_image'][::-1])
    #     ax1.set_title('Agent View')
        
    #     ax2 = fig1.add_subplot(122)
    #     ax2.imshow(obs['robot0_eye_in_hand_image'][::-1])
    #     ax2.set_title('Gripper View')
        
    #     plt.tight_layout()
    #     plt.show()

    #     if terminated or truncated:
    #         print("Episode finished")
    #         break

if __name__ == "__main__":
    main()