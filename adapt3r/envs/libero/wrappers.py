import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import matplotlib.pyplot as plt
from copy import deepcopy

import numpy as np
np.set_printoptions(suppress=True)
import torch
import open3d as o3d
from scipy.spatial.transform import Rotation

import gymnasium
from gymnasium.vector.utils import batch_space, concatenate
from libero.libero.envs import SubprocVectorEnv, DummyVectorEnv
from robosuite.utils.observables import Observable
import robosuite.utils.transform_utils as T
import robosuite as suite

import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs import *

from adapt3r.envs.utils.frame_stack import FrameStackObservationFixed
from adapt3r.utils.geometry import posRotMat2Mat, quat2mat
import adapt3r.envs.libero.utils as lu
from adapt3r.envs.libero.custom_task_map import libero_custom_task_map
from adapt3r.envs.libero.domain_randomization_wrapper import FixedDomainRandomizationWrapper
import adapt3r.envs.utils as eu
from scipy.spatial.transform import Rotation as R


RANDOMIZE_COLOR_ARGS = {
    "geom_names": None,  # all geoms are randomized
    "randomize_local": False,  # sample nearby colors
    "randomize_material": True,  # randomize material reflectance / shininess / specular
    "local_rgb_interpolation": 0.2,
    "local_material_interpolation": 0.3,
    "texture_variations": ["rgb", "checker", "noise", "gradient"],  # all texture variation types
    "randomize_skybox": True,  # by default, randomize skybox too
}

RANDOMIZE_LIGHTING_ARGS = {
    "light_names": None,  # all lights are randomized
    "randomize_position": True,
    "randomize_direction": True,
    "randomize_specular": True,
    "randomize_ambient": True,
    "randomize_diffuse": True,
    "randomize_active": True,
    "position_perturbation_size": 0.1,
    "direction_perturbation_size": 0.35,
    "specular_perturbation_size": 0.1,
    "ambient_perturbation_size": 0.1,
    "diffuse_perturbation_size": 0.1,
}


class ControlEnv:
    def __init__(
        self,
        bddl_file_name,
        robots=["Panda"],
        controller="OSC_POSE",
        abs_action=False,
        gripper_types="default",
        initialization_noise=None,
        use_camera_obs=True,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names=[
            "agentview",
            "robot0_eye_in_hand",
        ],
        camera_heights=128,
        camera_widths=128,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mujoco",
        renderer_config=None,
        wrapper=None,
        **kwargs,
    ):
        assert os.path.exists(
            bddl_file_name
        ), f"[error] {bddl_file_name} does not exist!"

        controller_configs = suite.load_controller_config(default_controller=controller)
        controller_configs['control_delta'] = not abs_action

        problem_info = BDDLUtils.get_problem_info(bddl_file_name)
        # Check if we're using a multi-armed environment and use env_configuration argument if so

        # Create environment
        self.problem_name = problem_info["problem_name"]
        self.domain_name = problem_info["domain_name"]
        self.language_instruction = problem_info["language_instruction"]
        self.env = TASK_MAPPING[self.problem_name](
            bddl_file_name,
            robots=robots,
            controller_configs=controller_configs,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
            **kwargs,
        )

        if wrapper is not None:
            self.env = wrapper(self.env)

    @property
    def obj_of_interest(self):
        return self.env.obj_of_interest

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        ret = self.env.reset()
        return ret

    def check_success(self):
        return self.env._check_success()

    @property
    def _visualizations(self):
        return self.env._visualizations

    @property
    def robots(self):
        return self.env.robots

    @property
    def sim(self):
        return self.env.sim

    def get_sim_state(self):
        return self.env.sim.get_state().flatten()

    def _post_process(self):
        return self.env._post_process()

    def _update_observables(self, force=False):
        self.env._update_observables(force=force)

    def set_state(self, mujoco_state):
        self.env.sim.set_state_from_flattened(mujoco_state)

    def reset_from_xml_string(self, xml_string):
        self.env.reset_from_xml_string(xml_string)

    def seed(self, seed):
        self.env.seed(seed)

    def set_init_state(self, init_state):
        return self.regenerate_obs_from_state(init_state)

    def regenerate_obs_from_state(self, mujoco_state):
        self.set_state(mujoco_state)
        self.env.sim.forward()
        self.check_success()
        self._post_process()
        self._update_observables(force=True)
        return self.env._get_observations()

    def close(self):
        self.env.close()
        del self.env


class OffScreenRenderEnv(ControlEnv):
    """
    For visualization and evaluation.
    """

    def __init__(self, **kwargs):
        # This shouldn't be customized
        kwargs["has_renderer"] = False
        kwargs["has_offscreen_renderer"] = True
        super().__init__(**kwargs)





class LiberoVectorWrapper(gymnasium.Env):
    def __init__(self,
                 env_factory,
                 env_num=1):
        if type(env_factory) == list:
            env_num = len(env_factory)
        else:
            env_factory = [env_factory for _ in range(env_num)]
        
        if env_num == 1:
            env = DummyVectorEnv(env_factory)
        else:
            env = SubprocVectorEnv(env_factory)

        self._env = env
        self.action_space = batch_space(self._env.action_space[0], env_num)
        self.observation_space = batch_space(self._env.observation_space[0], env_num)

    def reset(self, init_states=None, *args, **kwargs):
        obs, info = self._env.reset(*args, **kwargs)

        obs = self.process_obs(obs)
        if init_states is not None:

            obs = self.process_obs(self._env.set_init_state(init_states))
        return obs, info
    
    def step(self, *args, **kwargs):
        obs, reward, terminated, truncated, info = self._env.step(*args, **kwargs)
        obs = self.process_obs(obs)
        return obs, reward, terminated, truncated, info
    
    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)

    def process_obs(self, obs):
        """LIBERO vectorization wrapper does not handle dict obs well"""
        obs_out = {key: [] for key in obs[0]}
        for env_obs in obs:
            for key in obs_out:
                obs_out[key].append(env_obs[key])
        for key in obs_out:
            obs_out[key] = np.array(obs_out[key])
        return obs_out


class LiberoFrameStack(FrameStackObservationFixed):
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


class LiberoWrapper(gymnasium.Env):
    def __init__(self,
                 shape_meta,
                 task_id=None,
                 benchmark=None,
                 bddl_file=None,
                 img_height=128,
                 img_width=128,
                 hd_rendering=False, # render in HD and then resize according to the above parameters, only enable for when collecting videos
                 cameras=('agentview', 'robot0_eye_in_hand'),
                 abs_action=False,                
                 camera_pose_variations=False,
                 distractor_objects=False,
                 lighting_variations=False,
                 color_variations=False,
                 skip_render=False,
                 robot='Panda',
                 device="cuda",):
        self.img_width = img_width
        self.img_height = img_height
        self.hd_rendering = hd_rendering
        obs_meta = shape_meta['observation']
        self.rgb_outputs = list(obs_meta['rgb']) if not skip_render else []
        self.depth_outputs = list(obs_meta['depth']) if not skip_render else []
        self.lowdim_outputs = list(obs_meta['lowdim'])
        self.cameras = cameras
        self.camera_pose_variations = camera_pose_variations
        self.abs_action = abs_action

        self.device = device

        if lighting_variations or color_variations:
            wrapper = lambda env: FixedDomainRandomizationWrapper(
                env=env,
                randomize_camera=False,
                randomize_dynamics=False,
                randomize_color=color_variations,
                randomize_lighting=lighting_variations,
                color_randomization_args=RANDOMIZE_COLOR_ARGS,
                lighting_randomization_args=RANDOMIZE_LIGHTING_ARGS,
                randomize_on_reset=True,
                randomize_every_n_steps=100000, # don't randomize mid episode,
            )
        else:
            wrapper = None
    
        assert (benchmark is not None or task_id is not None) != (bddl_file is not None), \
            f'exactly one of benchmark and bddl file must not be None, got {benchmark}, {task_id}, {bddl_file}'
        if bddl_file is None:
            if distractor_objects:
                bddl_file = os.path.join('quest/libero_distractor/bddl_files', 
                                         benchmark.name, 
                                         f'{libero_custom_task_map[benchmark.name][task_id]}.bddl')
            else:
                bddl_file = benchmark.get_task_bddl_file_path(task_id)

        env = OffScreenRenderEnv(
            robots=[robot],
            bddl_file_name=bddl_file,
            camera_heights=self.img_height,
            camera_widths=self.img_width,
            camera_depths=len(self.depth_outputs) > 0,
            camera_names=cameras,
            abs_action=abs_action,
            hard_reset=True,
            wrapper=wrapper,
            has_offscreen_renderer=not skip_render,
        )
        self.env = env

        if self.hd_rendering:
            new_sensors, new_names = self.env.env._create_camera_sensors(
                cam_name='agentview',
                cam_w=512,
                cam_h=512,
                cam_d=False,
                cam_segs=None,
            )
            observable = Observable(
                name='agentview_image_hd',
                sensor=new_sensors[0],
                sampling_rate=self.env.env.control_freq,
            )
            self.env.env.add_observable(observable)            

        if camera_pose_variations:

            # rotation_angle_deg = 10
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
        for camera_name in self.cameras:
            obs_space_dict[f'{camera_name}_intrinsic'] = gymnasium.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(3, 3),
                dtype=np.float32
            )
            obs_space_dict[f'{camera_name}_extrinsic'] = gymnasium.spaces.Box(
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
        obs_space_dict['hand_mat_inv'] = gymnasium.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(4, 4),
            dtype=np.float32
        )
        self.observation_space = gymnasium.spaces.Dict(obs_space_dict)

        # TODO: this action space doesn't work with absolute actions, need to fix bounds
        self.action_space = gymnasium.spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)
        self.render_out = None

    def reset(self, init_states=None, **kwargs):
        raw_obs = self.env.reset()
        if init_states is not None:
            obs = self.set_init_state(init_states)
        else:
            obs = self.make_obs(raw_obs)
        return obs, {}

    def step(self, action):
        raw_obs, reward, truncated, info = self.env.step(action)
        obs = self.make_obs(raw_obs)
        info['success'] = self.env.check_success()
        terminated = info['success']
        terminated = truncated = True
        return obs, reward, terminated, truncated, info
    
    def set_init_state(self, init_state):
        if init_state is not None:
            self.env.set_init_state(init_state)
        else:
            self.env.set_init_state(self.env.get_sim_state().flatten())

        if self.camera_pose_variations:
            self.env.sim.model.cam_pos[self.cam_id] = self.new_position
            self.env.sim.model.cam_quat[self.cam_id] = self.new_rotation

        if self.abs_action:
            goal_pos = self.env.sim.data.get_site_xpos('gripper0_grip_site')
            goal_ori = Rotation.from_matrix(
                self.env.sim.data.get_site_xmat('gripper0_grip_site')
            ).as_rotvec()
            dummy = np.concatenate((goal_pos, goal_ori, [-1]))
        else:
            dummy = np.zeros((7,))
        for _ in range(5):
            raw_obs, _, _, _ = self.env.step(dummy)
        
        return self.make_obs(raw_obs)

    def make_obs(self, raw_obs):
        obs = {}


        if self.hd_rendering:
            self.render_out = raw_obs['agentview_image_hd'][::-1]
        else:
            self.render_out = raw_obs[eu.camera_name_to_image_key(self.cameras[0])][::-1]

        for key in self.lowdim_outputs:
            obs[key] = raw_obs[key]

        for cam_name in self.cameras:
            K = self.get_camera_intrinsic_matrix(camera_name=cam_name)
            R = self.get_camera_extrinsic_matrix(camera_name=cam_name)

            image_key = eu.camera_name_to_image_key(cam_name)
            image = np.ascontiguousarray(raw_obs[image_key][::-1])
            obs[image_key] = image
            
            if len(self.depth_outputs) > 0:
                depth_key = eu.camera_name_to_depth_key(cam_name)
                depth_raw = raw_obs[depth_key].squeeze()[::-1]
                depth_meters = self.depth_im_to_meters(depth_raw)
                depth = np.ascontiguousarray((depth_meters * 1000).astype(np.uint16))
                obs[depth_key] = depth[..., np.newaxis]

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
    
    def render(self, *args, **kwargs):
        return self.render_out
    
    def depth_im_to_meters(self, depth_im):
        extent = self.env.sim.model.stat.extent
        near = self.env.sim.model.vis.map.znear * extent
        far = self.env.sim.model.vis.map.zfar * extent
        depth_im_m = near / (1 - np.array(depth_im) * (1 - near / far)) 
        return depth_im_m

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
