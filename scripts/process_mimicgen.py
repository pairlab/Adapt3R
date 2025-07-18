import os
import json
import h5py
import time
import argparse
import numpy as np
from copy import deepcopy
from tqdm import tqdm

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
from robomimic.envs.env_base import EnvBase
import mimicgen.utils.robomimic_utils as RobomimicUtils

import sys
from adapt3r.utils.geometry import posRotMat2Mat, quat2mat

import open3d as o3d
from scipy.spatial.transform import Rotation

import torch
# import pytorch3d.ops as ops3d


def extract_trajectory(
    env, 
    initial_state, 
    states, 
    actions,
    done_mode,
    camera_names, 
    camera_height=84, 
    camera_width=84,
    randomize_camera_poses=False,
):
    """
    Helper function to extract observations, rewards, and dones along a trajectory using
    the simulator environment.

    Args:
        env (instance of EnvBase): environment
        initial_state (dict): initial simulation state to load
        states (np.array): array of simulation states to load to extract information
        actions (np.array): array of actions
        done_mode (int): how to write done signal. If 0, done is 1 whenever s' is a 
            success state. If 1, done is 1 at the end of each trajectory. 
            If 2, do both.
    """
    assert isinstance(env, EnvBase)
    assert states.shape[0] == actions.shape[0]

    # load the initial state
    env.reset()
    obs = env.reset_to(initial_state)

    if randomize_camera_poses:
        base_pos = env.env.sim.data.body('gripper0_eef').xpos

        camera_name = 'agentview'
        cam_id = env.env.sim.model.camera_name2id(camera_name)
        old_position = env.env.sim.model.cam_pos[cam_id].copy()
        old_rotation = env.env.sim.model.cam_quat[cam_id].copy()

        theta = np.random.uniform(-2 * np.pi / 3, 2 * np.pi / 3)
        r = old_position[0] - base_pos[0]
        x = r * np.cos(theta) + base_pos[0]
        y = r * np.sin(theta) + base_pos[1]
        z = old_position[2]
        new_position = np.array([x, y, z])

        Rz = Rotation.from_euler('z', theta).as_matrix()
        R_ref = Rotation.from_quat(old_rotation, scalar_first=True).as_matrix()
        R_total = Rz @ R_ref
        new_quat = Rotation.from_matrix(R_total).as_quat(scalar_first=True)

        env.env.sim.model.cam_pos[cam_id] = new_position
        env.env.sim.model.cam_quat[cam_id] = new_quat
        env.env.sim.step()
        obs = env.get_observation()

    traj = dict(
        obs=[], 
        rewards=[], 
        dones=[],
        actions=np.array(actions), 
        states=np.array(states), 
        initial_state_dict=initial_state,
        abs_actions=[],
    )

    traj_len = states.shape[0]

    
    for t in range(1, traj_len + 1):

        # get next observation
        if t == traj_len:
            # play final action to get next observation for last timestep
            next_obs, _, _, _ = env.step(actions[t - 1])
        else:
            # reset to previous state and step with action to update controller properly
            prev_state = {"states": states[t - 1]}
            env.reset_to(prev_state)
            # step with the action to update both state and controller
            next_obs, _, _, _ = env.step(actions[t - 1])

        for cam_name in camera_names:
            depth_f32 = obs[cam_name + "_depth"]
            depth_uint16 = (depth_f32 * 1000).astype(np.uint16)
            obs[cam_name + "_depth"] = depth_uint16

            # R = np.linalg.inv(env.get_camera_extrinsic_matrix(camera_name=cam_name))
            R = env.get_camera_extrinsic_matrix(camera_name=cam_name)
            K = env.get_camera_intrinsic_matrix(camera_name=cam_name, camera_height=camera_height, camera_width=camera_width)
            
            obs[cam_name + "_extrinsic"] = R.astype(np.float32)
            obs[cam_name + "_intrinsic"] = K.astype(np.float32)

    
        eef_data = env.base_env.sim.data.body('gripper0_eef')
        hand_pos = eef_data.xpos
        hand_quat = eef_data.xquat
        hand_rot_mat = quat2mat(hand_quat)
        hand_mat = posRotMat2Mat(hand_pos, hand_rot_mat)
        hand_mat_inv = np.linalg.inv(hand_mat)
        obs["hand_mat"] = hand_mat.astype(np.float32)
        obs["hand_mat_inv"] = hand_mat_inv.astype(np.float32)

        # infer reward signal
        r = env.get_reward()

        # infer done signal
        done = False
        if (done_mode == 1) or (done_mode == 2):
            done = done or (t == traj_len)
        if (done_mode == 0) or (done_mode == 2):
            done = done or env.is_success()["task"]
        done = int(done)

        # Extract absolute action from controller after proper state/action update
        controller = env.env.robots[0].controller
        goal_pos = controller.goal_pos
        goal_ori = Rotation.from_matrix(controller.goal_ori).as_rotvec()
        gripper = actions[t - 1][..., -1:]
        abs_action = np.concatenate((goal_pos, goal_ori, gripper))
        traj["abs_actions"].append(abs_action.astype(np.float32))

        # collect transition
        traj["obs"].append(deepcopy(obs))
        traj["rewards"].append(r)
        traj["dones"].append(done)

        # update for next iter
        obs = deepcopy(next_obs)

    # convert list of dict to dict of list for obs dictionaries (for convenient writes to hdf5 dataset)
    traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])

    # list to numpy array
    for k in traj:
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                traj[k][kp] = np.array(traj[k][kp])
        else:
            traj[k] = np.array(traj[k])

    return traj

def dataset_states_to_obs(args):
    if args.depth:
        assert len(args.camera_names) > 0, "must specify camera names if using depth"

    # create environment to use for data processing
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    env = RobomimicUtils.create_env(
        env_meta=env_meta,
        camera_names=args.camera_names, 
        camera_height=args.camera_height, 
        camera_width=args.camera_width, 
        use_depth_obs=args.depth,
    )

    # list of all demonstration episodes (sorted in increasing number order)
    f = h5py.File(args.dataset, "r")
    demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # maybe reduce the number of demonstrations to playback
    if args.n is not None:
        demos = demos[:args.n]

    # output file in same directory as input file
    output_path = args.output_name

    f_out = h5py.File(output_path, "w")
    data_grp = f_out.create_group("data")
    print("input file: {}".format(args.dataset))
    print("output file: {}".format(output_path))

    total_samples = 0
    for ind in tqdm(range(len(demos))):
        ep = demos[ind]

        # prepare initial state to reload from
        states = f["data/{}/states".format(ep)][()]
        initial_state = dict(states=states[0])
        if EnvUtils.is_robosuite_env(env_meta):
            initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]

        # extract obs, rewards, dones
        actions = f["data/{}/actions".format(ep)][()]
        traj = extract_trajectory(
            env=env, 
            initial_state=initial_state, 
            states=states, 
            actions=actions,
            done_mode=args.done_mode,
            camera_names=args.camera_names, 
            camera_height=args.camera_height, 
            camera_width=args.camera_width,
            randomize_camera_poses=args.randomize_camera_poses,
        )

        # maybe copy reward or done signal from source file
        if args.copy_rewards:
            traj["rewards"] = f["data/{}/rewards".format(ep)][()]
        if args.copy_dones:
            traj["dones"] = f["data/{}/dones".format(ep)][()]

        # store transitions
        ep_data_grp = data_grp.create_group(ep)
        ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
        ep_data_grp.create_dataset("abs_actions", data=np.array(traj["abs_actions"]))
        ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
        ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
        ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
 
        for k in traj["obs"]:
            ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]))

        # episode metadata
        if EnvUtils.is_robosuite_env(env_meta):
            ep_data_grp.attrs["model_file"] = traj["initial_state_dict"]["model"]
        ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0]

        total_samples += traj["actions"].shape[0]

    # copy over all filter keys that exist in the original hdf5
    if "mask" in f:
        f.copy("mask", f_out)

    # global metadata
    data_grp.attrs["total"] = total_samples
    data_grp.attrs["env_args"] = json.dumps(env.serialize(), indent=4)
    print("Wrote {} trajectories to {}".format(len(demos), output_path))

    f.close()
    f_out.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hdf5_path",
        type=str,
        required=True,
        help="path to input hdf5 dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="path to output hdf5 dataset",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="(optional) stop after n trajectories are processed",
    )

    parser.add_argument(
        "--camera_names",
        type=str,
        nargs='+',
        default=["agentview", "robot0_eye_in_hand"],
        help="(optional) camera name(s) to use for image observations. Leave out to not use image observations.",
    )
    parser.add_argument(
        "--camera_height",
        type=int,
        default=128, 
        help="(optional) height of image observations",
    )
    parser.add_argument(
        "--camera_width",
        type=int,
        default=128, 
        help="(optional) width of image observations",
    )
    parser.add_argument(
        "--depth", 
        action='store_true',
        help="(optional) use depth observations for each camera",
    )
    parser.add_argument(
        "--done_mode",
        type=int,
        default=2,
        help="how to write done signal. If 0, done is 1 whenever s' is a success state.\
            If 1, done is 1 at the end of each trajectory. If 2, both.",
    )
    parser.add_argument(
        "--copy_rewards", 
        action='store_true',
        help="(optional) copy rewards from source file instead of inferring them",
    )
    parser.add_argument(
        "--copy_dones", 
        action='store_true',
        help="(optional) copy dones from source file instead of inferring them",
    )
    parser.add_argument(
        "--randomize_camera_poses",
        action='store_true',
        help="(optional) randomize camera poses",
    )
    parser.add_argument(
        "--allow_overwrite",
        action='store_true',
        help="(optional) allow overwriting existing output file",
    )
  
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)

    input_path = args.hdf5_path
    output_dir = args.output_dir
            
    print(f"Processing {input_path}...")
    
    args.dataset = input_path
    args.output_name = output_dir

    if os.path.isfile(args.output_name) and not args.allow_overwrite:
        print(f"Output file {args.output_name} already exists. Skipping processing.")
        exit(0)
    
    print(f"input file: {args.dataset}")
    print(f"output file: {args.output_name}")
    
    dataset_states_to_obs(args)

