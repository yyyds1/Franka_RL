import tempfile
from pathlib import Path
from typing import Dict, List
import os
import json

import numpy as np
import torch
import cv2
from tqdm import trange
import sapien
import transforms3d.quaternions
import trimesh

from dex_retargeting import yourdfpy as urdf
from dex_retargeting.constants import RobotName, HandType, get_default_config_path, RetargetingType
from retargeting_config import RetargetingConfig
# from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting
from hand_viewer import HandDatasetSAPIENViewer

from Franka_RL.dataset import DexhandData
from Franka_RL.robots import DexHand
from Franka_RL.dataset.transform import quat_to_rotmat


ROBOT2MANO = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ]
)
ROBOT2MANO_POSE = sapien.Pose(q=transforms3d.quaternions.mat2quat(ROBOT2MANO))


def prepare_position_retargeting(joint_pos: np.array, link_hand_indices: np.ndarray):
    link_pos = joint_pos[link_hand_indices]
    return link_pos


def prepare_vector_retargeting(joint_pos: np.array, link_hand_indices_pairs: np.ndarray):
    joint_pos = joint_pos @ ROBOT2MANO
    origin_link_pos = joint_pos[link_hand_indices_pairs[0]]
    task_link_pos = joint_pos[link_hand_indices_pairs[1]]
    return task_link_pos - origin_link_pos


class RobotHandDatasetSAPIENViewer(HandDatasetSAPIENViewer):
    def __init__(self, robot: List[DexHand], hand_type: HandType, headless=False, use_ray_tracing=False):
        super().__init__(headless=headless, use_ray_tracing=use_ray_tracing)

        self.dexhands = robot
        self.robot_names = [dexhand.name for dexhand in robot]
        self.robots: List[sapien.Articulation] = []
        self.robot_file_names: List[str] = []
        self.retargetings: List[SeqRetargeting] = []
        self.retarget2sapien: List[np.ndarray] = []

        # Load optimizer and filter
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        loader.load_multiple_collisions_from_file = True
        for dexhand in self.dexhands:

            # Add 6-DoF dummy joint at the root of each robot to make them move freely in the space
            override = dict(add_dummy_free_joint=True)
            config = RetargetingConfig.from_dict(dexhand.retargeting_cfg, override=override)
            retargeting = config.build()
            robot_file_name = Path(config.urdf_path).stem
            self.robot_file_names.append(robot_file_name)
            self.retargetings.append(retargeting)

            # Build robot
            urdf_path = Path(config.urdf_path)
            if "glb" not in urdf_path.stem:
                urdf_path = urdf_path.with_stem(urdf_path.stem + "_glb")
            robot_urdf = urdf.URDF.load(str(urdf_path), add_dummy_free_joints=True, build_scene_graph=False)
            urdf_name = urdf_path.name
            temp_dir = tempfile.mkdtemp(prefix="dex_retargeting-")
            temp_path = f"{temp_dir}/{urdf_name}"
            robot_urdf.write_xml_file(temp_path)

            robot = loader.load(temp_path)
            self.robots.append(robot)
            sapien_joint_names = [joint.name for joint in robot.get_active_joints()]
            retarget2sapien = np.array([retargeting.joint_names.index(n) for n in sapien_joint_names]).astype(int)
            self.retarget2sapien.append(retarget2sapien)

    def load_object_hand(self, data: Dict):
        super().load_object_hand(data)
        obj_ids = data["obj_ids"]
        obj_mesh_files = data["object_mesh_file"]

        # Load the same YCB objects for n times, n is the number of robots
        # So that for each robot, there will be an identical set of objects
        for _ in range(len(self.robots)):
            for obj_id, obj_mesh_file in zip(obj_ids, obj_mesh_files):
                self._load_object(obj_id, obj_mesh_file)

    def render_data(self, data: Dict, fps=5, y_offset=0.8, record_traj=False):
        # Set table and viewer pose for better visual effect only
        global_y_offset = -y_offset * len(self.robots) / 2
        self.table.set_pose(sapien.Pose([0.5, global_y_offset + 0.2, 0]))
        if not self.headless:
            self.viewer.set_camera_xyz(1.5, global_y_offset, 1)
        else:
            local_pose = self.camera.get_local_pose()
            local_pose.set_p(np.array([1.5, global_y_offset, 1]))
            self.camera.set_local_pose(local_pose)

        hand_pose = data["hand_pose"]
        object_pose = data["object_pose"]
        num_frame = hand_pose.shape[0]
        num_copy = len(self.robots) + 1
        num_objects = len(data["obj_ids"])
        pose_offsets = []

        for i in range(len(self.robots) + 1):
            pose = sapien.Pose([0, -y_offset * i, 0])
            pose_offsets.append(pose)
            if i >= 1:
                self.robots[i - 1].set_pose(pose)

        # Skip frames where human hand is not detected in DexYCB dataset
        start_frame = 0
        for i in range(0, num_frame):
            init_hand_pose_frame = hand_pose[i]
            vertex, joint = self._compute_hand_geometry(init_hand_pose_frame)
            if vertex is not None:
                start_frame = i
                break

        if self.headless:
            robot_names = [robot.name for robot in self.robot_names]
            robot_names = "_".join(robot_names)
            video_path = Path(__file__).parent.resolve() / f"data/{robot_names}_video.mp4"
            writer = cv2.VideoWriter(
                str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (self.camera.get_width(), self.camera.get_height())
            )

        if record_traj:
            data = {}
            mesh = trimesh.load(data["object_mesh_file"][0], process=False, force="mesh")
            for dexhand in self.dexhands:
                data[dexhand.name] = DexhandData.empty_traj(num_frame - start_frame, data["obj_ids"][0], dexhand)
                data[dexhand.name]["obj_pcl"] = DexhandData.random_sampling_pc(mesh)

        # Loop rendering
        step_per_frame = int(60 / fps)
        for i in trange(start_frame, num_frame):
            object_pose_frame = object_pose[i]
            hand_pose_frame = hand_pose[i]
            vertex, joint = self._compute_hand_geometry(hand_pose_frame)

            # Update poses for YCB objects
            for k in range(num_objects):
                pos_quat = object_pose_frame[k]

                # Quaternion convention: xyzw -> wxyz
                pose = self.camera_pose * sapien.Pose(pos_quat[4:], np.concatenate([pos_quat[3:4], pos_quat[:3]]))
                self.objects[k].set_pose(pose)
                for copy_ind in range(num_copy):
                    self.objects[k + copy_ind * num_objects].set_pose(pos_quat[:3] + pose_offsets[copy_ind], pos_quat[3:])

            # Update pose for human hand
            self._update_hand(vertex)

            # Update poses for robot hands
            for robot, retargeting, retarget2sapien in zip(self.robots, self.retargetings, self.retarget2sapien):
                indices = retargeting.optimizer.target_link_human_indices
                ref_value = joint[indices, :]
                qpos = retargeting.retarget(ref_value)[retarget2sapien]
                robot.set_qpos(qpos)

            self.scene.update_render()
            if self.headless:
                self.camera.take_picture()
                rgb = self.camera.get_picture("Color")[..., :3]
                rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
                writer.write(rgb[..., ::-1])
            else:
                for _ in range(step_per_frame):
                    self.viewer.render()

            if record_traj:
                for robot, dexhand, copy_ind in zip(self.robots, self.dexhands, list(range(1, num_copy))):
                    
                    data[dexhand.name]["wrist_pos"][i - start_frame, :3] = robot.get_root_pose.p - pose_offsets[copy_ind]
                    data[dexhand.name]["wrist_pos"][i - start_frame, 3:] = robot.get_root_pose.q
                    
                    def get_indices(list1: List, list2: List):
                        return [list2.index(element) for element in list1]

                    data[dexhand.name]["joints_pos"][i - start_frame] = robot.get_qpos[get_indices(dexhand.dof_names, [joint.name for joint in robot.get_active_joints()])]
                    
                    body_pos = [link.get_pose() for link in robot.get_links()]
                    data[dexhand.name]["body_pos"][i - start_frame, :3] = body_pos[get_indices(dexhand.body_names, [link.name for link in robot.get_links()])].p
                    data[dexhand.name]["body_pos"][i - start_frame, 3:] = body_pos[get_indices(dexhand.body_names, [link.name for link in robot.get_links()])].q
                    
                    obj_pos = self.objects[copy_ind * num_objects].get_pose()
                    data[dexhand.name]["obj_pose"][i - start_frame, :3] = obj_pos.p - pose_offsets[copy_ind]
                    data[dexhand.name]["obj_pose"][i - start_frame, 3:] = obj_pos.q

                    curr_obj_pcl = (quat_to_rotmat(obj_pos.q) @ data[dexhand.name]["obj_pcl"].T[None]).transpose(-1, -2) + obj_pos.p - pose_offsets[copy_ind]
                    tips = joint[16:, :]
                    data[dexhand.name]["tip_distance"][i - start_frame] = DexhandData.compute_chamfer_distance(tips, curr_obj_pcl)
        
        if record_traj:
            for dexhand in self.dexhands:
                data[dexhand.name]["wrist_vel"][:, :3] = DexhandData.compute_velocity(data[dexhand.name]["wrist_pos"][:, :3])
                data[dexhand.name]["wrist_vel"][:, 3:] = DexhandData.compute_angular_velocity(quat_to_rotmat(data[dexhand.name]["wrist_pos"][:, 3:]))
                data[dexhand.name]["body_vel"][:, :3] = DexhandData.compute_velocity(data[dexhand.name]["body_pos"][:, :, :3])
                data[dexhand.name]["body_vel"][:, 3:] = DexhandData.compute_angular_velocity(quat_to_rotmat(data[dexhand.name]["body_pos"][:, :, 3:]))

                save_file = f"{data['capture_name']}_{dexhand.name}.json"
                save_path = os.path.join(data["data_dir"], "retargeting_result", save_file)

                for key, val in data[dexhand.name].items():
                    if type(val) is torch.Tensor:
                        data[dexhand.name][key] = val.to_list()

                with open(save_path, mode='w', encoding='utf-8') as f:
                    json.dump(data[dexhand.name], f, indent=4)


        if not self.headless:
            self.viewer.paused = True
            self.viewer.render()
        else:
            writer.release()
