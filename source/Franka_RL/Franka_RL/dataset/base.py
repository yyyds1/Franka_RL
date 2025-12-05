from abc import ABC, abstractmethod
import os
import json
from scipy.ndimage import gaussian_filter1d
import numpy as np
import torch
from Franka_RL.dataset.transform import aa_to_rotmat, caculate_align_mat, rotmat_to_aa
from torch.utils.data import Dataset
from pytorch3d.ops import sample_points_from_meshes
from termcolor import cprint
import pickle
from Franka_RL.robots import DexHand
from Franka_RL.dataset.transform import quat_to_rotmat, rotmat_to_euler
import chamfer_distance as chd
from isaaclab.utils.math import sample_uniform, quat_error_magnitude, transform_points, quat_mul, quat_conjugate


class DexhandData(Dataset, ABC):
    def __init__(
        self,
        *,
        data_dir: str, 
        skip: int = 2,
        device: str = 'cpu',
        dexhand: DexHand | None = None,
    ):
        # traj info
        self.data_dir = data_dir
        self.skip = skip
        self.data_pathes = None
        
        self.device = device
        self.dexhand = dexhand
        self.data = None
        self.fps = None
        self.max_traj_length = 1000

    @abstractmethod
    def __getitem__(self, idx):
        pass

    @staticmethod
    def compute_velocity(p, time_delta, guassian_filter=True):
        # [T, K, 3]
        velocity = np.gradient(p.cpu().numpy(), axis=0) / time_delta
        if guassian_filter:
            velocity = gaussian_filter1d(velocity, 2, axis=0, mode="nearest")
        return torch.from_numpy(velocity).to(p)

    @staticmethod
    def compute_angular_velocity(r, time_delta: float, guassian_filter=True):
        # [T, K, 3, 3]
        diff_r = r[1:] @ r[:-1].transpose(-1, -2)  # [T-1, K, 3, 3]
        diff_aa = rotmat_to_aa(diff_r).cpu().numpy()  # [T-1, K, 3]
        diff_angle = np.linalg.norm(diff_aa, axis=-1)  # [T-1, K]
        diff_axis = diff_aa / (diff_angle[:, :, None] + 1e-8)  # [T-1, K, 3]
        angular_velocity = diff_axis * diff_angle[:, :, None] / time_delta  # [T-1, K, 3]
        angular_velocity = np.concatenate([angular_velocity, angular_velocity[-1:]], axis=0)  # [T, K, 3]
        if guassian_filter:
            angular_velocity = gaussian_filter1d(angular_velocity, 2, axis=0, mode="nearest")
        return torch.from_numpy(angular_velocity).to(r)

    @staticmethod
    def compute_dof_velocity(dof, time_delta, guassian_filter=True):
        # [T, K]
        velocity = np.gradient(dof.cpu().numpy(), axis=0) / time_delta
        if guassian_filter:
            velocity = gaussian_filter1d(velocity, 2, axis=0, mode="nearest")
        return torch.from_numpy(velocity).to(dof)
    
    @staticmethod
    def compute_chamfer_distance(tips, obj_verts_transf):
        ch_dist = chd.ChamferDistance()
        tips_near, _, _, _ = ch_dist(tips, obj_verts_transf)
        return torch.sqrt(tips_near)

    @staticmethod
    def random_sampling_pc(mesh):
        numpy_random_state = np.random.get_state()
        torch_random_state = torch.random.get_rng_state()
        torch_random_state_cuda = torch.cuda.get_rng_state()
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        rs_verts_obj = sample_points_from_meshes(mesh, 2048, return_normals=False).squeeze(0)

        # reset seed
        np.random.set_state(numpy_random_state)
        torch.random.set_rng_state(torch_random_state)
        torch.cuda.set_rng_state(torch_random_state_cuda)

        return rs_verts_obj



    @staticmethod
    def empty_traj(traj_len, obj_id, dexhand: DexHand):
        data = {}
        data["traj_len"] = traj_len
        data["dexhand"] = dexhand.name
        data["obj_id"] = obj_id
        data["wrist_pos"] = np.zeros((traj_len, 7), dtype=np.float32)
        data["wrist_vel"] = np.zeros((traj_len, 6), dtype=np.float32)
        data["joints_pos"] = np.zeros((traj_len, dexhand.n_dofs), dtype=np.float32)
        data["body_pos"] = np.zeros((traj_len, dexhand.n_bodies, 7), dtype=np.float32)
        data["body_vel"] = np.zeros((traj_len, dexhand.n_bodies, 6), dtype=np.float32)
        data["obj_pose"] = np.zeros((traj_len, 7), dtype=np.float32)
        data["obj_vel"] = np.zeros((traj_len, 6), dtype=np.float32)
        data["obj_pcl"] = np.zeros((2048, 3), dtype=np.float32)
        data["tip_distance"] = np.zeros((traj_len, 5), dtype=np.float32)

        return data
    
    def allocate_buffer(self, traj_num):
        self.data = {}
        self.data["traj_num"] = traj_num
        self.data["max_traj_length"] = self.max_traj_length
        self.data["traj_len"] = torch.empty((traj_num), dtype=torch.int, device=self.device)
        self.data["wrist_pos"] = torch.empty((traj_num, self.max_traj_length, 7), dtype=torch.float32, device=self.device)
        self.data["wrist_vel"] = torch.empty((traj_num, self.max_traj_length, 6), dtype=torch.float32, device=self.device)
        self.data["joints_pos"] = torch.empty((traj_num, self.max_traj_length, self.dexhand.n_dofs), dtype=torch.float32, device=self.device)
        self.data["joints_vel"] = torch.empty((traj_num, self.max_traj_length, self.dexhand.n_dofs), dtype=torch.float32, device=self.device)
        self.data["body_pos"] = torch.empty((traj_num, self.max_traj_length, self.dexhand.n_bodies, 7), dtype=torch.float32, device=self.device)
        self.data["body_vel"] = torch.empty((traj_num, self.max_traj_length, self.dexhand.n_bodies, 6), dtype=torch.float32, device=self.device)
        self.data["obj_pose"] = torch.empty((traj_num, self.max_traj_length, 7), dtype=torch.float32, device=self.device)
        self.data["obj_vel"] = torch.empty((traj_num, self.max_traj_length, 6), dtype=torch.float32, device=self.device)
        self.data["obj_pcl"] = torch.empty((traj_num, 2048, 3), dtype=torch.float32, device=self.device)
        self.data["tip_distance"] = torch.empty((traj_num, self.max_traj_length, 5), dtype=torch.float32, device=self.device)
        self.data["obj_id"] = []
        self.data["obj_usd"] = []

    def load_data(self):
        path = os.path.join(self.data_dir, "retargeting_result")

        json_paths = []
        for filename in os.listdir(path):
            if(filename.endswith(".json")):
                filepath = os.path.join(path, filename)
                json_paths.append(filepath)

        self.allocate_buffer(len(json_paths))

        for i, json_file in enumerate(json_paths):
            with open(json_file, 'r', encoding='utf-8') as f:
                traj_data = json.load(f)
                self.data["traj_len"][i] = min(self.max_traj_length, traj_data["traj_len"])
                traj_len = self.data["traj_len"][i]
                self.data["wrist_pos"][i, :traj_len] = torch.tensor(traj_data["wrist_pos"][:traj_len], device=self.device, dtype=torch.float32)
                self.data["wrist_vel"][i, :traj_len] = torch.tensor(traj_data["wrist_vel"][:traj_len], device=self.device, dtype=torch.float32)
                self.data["joints_pos"][i, :traj_len] = torch.tensor(traj_data["joints_pos"][:traj_len], device=self.device, dtype=torch.float32)
                self.data["joints_vel"][i, :traj_len] = self.compute_dof_velocity(self.data["joints_pos"][i, :traj_len], time_delta=1 / self.fps)
                self.data["body_pos"][i, :traj_len] = torch.tensor(traj_data["body_pos"][:traj_len], device=self.device, dtype=torch.float32)
                self.data["body_vel"][i, :traj_len] = torch.tensor(traj_data["body_vel"][:traj_len], device=self.device, dtype=torch.float32)
                self.data["obj_pose"][i, :traj_len] = torch.tensor(traj_data["obj_pose"][:traj_len], device=self.device, dtype=torch.float32)
                self.data["obj_vel"][i, :traj_len] = torch.tensor(traj_data["obj_vel"][:traj_len], device=self.device, dtype=torch.float32)
                self.data["obj_pcl"][i] = torch.tensor(traj_data["obj_pcl"], device=self.device, dtype=torch.float32)
                self.data["tip_distance"][i, :traj_len] = torch.tensor(traj_data["tip_distance"][:traj_len], device=self.device, dtype=torch.float32)
                self.data["obj_id"].append(traj_data["obj_id"])
                
    def get_act_obs_pair(self):
        self.load_data()
        obs = torch.tensor([]).to(self.device).to(dtype=torch.float32)
        act = torch.tensor([]).to(self.device).to(dtype=torch.float32)


        for traj_id in range(self.data["traj_num"]):
            for frame_id in range(self.data["traj_len"][traj_id] - 1):
                proprioception = {}
                proprioception["joint_pos"] = self.data["joints_pos"][traj_id, frame_id] # num_dof
                proprioception["joint_vel"] = self.data["joints_vel"][traj_id, frame_id] # num_dof
                proprioception["cos_joint_pos"] = torch.cos(proprioception["joint_pos"]) # num_dof
                proprioception["sin_joint_pos"] = torch.sin(proprioception["joint_vel"]) # num_dof
                proprioception["wrist_pos"] = torch.cat([torch.zeros_like(self.data["wrist_pos"][traj_id, frame_id, :3]), self.data["wrist_pos"][traj_id, frame_id, 3:]], dim=-1) # ignore wrist position 7

                # target state
                next_target_state = {}
                obs_future_length = 5
                cur_idx = frame_id + 1
                cur_idx = torch.stack([cur_idx + t for t in range(obs_future_length)], dim=-1)
                cur_idx = torch.clamp(cur_idx, 0, self.data["traj_len"][traj_id] - 1)

                target_wrist_pos = self.data["wrist_pos"][traj_id, cur_idx, :3]
                cur_wrist_pos = self.data["wrist_pos"][traj_id, frame_id, :3]
                next_target_state["delta_wrist_pos"] = torch.flatten(target_wrist_pos - cur_wrist_pos[None])

                target_wrist_vel = self.data["wrist_vel"][traj_id, cur_idx, :3]
                cur_wrist_vel = self.data["wrist_vel"][traj_id, frame_id, :3]
                next_target_state["wrist_vel"] = torch.flatten(target_wrist_vel)
                next_target_state["delta_wrist_vel"] = torch.flatten(target_wrist_vel - cur_wrist_vel[None])

                target_wrist_rot = self.data["wrist_pos"][traj_id, cur_idx, 3:]
                cur_wrist_rot = self.data["wrist_pos"][traj_id, frame_id, 3:]

                next_target_state["wrist_quat"] = torch.flatten(target_wrist_rot)
                next_target_state["delta_wrist_quat"] = torch.flatten(
                    quat_mul(
                        cur_wrist_rot.repeat(obs_future_length).reshape(obs_future_length, -1),
                        quat_conjugate(target_wrist_rot),
                    )
                )

                target_wrist_ang_vel = self.data["wrist_vel"][traj_id, cur_idx, 3:]
                cur_wrist_ang_vel = self.data["wrist_vel"][traj_id, frame_id, 3:]
                next_target_state["wrist_ang_vel"] = torch.flatten(target_wrist_ang_vel)
                next_target_state["delta_wrist_ang_vel"] = torch.flatten(target_wrist_ang_vel - cur_wrist_ang_vel[None])

                target_joints_pos = self.data["body_pos"][traj_id, cur_idx, 1:, :3]
                cur_joint_pos = self.data["body_pos"][traj_id, frame_id, 1:, :3]
                next_target_state["delta_joints_pos"] = torch.flatten(target_joints_pos - cur_joint_pos[None])

                target_joints_vel = self.data["body_vel"][traj_id, cur_idx, 1:, :3]
                cur_joint_vel = self.data["body_vel"][traj_id, frame_id, 1:, :3]
                next_target_state["joints_vel"] = torch.flatten(target_joints_vel)
                next_target_state["delta_joints_vel"] = torch.flatten(target_joints_vel - cur_joint_vel[None])

                obs = torch.cat(
                    [
                        obs,
                        torch.cat(
                            [
                                proprioception[ob]
                                for ob in [
                                    "joint_pos",
                                    "joint_vel",
                                    "cos_joint_pos",
                                    "sin_joint_pos",
                                    "wrist_pos",
                                ]
                            ] + [
                                next_target_state[ob]
                                for ob in [
                                    "delta_wrist_pos",
                                    "wrist_vel",
                                    "delta_wrist_vel",
                                    "wrist_quat",
                                    "delta_wrist_quat",
                                    "wrist_ang_vel",
                                    "delta_wrist_ang_vel",
                                    "delta_joints_pos",
                                    "joints_vel",
                                    "delta_joints_vel",
                                ]
                            ],
                            dim=-1,
                        )
                    ],
                    dim=0
                )

                act = torch.cat(
                    [
                        act,
                        torch.cat(
                            [
                                self.data["wrist_pos"][traj_id, frame_id + 1, :3],
                                rotmat_to_euler(quat_to_rotmat(self.data["wrist_pos"][traj_id, frame_id + 1, 3:]), "XYZ"),
                                self.data["joints_pos"][traj_id, frame_id + 1]
                            ],
                            dim=-1,
                        )
                    ],
                    dim=0,
                )
    
        return obs, act