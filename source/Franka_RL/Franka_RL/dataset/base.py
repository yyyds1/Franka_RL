from abc import ABC, abstractmethod
import os
from scipy.ndimage import gaussian_filter1d
import numpy as np
import torch
from Franka_RL.dataset.transform import aa_to_rotmat, caculate_align_mat, rotmat_to_aa
from torch.utils.data import Dataset
from pytorch3d.ops import sample_points_from_meshes
from termcolor import cprint
import pickle
from Franka_RL.robots import DexHand
import chamfer_distance as chd


class DexhandData(Dataset, ABC):
    def __init__(
        self,
        *,
        data_dir: str, 
        skip: int = 2,
        device: str = 'cpu',
        dexhand = None,
    ):
        # traj info
        self.data_dir = data_dir
        self.skip = skip
        self.data_pathes = None
        
        self.device = device
        self.dexhand = dexhand
        self.data = None
        self.fps = None
        # self.traj_num = 0
        # self.max_traj_length = 0
        # self.traj_len = None # length of each traj [traj_num]

        # # robot info
        # self.robot = dexhand
        # self.robot_usd_path = None
        # self.num_dof = 0
        # self.num_body = 0

        # # object info
        # self.use_obj = False # Whether the dataset includes object traj
        # self.obj_num = 0
        # self.obj_usd = None # objects' usd path [obj_num]
        # self.obj_pcl = None # objects' point cloud [obj_num, point_num, 3] 

        # # robot traj
        # self.wrist_pos = None # wrist pose [traj_num, max_traj_length, 7]
        # self.wrist_vel = None # wrist velocity [traj_num, max_traj_length, 6]
        # self.joints_pos = None # joints' angle [traj_num, max_traj_length, num_dof]
        # self.joints_vel = None # joints' velocity [traj_num, max_traj_length, num_dof]
        # self.body_pos = None # bodies' pose [traj_num, max_traj_length, num_body, 7]
        # self.body_vel = None # bodies' vel [traj_num, max_traj_length, num_body, 6]

        # # object traj
        # self.obj_pose = None # objects' pose [traj_num, max_traj_length, 7]
        # self.obj_vel = None # objects' pose [traj_num, max_traj_length, 6]
        # self.obj_id = None # objects' id in each traj [traj_num]
        # self.tip_distance = None # Chamfer distance between tip and object [traj_num, max_traj_length]

        # def __len__(self):
    #     return len(self.data_pathes)

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

    # def process_data(self, data, idx, rs_verts_obj):
    #     data["obj_trajectory"] = self.mujoco2gym_transf @ data["obj_trajectory"]
    #     data["wrist_pos"] = (self.mujoco2gym_transf[:3, :3] @ data["wrist_pos"].T).T + self.mujoco2gym_transf[:3, 3]
    #     data["wrist_rot"] = rotmat_to_aa(self.mujoco2gym_transf[:3, :3] @ data["wrist_rot"])
    #     for k in data["mano_joints"].keys():
    #         data["mano_joints"][k] = (
    #             self.mujoco2gym_transf[:3, :3] @ data["mano_joints"][k].T
    #         ).T + self.mujoco2gym_transf[:3, 3]

    #     # caculate distance
    #     obj_verts_transf = (data["obj_trajectory"][:, :3, :3] @ rs_verts_obj.T[None]).transpose(-1, -2) + data[
    #         "obj_trajectory"
    #     ][:, :3, 3][:, None]

    #     tip_list = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]

    #     tips = torch.cat(
    #         [data["mano_joints"][t_k][:, None] for t_k in (tip_list)],
    #         dim=1,
    #     )
    #     tips_near, _, _, _ = self.ch_dist(tips, obj_verts_transf)
    #     # tips_contact = tips_near <= 0.008**2  # ? 8mm, ch_dist return square distance
    #     data["tips_distance"] = torch.sqrt(tips_near)

    #     data["obj_velocity"] = self.compute_velocity(
    #         data["obj_trajectory"][:, None, :3, 3], 1 / (120 / self.skip), guassian_filter=True
    #     ).squeeze(1)
    #     data["obj_angular_velocity"] = self.compute_angular_velocity(
    #         data["obj_trajectory"][:, None, :3, :3], 1 / (120 / self.skip), guassian_filter=True
    #     ).squeeze(1)
    #     data["wrist_velocity"] = self.compute_velocity(
    #         data["wrist_pos"][:, None], 1 / (120 / self.skip), guassian_filter=True
    #     ).squeeze(1)
    #     data["wrist_angular_velocity"] = self.compute_angular_velocity(
    #         aa_to_rotmat(data["wrist_rot"][:, None]), 1 / (120 / self.skip), guassian_filter=True
    #     ).squeeze(1)
    #     data["mano_joints_velocity"] = {}
    #     for k in data["mano_joints"].keys():
    #         data["mano_joints_velocity"][k] = self.compute_velocity(
    #             data["mano_joints"][k], 1 / (120 / self.skip), guassian_filter=True
    #         )

    #     if len(data["obj_trajectory"]) > self.max_seq_len:
    #         cprint(
    #             f"WARN: {self.data_pathes[idx]} is too long : {len(data['obj_trajectory'])}, cut to {self.max_seq_len}",
    #             "yellow",
    #         )
    #         data["obj_trajectory"] = data["obj_trajectory"][: self.max_seq_len]
    #         data["obj_velocity"] = data["obj_velocity"][: self.max_seq_len]
    #         data["obj_angular_velocity"] = data["obj_angular_velocity"][: self.max_seq_len]
    #         data["wrist_pos"] = data["wrist_pos"][: self.max_seq_len]
    #         data["wrist_rot"] = data["wrist_rot"][: self.max_seq_len]
    #         for k in data["mano_joints"].keys():
    #             data["mano_joints"][k] = data["mano_joints"][k][: self.max_seq_len]
    #         data["wrist_velocity"] = data["wrist_velocity"][: self.max_seq_len]
    #         data["wrist_angular_velocity"] = data["wrist_angular_velocity"][: self.max_seq_len]
    #         for k in data["mano_joints_velocity"].keys():
    #             data["mano_joints_velocity"][k] = data["mano_joints_velocity"][k][: self.max_seq_len]
    #         data["tips_distance"] = data["tips_distance"][: self.max_seq_len]

    # def load_retargeted_data(self, data, retargeted_data_path):
    #     if not os.path.exists(retargeted_data_path):
    #         if self.verbose:
    #             cprint(f"\nWARNING: {retargeted_data_path} does not exist.", "red")
    #             cprint(f"WARNING: This may lead to a slower transfer process or even failure to converge.", "red")
    #             cprint(
    #                 f"WARNING: It is recommended to first execute the retargeting code to obtain initial values.\n",
    #                 "red",
    #             )
    #         data.update(
    #             {
    #                 "opt_wrist_pos": data["wrist_pos"],
    #                 "opt_wrist_rot": data["wrist_rot"],
    #                 "opt_dof_pos": torch.zeros([data["wrist_pos"].shape[0], self.dexhand.n_dofs], device=self.device),
    #             }
    #         )
    #     else:
    #         opt_params = pickle.load(open(retargeted_data_path, "rb"))
    #         data.update(
    #             {
    #                 "opt_wrist_pos": torch.tensor(opt_params["opt_wrist_pos"], device=self.device),
    #                 "opt_wrist_rot": torch.tensor(opt_params["opt_wrist_rot"], device=self.device),
    #                 "opt_dof_pos": torch.tensor(opt_params["opt_dof_pos"], device=self.device),
    #                 # "opt_joints_pos": torch.tensor(opt_params["opt_joints_pos"], device=self.device), # ? only used for ablation study
    #             }
    #         )
    #     data["opt_wrist_velocity"] = self.compute_velocity(
    #         data["opt_wrist_pos"][:, None], 1 / (120 / self.skip), guassian_filter=True
    #     ).squeeze(1)
    #     data["opt_wrist_angular_velocity"] = self.compute_angular_velocity(
    #         aa_to_rotmat(data["opt_wrist_rot"][:, None]), 1 / (120 / self.skip), guassian_filter=True
    #     ).squeeze(1)
    #     data["opt_dof_velocity"] = self.compute_dof_velocity(
    #         data["opt_dof_pos"], 1 / (120 / self.skip), guassian_filter=True
    #     )
    #     # data["opt_joints_velocity"] = self.compute_velocity(
    #     #     data["opt_joints_pos"], 1 / (120 / self.skip), guassian_filter=True
    #     # ) # ? only used for ablation study
    #     if len(data["opt_wrist_pos"]) > self.max_seq_len:
    #         data["opt_wrist_pos"] = data["opt_wrist_pos"][: self.max_seq_len]
    #         data["opt_wrist_rot"] = data["opt_wrist_rot"][: self.max_seq_len]
    #         data["opt_wrist_velocity"] = data["opt_wrist_velocity"][: self.max_seq_len]
    #         data["opt_wrist_angular_velocity"] = data["opt_wrist_angular_velocity"][: self.max_seq_len]
    #         data["opt_dof_pos"] = data["opt_dof_pos"][: self.max_seq_len]
    #         data["opt_dof_velocity"] = data["opt_dof_velocity"][: self.max_seq_len]
    #         # ? only used for ablation study
    #         # data["opt_joints_pos"] = data["opt_joints_pos"][: self.max_seq_len]
    #         # data["opt_joints_velocity"] = data["opt_joints_velocity"][: self.max_seq_len]
    #     assert len(data["opt_wrist_pos"]) == len(data["obj_trajectory"])


