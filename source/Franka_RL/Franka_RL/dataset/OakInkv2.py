import json
import os
from pathlib import Path
import pickle
from functools import lru_cache

import numpy as np
import torch

from .base import DexhandData
from .decorators import register_dataset

from .transform import quat_to_aa, rotmat_to_quat, rotmat_to_aa

from Franka_RL.dataset.oakink2_layer.smplx import SMPLXLayer

@register_dataset("OakInkv2_rh")
class OakInkv2DatasetRH(DexhandData):
    def __init__(
        self, 
        *, 
        data_dir = "dataset/OakInk-v2", 
        skip: int = 2,
        device = 'cpu', 
        dexhand=None,
        **kwargs,
    ):
        super().__init__(data_dir=data_dir, device=device, dexhand=dexhand, skip=skip)

        pathes = os.listdir(os.path.join(data_dir, "anno_preview"))
        pathes = [os.path.join(data_dir, "anno_preview", p) for p in pathes]
        pathes.sort(key=lambda x: x.split("/")[-1])
        self.data_pathes = pathes
        # * We use the first 5 digits of hash as the index
        self.seq_hashes = {os.path.split(p)[-1].split("_")[5][:5]: i for i, p in enumerate(pathes)}
        self.fps = 120 / self.skip

    def __getitem__(self, index):
        capture_name = index
        if type(index) == str:
            index = (index.split("@")[0], int(index.split("@")[1]))

        assert (
            type(index) == tuple and len(index) == 2 and type(index[0]) == str and type(index[1]) == int
        ), "index error"
        assert (
            index[0] in self.seq_hashes
        ), f"index {index[0]} not found, please check the 5 digits hash (first 5 digits of the sequence hash) in the data_pathes"
        idx = self.seq_hashes[index[0]]
        stage = index[1]

        anno = self.data_pathes[idx]
        anno = pickle.load(open(anno, "rb"))
        # anno = np.load(anno, allow_pickle=True)


        frame_id_list = anno["mocap_frame_id_list"]  # ! 120HZ
        frame_id_list = frame_id_list[:: self.skip]

        program_filepath = os.path.join(
            self.data_dir,
            "program",
            "program_info",
            f"{os.path.splitext(os.path.split(self.data_pathes[idx])[1])[0]}.json",
        )
        program_info = {}
        with open(program_filepath, "r") as ifs:
            _program_info = json.load(ifs)
            for k, v in _program_info.items():
                seg_pair_def = eval(k)
                program_info[seg_pair_def] = v

        left_hand_range = list(program_info.keys())[stage][0]
        right_hand_range = list(program_info.keys())[stage][1]

        def intersection(lst1, lst2):
            begin = max(lst1[0], lst2[0])
            end = min(lst1[1], lst2[1])
            if begin < end:
                return [begin, end]
            else:
                assert False, f"no intersection between {lst1} and {lst2}"

        assert (
            right_hand_range is not None
        ), f"Right hand data is empty. Please check if {index} is a left-hand-only task."

        if left_hand_range is not None:
            right_hand_range = intersection(left_hand_range, right_hand_range)

        program_info_selected = program_info[list(program_info.keys())[stage]]

        frame_id_list = [f for f in frame_id_list if right_hand_range[0] <= f <= right_hand_range[1]]

        # extrinsics = anno["cam_extr"][anno["cam_selection"][0]][frame_id_list[0]]
        # extrinsics = torch.tensor(extrinsics, device=self.device)
        extrinsics = torch.eye(4, dtype=torch.float32)

        object_list = anno["obj_list"]
        obj_id = [program_info_selected["obj_list_rh"][0]]
        obj_mesh_path = []
        for obj in obj_id:
            obj_filedir = os.path.join(self.data_dir, "object_preview", "align_ds", obj)
            if(not os.path.exists(obj_filedir)):
                return None
            candidate_list = [el for el in os.listdir(obj_filedir) if os.path.splitext(el)[-1] in [".obj", ".ply"]]
            assert len(candidate_list) == 1
            obj_filename = candidate_list[0]
            obj_filepath = os.path.join(obj_filedir, obj_filename)
            obj_mesh_path.append(obj_filepath)

        length = len(frame_id_list)


        SMPLX_ROT_MODE = "quat"
        SMPLX_DIM_SHAPE_ALL = 300

        self.smplx_layer = SMPLXLayer(
            Path(__file__).parent.resolve() / "oakink2_layer" / "body_utils" / "body_models" / "smplx",
            dtype=torch.float32,
            rot_mode=SMPLX_ROT_MODE,
            num_betas=SMPLX_DIM_SHAPE_ALL,
            gender="neutral",
            use_body_upper_asset=Path(__file__).parent.resolve() / "oakink2_layer" / "smplx_extra" / "body_upper_idx.pt",
        ).to(self.device)

        hand_pose = []
        object_pose = []
        smplx_result = []

        for frame_id in frame_id_list:

            # raw_mano = anno["raw_mano"][frame_id]
            # rh_pose_coeffs = raw_mano["rh__pose_coeffs"]
            # rh_tsl = raw_mano["rh__tsl"]
            # rh_pose_coeffs = quat_to_aa(rh_pose_coeffs).view(1, 48)
            # rh_pose_coeffs = torch.cat([rh_pose_coeffs, rh_tsl], dim=-1)

            raw_smplx = anno["raw_smplx"][frame_id]
            smplx_result.append(raw_smplx)
            rh_pose_coeffs = quat_to_aa(raw_smplx["right_hand_pose"]).view(1, 45)
            hand_pose.append(rh_pose_coeffs)
            

            obj_transf = anno["obj_transf"]
            _object_pose = []
            for obj in obj_id:
                p, q = obj_transf[obj][frame_id][:3, 3], rotmat_to_quat(obj_transf[obj][frame_id][:3, :3])
                _object_pose.append(np.concatenate([p, q], axis=-1))
            object_pose.append(_object_pose)

        # hand_pose = np.array(hand_pose)
        hand_pose = torch.cat(hand_pose, dim=0)
        object_pose = np.array(object_pose)

        smplx_data = {k: [] for k in smplx_result[0].keys()}
        for smplx_k in smplx_result[0].keys():
            for d in smplx_result:
                smplx_data[smplx_k].append(d[smplx_k])
        smplx_data = {k: torch.concat(v).to(self.device) for k, v in smplx_data.items()}

        smplx_results = self.smplx_layer(**smplx_data)

        mano_joints = torch.stack([
            smplx_results.joints[:, 21].detach(),
            smplx_results.joints[:, 40].detach(),
            smplx_results.joints[:, 41].detach(),
            smplx_results.joints[:, 42].detach(),
            smplx_results.vertices[:, 7706].detach(),  # reselect tip
            smplx_results.joints[:, 43].detach(),
            smplx_results.joints[:, 44].detach(),
            smplx_results.joints[:, 45].detach(),
            smplx_results.vertices[:, 7818].detach(),  # reselect tip
            smplx_results.joints[:, 46].detach(),
            smplx_results.joints[:, 47].detach(),
            smplx_results.joints[:, 48].detach(),
            smplx_results.vertices[:, 8046].detach(),  # reselect tip
            smplx_results.joints[:, 49].detach(),
            smplx_results.joints[:, 50].detach(),
            smplx_results.joints[:, 51].detach(),
            smplx_results.vertices[:, 7929].detach(),  # reselect tip
            smplx_results.joints[:, 52].detach(),
            smplx_results.joints[:, 53].detach(),
            smplx_results.joints[:, 54].detach(),
            smplx_results.vertices[:, 8096].detach(),  # reselect tip
        ], dim=1).numpy()

        mano_vertex = smplx_results.vertices[:, 7940:8718].detach().numpy()

        mano_rot_offset = self.dexhand.relative_rotation
        wrist_rot = rotmat_to_aa(smplx_results.transform_abs[:, 21, :3, :3].detach() @ torch.tensor(
            np.repeat(mano_rot_offset[None], smplx_results.transform_abs.shape[0], axis=0), device=self.device
        ))

        hand_pose = torch.cat([wrist_rot, hand_pose, smplx_results.joints[:, 21]], dim=-1).unsqueeze(1).numpy()
        # hand_pose = torch.cat([wrist_rot, smplx_results.joints[:, ], smplx_results.joints[:, 21]], dim=-1).unsqueeze(1).numpy()
        hand_shape = anno["raw_smplx"][frame_id_list[0]]["body_shape"][0, :10].numpy()
        # hand_shape = anno["raw_mano"][frame_id_list[0]]["rh__betas"].view(10).numpy()

        data = dict(
            data_dir=self.data_dir,
            hand_pose=hand_pose,
            object_pose=object_pose,
            extrinsics=extrinsics,
            obj_ids=obj_id,
            hand_shape=hand_shape,
            object_mesh_file=obj_mesh_path,
            capture_name=capture_name,
            mano_joints=mano_joints,
            mano_vertex=mano_vertex,
        )

        return data

    def available_index(self):
        available_index_list = []

        def intersection(lst1, lst2):
            begin = max(lst1[0], lst2[0])
            end = min(lst1[1], lst2[1])
            if begin < end:
                return [begin, end]
            else:
                assert False, f"no intersection between {lst1} and {lst2}"

        for seq_hash, idx in self.seq_hashes.items():
            program_filepath = os.path.join(
            self.data_dir,
            "program",
            "program_info",
            f"{os.path.splitext(os.path.split(self.data_pathes[idx])[1])[0]}.json",
            ) 

            with open(program_filepath, "r") as ifs:
                _program_info = json.load(ifs)
                stage = 0
                for k, v in _program_info.items():
                    seg_pair_def = eval(k)
                    left_hand_range = seg_pair_def[0]
                    right_hand_range = seg_pair_def[1]

                    if right_hand_range is not None and len(v["obj_list_rh"]) != 0:
                        if left_hand_range is None or max(left_hand_range[0], right_hand_range[0]) < min(left_hand_range[1], right_hand_range[1]):
                            available_index_list.append(seq_hash + '@' + str(stage))
                    
                    stage += 1

        return available_index_list
    
    def load_data(self):
        super().load_data()
        offset = torch.tensor([0., 0., 0.5], device=self.device)
        self.data["wrist_pos"][..., :3] += offset
        self.data["body_pos"][..., :3] += offset
        self.data["obj_pose"][..., :3] += offset

        for obj_id in self.data["obj_id"]:
            obj_usd_dir = Path(os.path.join(self.data_dir, "usd_objects", obj_id))
            filenames = list(obj_usd_dir.rglob("*.usd"))
            self.data["obj_usd"].append(str(filenames[0]))