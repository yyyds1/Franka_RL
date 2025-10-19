import json
import os
import pickle
from functools import lru_cache

import numpy as np
import torch

from .base import DexhandData
from .decorators import register_dataset

from .transform import quat_to_aa, rotmat_to_quat

@register_dataset("OakInkv2_rh")
class OakInkv2Dataset(DexhandData):
    def __init__(
        self, 
        *, 
        data_dir = "dataset/OakInk-v2", 
        skip: int = 2,
        device = 'cpu', 
        dexhand=None,
        **kwargs,
    ):
        super().__init__(data_dir=data_dir, device=device, dexhand=dexhand)

        pathes = os.listdir(os.path.join(data_dir, "anno_preview"))
        pathes = [os.path.join(data_dir, "anno_preview", p) for p in pathes]
        pathes.sort(key=lambda x: x.split("/")[-1])
        self.data_pathes = pathes
        # * We use the first 5 digits of hash as the index
        self.seq_hashes = {os.path.split(p)[-1].split("_")[5][:5]: i for i, p in enumerate(pathes)}

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

        extrinsics = anno["cam_extr"][anno["cam_selection"][0]][frame_id_list[0]]
        extrinsics = torch.tensor(extrinsics, device=self.device)

        object_list = anno["obj_list"]
        obj_id = [program_info_selected["obj_list_rh"][0]]
        obj_mesh_path = []
        for obj in obj_id:
            obj_filedir = os.path.join(self.data_dir, "object_preview", "align_ds", obj)
            candidate_list = [el for el in os.listdir(obj_filedir) if os.path.splitext(el)[-1] in [".obj", ".ply"]]
            assert len(candidate_list) == 1
            obj_filename = candidate_list[0]
            obj_filepath = os.path.join(obj_filedir, obj_filename)
            obj_mesh_path.append(obj_filepath)

        length = len(frame_id_list)

        hand_pose = []
        hand_shape = []
        object_pose = []

        for frame_id in frame_id_list:

            raw_mano = anno["raw_mano"][frame_id]
            rh_pose_coeffs = raw_mano["rh__pose_coeffs"]
            rh_tsl = raw_mano["rh__tsl"]
            rh_betas = raw_mano["rh__betas"]
            rh_pose_coeffs = quat_to_aa(rh_pose_coeffs).view(1, 48)
            rh_pose_coeffs = torch.cat([rh_pose_coeffs, rh_tsl], dim=-1)

            hand_pose.append(rh_pose_coeffs)
            hand_shape.append(rh_betas)

            obj_transf = anno["obj_transf"]
            _object_pose = []
            for obj in obj_id:
                p, q = obj_transf[obj][frame_id][:3, 3], rotmat_to_quat(obj_transf[obj][frame_id][:3, :3])
                _object_pose.append(torch.cat([p, q], dim=-1))
            object_pose.append(_object_pose)
        
        hand_pose = torch.cat(hand_pose, dim=0)
        hand_shape = torch.cat(hand_shape, dim=0)
        object_pose = torch.tensor(object_pose, device=self.device)

        data = dict(
            hand_pose=hand_pose,
            object_pose=object_pose,
            extrinsics=extrinsics,
            obj_ids=obj_id,
            hand_shape=hand_shape,
            object_mesh_file=obj_mesh_path,
            capture_name=capture_name,
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

        for seq_hash, idx in self.seq_hashes.items:
            program_filepath = os.path.join(
            self.data_dir,
            "program",
            "program_info",
            f"{os.path.splitext(os.path.split(self.data_pathes[idx])[1])[0]}.json",
            ) 

            with open(program_filepath, "r") as ifs:
                _program_info = json.load(ifs)
                stage = 0
                for k in _program_info.keys():
                    seg_pair_def = eval(k)
                    left_hand_range = seg_pair_def[0]
                    right_hand_range = seg_pair_def[1]

                    if right_hand_range is not None:
                        available_index_list.append(seq_hash + str(stage))
                    
                    stage += 1

        return available_index_list