# import zarr
# # 打开 Zarr 数组（假设数据集在当前目录下）
# z = zarr.open('./dataset/data/eepose', mode='r')

# # 读取整个数组
# data = z[:]
# pass
# # 或者读取部分数据（支持切片）
# subset = z[0:100, 50:200]  # 例如读取前100行，第50到199列

# print("Shape:", z.shape)
# print("Chunks:", z.chunks)
# print("Dtype:", z.dtype)
# print("Data:", data)

# import argparse

# from isaaclab.app import AppLauncher

# # add argparse arguments
# parser = argparse.ArgumentParser(description="This script demonstrates different single-arm manipulators.")
# # append AppLauncher cli args
# AppLauncher.add_app_launcher_args(parser)
# # parse the arguments
# args_cli = parser.parse_args()

# # launch omniverse app
# app_launcher = AppLauncher(args_cli)
# simulation_app = app_launcher.app

# import torch
# import Franka_RL.models.pointtransformer as pointtransformer

# ckpt = torch.load("./ckpt/model.pth", map_location='cpu')

# submodule_state_dict = {}
# for key in ckpt["model"].keys():
#     if(key.startswith("module.eps_model.scene_model.")):
#         new_key = key.replace("module.eps_model.scene_model.", "", 1)
#         submodule_state_dict[new_key] = ckpt["model"][key]

# pointencoder = pointtransformer.pointtransformer_enc_repro()
# pointencoder.load_state_dict(submodule_state_dict, strict=True)

# pass