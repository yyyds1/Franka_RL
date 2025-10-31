from copy import copy

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from hydra.utils import instantiate

from Franka_RL.utils import model_utils
from Franka_RL.models.rope import create_rope_transformer_encoder

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.pe = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        x = x + self.pe[:, : x.shape[1], : x.shape[2]]
        return x


class Transformer(nn.Module):
    def __init__(self, config, num_out: int):
        super().__init__()
        self.config = config

        self.mask_keys = {}
        input_models = {}
        self.obs_slice = {}

        for input_key, input_config in config.input_models.items():
            input_models[input_key] = instantiate(input_config)
            self.mask_keys[input_key] = input_config.config.get("mask_key", None)
            self.obs_slice[input_config.config.obs_key] = slice(input_config.config.slice_start_idx, input_config.config.slice_end_idx, input_config.config.get("slice_step", 1))

        self.input_models = nn.ModuleDict(input_models)
        self.feature_size = self.config.transformer_token_size * len(input_models)

        # Transformer layers with RoPE support
        self.use_rope = config.get("use_rope", False)
        
        if self.use_rope:
            # Use RoPE - no additional positional encoding needed
            print(f"[Transformer] Using RoPE with theta={config.get('rope_theta', 10000.0)}, "
                  f"max_seq_len={config.get('max_sequence_length', 2048)}")
            self.seqTransEncoder = create_rope_transformer_encoder(
                d_model=config.latent_dim,
                nhead=config.num_heads,
                num_layers=config.num_layers,
                dim_feedforward=config.ff_size,
                dropout=config.dropout,
                activation=config.activation if isinstance(config.activation, str) else "relu",
                rope_theta=config.get("rope_theta", 10000.0),
                max_seq_len=config.get("max_sequence_length", 2048),
            )
        else:
            # Use traditional positional encoding
            print(f"[Transformer] Using traditional PositionalEncoding")
            self.sequence_pos_encoder = PositionalEncoding(config.latent_dim)
            seqTransEncoderLayer = nn.TransformerEncoderLayer(
                d_model=config.latent_dim,
                nhead=config.num_heads,
                dim_feedforward=config.ff_size,
                dropout=config.dropout,
                activation=model_utils.get_activation_func(config.activation, return_type="functional"),
            )
            self.seqTransEncoder = nn.TransformerEncoder(
                seqTransEncoderLayer, num_layers=config.num_layers
            )

        if config.get("output_model", None) is not None:
            self.output_model = instantiate(config.output_model)

    def get_extracted_features(self, input_dict):
        # 检查是否是 TensorDict
        is_tensordict = hasattr(input_dict, 'batch_size') and hasattr(input_dict, 'keys')
        
        # 处理 TensorDict、字典或张量输入
        if is_tensordict or isinstance(input_dict, dict):
            # 获取第一个值来确定 batch_size 和 device
            first_value = next(iter(input_dict.values()))
            
            # 确保值是 2D 张量 (batch, feature)
            if len(first_value.shape) == 1:
                raise ValueError(
                    f"TensorDict contains 1D tensors. Expected 2D tensors with shape (batch, feature). "
                    f"First value shape: {first_value.shape}. "
                    f"Please check how observations are extracted from the environment."
                )
            else:
                batch_size = first_value.shape[0]
                device = first_value.device
                # 将字典的所有值拼接成一个张量
                input_tensor = torch.cat([v for v in input_dict.values()], dim=-1)
            
           
            original_input_dict = input_dict
            is_dict_input = True
        else:
           
            if not hasattr(input_dict, 'shape'):
                raise TypeError(f"input_dict is not a tensor or dict, got {type(input_dict)}")
            
            if len(input_dict.shape) == 1:
                input_dict = input_dict.unsqueeze(0)
            
            batch_size = input_dict.shape[0]
            device = input_dict.device
            input_tensor = input_dict
            original_input_dict = None
            is_dict_input = False
        
        cat_obs = []
        cat_mask = []

        for model_name, input_model in self.input_models.items():
            input_key = input_model.config.obs_key
            key_obs = input_tensor[:, self.obs_slice[input_key]]

            if input_model.config.get("operations", None) is not None:
                for operation in input_model.config.get("operations", []):
                    if operation.type == "permute":
                        key_obs = key_obs.permute(*operation.new_order)
                    elif operation.type == "reshape":
                        new_shape = copy(operation.new_shape)
                        if new_shape[0] == "batch_size":
                            new_shape[0] = batch_size
                        key_obs = key_obs.reshape(*new_shape)
                    elif operation.type == "squeeze":
                        key_obs = key_obs.squeeze(dim=operation.squeeze_dim)
                    elif operation.type == "unsqueeze":
                        key_obs = key_obs.unsqueeze(dim=operation.unsqueeze_dim)
                    elif operation.type == "expand":
                        key_obs = key_obs.expand(*operation.expand_shape)
                    elif operation.type == "positional_encoding":
                        key_obs = self.sequence_pos_encoder(key_obs)
                    elif operation.type == "encode":
                        key_obs = {input_key: key_obs}
                        key_obs = input_model(key_obs)
                    elif operation.type == "mask_multiply":
                        
                        mask_source = original_input_dict if is_dict_input else input_tensor
                        mask_data = mask_source[self.mask_keys[model_name]]
                        num_mask_dims = len(mask_data.shape)
                        num_obs_dims = len(key_obs.shape)
                        extra_needed_dims = num_obs_dims - num_mask_dims
                        key_obs = key_obs * mask_data.view(
                            *mask_data.shape,
                            *((1,) * extra_needed_dims),
                        )
                    elif operation.type == "mask_multiply_concat":
                        
                        mask_source = original_input_dict if is_dict_input else input_tensor
                        mask_data = mask_source[self.mask_keys[model_name]]
                        num_mask_dims = len(mask_data.shape)
                        num_obs_dims = len(key_obs.shape)
                        extra_needed_dims = num_obs_dims - num_mask_dims
                        key_obs = key_obs * mask_data.view(
                            *mask_data.shape,
                            *((1,) * extra_needed_dims),
                        )
                        key_obs = torch.cat(
                            [
                                key_obs,
                                mask_data.view(
                                    *mask_data.shape,
                                    *((1,) * extra_needed_dims),
                                ),
                            ],
                            dim=-1,
                        )
                    elif operation.type == "concat_obs":
                        # 使用原始字典（如果可用）来访问其他观测
                        obs_source = original_input_dict if is_dict_input else input_tensor
                        to_add_obs = obs_source[operation.obs_key]
                        if len(to_add_obs.shape) != len(key_obs.shape):
                            to_add_obs = to_add_obs.unsqueeze(1).expand(
                                to_add_obs.shape[0],
                                key_obs.shape[1],
                                to_add_obs.shape[-1],
                            )
                        key_obs = torch.cat([key_obs, to_add_obs], dim=-1)
                    else:
                        raise NotImplementedError(
                            f"Operation {operation} not implemented"
                        )
            else:
                # key_obs = {input_key: key_obs}
                key_obs = input_model(key_obs)

            if len(key_obs.shape) == 2:
                # Add a sequence dimension
                key_obs = key_obs.unsqueeze(1)

            cat_obs.append(key_obs)

            if self.mask_keys[model_name] is not None:
                # 使用原始字典（如果可用）来访问掩码
                if is_dict_input:
                    key_mask = original_input_dict[self.mask_keys[model_name]]
                else:
                    # 如果是张量输入但需要掩码，创建全有效的掩码
                    # 注意：这种情况下可能需要从其他地方传入掩码信息
                    key_mask = torch.ones(
                        batch_size,
                        key_obs.shape[1],
                        dtype=torch.bool,
                        device=device,
                    )
                # Our mask is 1 for valid and 0 for invalid
                # The transformer expects the mask to be 0 for valid and 1 for invalid
                key_mask = key_mask.logical_not()
            else:
                key_mask = torch.zeros(
                    batch_size,
                    key_obs.shape[1],
                    dtype=torch.bool,
                    device=device,
                )
            cat_mask.append(key_mask)

        # Concatenate all the features
        cat_obs = torch.cat(cat_obs, dim=1)
        cat_mask = torch.cat(cat_mask, dim=1)

        # obs creation works in batch_first but transformer expects seq_len first
        cat_obs = cat_obs.permute(1, 0, 2).contiguous()  # [seq_len, bs, d]

        cur_mask = cat_mask.unsqueeze(1).expand(-1, cat_obs.shape[0], -1)
        cur_mask = torch.repeat_interleave(cur_mask, self.config.num_heads, dim=0)

        output = self.seqTransEncoder(cat_obs, mask=cur_mask)[0]  # [bs, d]

        return output

    def forward(self, input_dict, **kwargs):
        # 使用 get_extracted_features 提取特征
        output = self.get_extracted_features(input_dict)
        
        # 通过输出模型（如果有）
        if hasattr(self, "output_model"):
            output = self.output_model(output)
        
        return output
