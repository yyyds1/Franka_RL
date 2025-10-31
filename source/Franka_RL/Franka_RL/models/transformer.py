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
        
        # Sequence caching for temporal attention (multi-step history)
        self.max_history_steps = config.get("max_sequence_length", 100)
        self.history_buffer = None  # Will be initialized on first forward: [num_envs, max_seq_len, feature_dim]
        self.step_counters = None  # Per-environment step counter: [num_envs]
        self.enable_sequence_caching = config.get("enable_sequence_caching", True)  # Can be disabled for debugging
        
        print(f"[Transformer] Sequence caching: {'ENABLED' if self.enable_sequence_caching else 'DISABLED'}, "
              f"max_history_steps={self.max_history_steps}")

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

    def clear_cache(self, env_indices=None):
        """
        Clear sequence history buffer for specified environments.
        
        Args:
            env_indices: Indices of environments to reset. If None, reset all.
        """
        if self.history_buffer is None:
            return
        
        if env_indices is None:
            # Reset all environments
            self.history_buffer.zero_()
            if self.step_counters is not None:
                self.step_counters.zero_()
        else:
            # Reset specific environments
            if len(env_indices) > 0:
                self.history_buffer[env_indices] = 0
                if self.step_counters is not None:
                    self.step_counters[env_indices] = 0
    
    def forward(self, input_dict, **kwargs):
        """
        Forward pass with optional sequence caching for temporal attention.
        
        If sequence caching is enabled, this accumulates observations over time
        and feeds the history to the Transformer to enable multi-step attention.
        """
        # Extract current step features: [batch_size, feature_dim]
        current_features = self.get_extracted_features(input_dict)
        
        # If sequence caching is disabled, use single-step processing (original behavior)
        if not self.enable_sequence_caching:
            if hasattr(self, "output_model"):
                output = self.output_model(current_features)
            else:
                output = current_features
            return output
        
        # === Sequence Caching Logic ===
        batch_size = current_features.shape[0]
        feature_dim = current_features.shape[-1]
        device = current_features.device
        
        # Initialize buffers ONLY on first forward pass
        # CRITICAL: Buffer should only store per-environment history, not per-timestep!
        # During rollout: batch_size = num_envs (e.g., 4096)
        # During PPO update: batch_size = num_envs * num_steps (e.g., 98304)
        # We should ONLY cache during rollout, NOT during update!
        if self.history_buffer is None:
            # Initialize with the number of environments (first forward call)
            self.history_buffer = torch.zeros(
                batch_size, self.max_history_steps, feature_dim,
                device=device, dtype=current_features.dtype,
                requires_grad=False  # Explicitly disable gradient to avoid inference mode issues
            )
            self.step_counters = torch.zeros(batch_size, dtype=torch.long, device=device)
            self.num_envs = batch_size  # Remember the actual number of environments
            print(f"[Transformer] Initialized history buffer for {batch_size} environments, "
                  f"shape {self.history_buffer.shape}, memory: {self.history_buffer.numel() * 4 / 1024**2:.1f} MB")
        
        # CRITICAL FIX: During PPO update, batch_size = num_envs * num_steps
        # We should NOT update buffer during training, only during rollout!
        # Detect if we're in update phase by checking if batch_size is much larger than num_envs
        is_rollout = (batch_size == self.num_envs)
        
        if not is_rollout:
            # During PPO update: Don't use sequence caching!
            # Just process current_features directly without temporal context
            # This is acceptable because PPO uses advantage estimation, not temporal modeling
            if hasattr(self, "output_model"):
                output = self.output_model(current_features)
            else:
                output = current_features
            return output
        
        # Update circular buffer for each environment
        # For efficiency, we use modulo indexing instead of rolling
        write_indices = self.step_counters % self.max_history_steps  # [batch_size]
        
        # Memory-efficient update: Use scatter_ directly on detached buffer
        # We must be careful about inference mode and gradient graphs
        # Strategy: Temporarily work with .data (raw storage) to bypass autograd
        indices_expanded = write_indices.view(batch_size, 1, 1).expand(-1, 1, feature_dim)
        
        # Direct in-place update using .data to avoid creating copies
        # This bypasses autograd tracking and is safe since buffer should never require grad
        with torch.no_grad():
            self.history_buffer.scatter_(1, indices_expanded, current_features.unsqueeze(1))
        
        # Increment step counters (also in no_grad to avoid tracking)
        with torch.no_grad():
            self.step_counters.add_(1)  # In-place add
        
        # ========== VECTORIZED SEQUENCE CONSTRUCTION ==========
        # For efficiency with large batch sizes (e.g., 4096), we use vectorized operations
        # instead of Python loops to construct and pad sequences
        
        # Compute valid lengths for all environments
        # valid_len = min(step_counter, max_history_steps)
        valid_lens = torch.clamp(self.step_counters, max=self.max_history_steps)  # [batch_size]
        max_valid_len = valid_lens.max().item()
        
        # For simplicity and efficiency, use the buffer as-is without reordering
        # This means we use the most recent max_valid_len steps for each environment
        # Reordering for circular buffer can be skipped in early training when buffer not full
        
        # Take the most recent valid_len steps from each environment
        # For now, use first valid_len steps (oldest to newest order)
        # Shape: [batch_size, max_valid_len, feature_dim]
        # Use detach() to break gradient graph, but avoid clone() to save memory
        # The slice creates a view (no copy), permute also creates a view
        # Only .contiguous() at the end will create a copy if needed
        history_batch = self.history_buffer[:, :max_valid_len, :].detach()
        
        # Create attention mask: True for padding (invalid), False for valid
        # Shape: [batch_size, max_valid_len]
        seq_indices = torch.arange(max_valid_len, device=device).unsqueeze(0).expand(batch_size, -1)
        mask_batch = seq_indices >= valid_lens.unsqueeze(1)  # True where index >= valid_len
        
        # Transpose to transformer format: [seq_len, batch, feature_dim]
        # .contiguous() will make a copy only if needed for the transformer
        history_batch = history_batch.permute(1, 0, 2).contiguous()
        
        # Expand mask for multi-head attention: [batch*num_heads, seq_len, seq_len]
        # The mask format: True means "mask this position" (invalid)
        # We create a causal mask + padding mask
        attn_mask = mask_batch.unsqueeze(1).expand(-1, max_valid_len, -1)  # [batch, seq, seq]
        attn_mask = torch.repeat_interleave(attn_mask, self.config.num_heads, dim=0)  # [batch*heads, seq, seq]
        
        # Debug logging (only print occasionally to avoid spam)
        if hasattr(self, '_forward_call_count'):
            self._forward_call_count += 1
        else:
            self._forward_call_count = 0
        
        # Log every 1000 steps to track sequence length growth
        if self._forward_call_count % 1000 == 0:
            avg_valid_len = sum(min(sc.item(), self.max_history_steps) for sc in self.step_counters) / batch_size
            max_seq = max(min(sc.item(), self.max_history_steps) for sc in self.step_counters)
            min_seq = min(min(sc.item(), self.max_history_steps) for sc in self.step_counters)
            print(f"[Transformer] Step {self._forward_call_count}: avg_seq_len={avg_valid_len:.1f}/{self.max_history_steps} "
                  f"(min={min_seq}, max={max_seq}), batch_size={batch_size}, feature_dim={feature_dim}")
        
        # Apply Transformer encoder
        # Output shape: [seq_len, batch, feature_dim]
        if self.use_rope:
            transformer_output = self.seqTransEncoder(history_batch, mask=attn_mask)
        else:
            # Add positional encoding for traditional transformer
            history_batch = self.sequence_pos_encoder(history_batch)
            transformer_output = self.seqTransEncoder(history_batch, mask=attn_mask)
        
        # Extract output corresponding to the most recent observation for each environment
        # Since we use history_buffer[:, :max_valid_len, :] without reordering,
        # the most recent observation is at index (valid_lens - 1) for each environment
        # Shape: transformer_output is [seq_len, batch, feature_dim]
        
        # For efficiency, we can use the last valid position for each environment
        # Create indices: [batch_size] with values (valid_len - 1)
        last_valid_indices = (valid_lens - 1).clamp(min=0)  # [batch_size]
        
        # Gather from transformer_output using advanced indexing
        # We need output at position last_valid_indices[b] for each batch b
        batch_indices = torch.arange(batch_size, device=device)
        output = transformer_output[last_valid_indices, batch_indices, :]  # [batch, feature_dim]
        
        # Pass through output model if present
        if hasattr(self, "output_model"):
            output = self.output_model(output)
        
        return output
