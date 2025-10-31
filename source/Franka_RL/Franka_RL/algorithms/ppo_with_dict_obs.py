import torch
from rsl_rl.algorithms import PPO
from rsl_rl.storage import RolloutStorage


class PPOWithDictObs(PPO):
    """扩展 PPO 以支持字典格式的观测和分离的 actor/critic 观测"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init_storage(self, training_type, num_envs, num_transitions_per_env, obs_dict, actions_shape):
        """
        初始化存储，传递字典格式的观测给 RolloutStorage
        
        Args:
            training_type: "rl" 或 "distillation"
            num_envs: 环境数量
            num_transitions_per_env: 每个环境的转换数
            obs_dict: 观测字典 {"policy": tensor, "critic": tensor, ...}
            actions_shape: 动作形状（可以是张量或元组）
        """
        # 处理动作形状
        if isinstance(actions_shape, torch.Tensor):
            actions_shape_tuple = tuple(actions_shape.shape)
        elif isinstance(actions_shape, int):
            actions_shape_tuple = (actions_shape,)
        elif isinstance(actions_shape, (tuple, list)):
            actions_shape_tuple = tuple(actions_shape)
        else:
            actions_shape_tuple = (actions_shape,)
        
        # 确保 obs_dict 中的值都是张量（带 batch 维度）
        obs_with_batch = {}
        for key, value in obs_dict.items():
            if isinstance(value, torch.Tensor):
                # 如果是 1D 张量（只有特征维度），添加 batch 维度
                if value.dim() == 1:
                    obs_with_batch[key] = value.unsqueeze(0).expand(num_envs, -1)
                else:
                    obs_with_batch[key] = value
            elif isinstance(value, int):
                # 如果是整数，创建对应形状的零张量
                obs_with_batch[key] = torch.zeros(num_envs, value, device=self.device)
            elif isinstance(value, (tuple, list)):
                # 如果是元组/列表，创建对应形状的零张量
                obs_with_batch[key] = torch.zeros(num_envs, *value, device=self.device)
            else:
                raise ValueError(f"Unsupported obs type for key '{key}': {type(value)}")
        
        # Create storage with dict observations
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            obs_with_batch,          # ✅ 字典格式！
            actions_shape_tuple,
            self.device,
        )
    
    def act(self, obs, privileged_obs=None):
        """Sample actions from the policy with separate actor and critic observations.
    
        Args:
            obs: Policy observations (for actor) - 可以是张量或字典
            privileged_obs: Privileged observations (for critic) - 可以是张量或字典
    
        Returns:
            actions: Sampled actions
        """
        # 将观测转换为字典格式（如果还不是）
        if not isinstance(obs, dict):
            obs_dict = {"policy": obs}
        else:
            obs_dict = obs
        
        if privileged_obs is not None and not isinstance(privileged_obs, dict):
            privileged_obs_dict = {"critic": privileged_obs}
        else:
            privileged_obs_dict = privileged_obs if privileged_obs is not None else obs_dict
        
        # 存储观测（字典格式）
        self.transition.observations = obs_dict
        self.transition.privileged_observations = privileged_obs_dict
        
        # 从字典中提取张量用于策略计算
        obs_tensor = obs if isinstance(obs, torch.Tensor) else obs_dict["policy"]
        privileged_obs_tensor = privileged_obs if isinstance(privileged_obs, torch.Tensor) else privileged_obs_dict.get("critic", obs_tensor)
        # Sample actions and compute distribution statistics
        actions = self.policy.act(obs_tensor)
        # The policy's distribution is updated inside policy.act()
        action_log_prob = self.policy.get_actions_log_prob(actions).detach()
        action_mean = self.policy.action_mean.detach()
        action_std = self.policy.action_std.detach()

        self.transition.actions = actions.detach()
        self.transition.actions_log_prob = action_log_prob
        self.transition.action_mean = action_mean
        self.transition.action_sigma = action_std

        # Evaluate value using critic (使用 privileged obs)
        values = self.policy.evaluate(privileged_obs_tensor).detach()
        self.transition.values = values

        # After distribution created in policy.act(), we can clamp std to stabilize KL
        if hasattr(self.policy, 'distribution') and self.policy.distribution is not None:
            try:
                # Clamp extremely small or large std to avoid numerical explosions
                if hasattr(self.policy, 'action_std'):
                    std = self.policy.action_std
                    clamped_std = torch.clamp(std, min=1e-6, max=10.0)
                    # If clamping changed values, rebuild distribution with same mean
                    if not torch.allclose(std, clamped_std):
                        mean = self.policy.action_mean
                        from torch.distributions import Normal
                        self.policy.distribution = Normal(mean, clamped_std)
            except Exception:
                pass
        return actions
    
    def compute_returns(self, last_privileged_obs):
        """
        计算回报，使用 privileged observations 计算最后一步的价值
        
        Args:
            last_privileged_obs: 最后一步的 privileged observations
        """
        # 使用 privileged observations 计算最后一步的价值
        last_values = self.policy.evaluate(last_privileged_obs).detach()
        self.storage.compute_returns(
            last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        )
