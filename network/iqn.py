"""
Implicit Quantile Network (IQN) with branching dueling architecture.

Adapted from Linesight (Trackmania AI) for F-Zero RL.
Supports MultiDiscrete action space via branching heads — one advantage
stream per action dimension, sharing a single value stream.

References:
  - IQN paper: https://arxiv.org/abs/1806.06923
  - Dueling DQN: https://arxiv.org/abs/1511.06581
  - Branching DQN: https://arxiv.org/abs/1711.08946
"""
import math

import torch
import torch.nn as nn
import numpy as np

from training.config import NetworkConfig, IQNConfig
from env.actions import ACTION_DIMS


def _init_orthogonal(module: nn.Module, gain: float = 1.0):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class IQNNetwork(nn.Module):
    """IQN with branching dueling heads for MultiDiscrete([3,3,2,2,2]).

    Architecture:
      CNN(screen) + MLP(float) → concat → IQN quantile embedding (Hadamard) →
      → shared V_head(1) + per-dimension A_heads([3,3,2,2,2])
      → Q_i = V + A_i - mean(A_i) for each dimension
    """

    def __init__(self, float_input_dim: int, net_cfg: NetworkConfig = None,
                 iqn_cfg: IQNConfig = None):
        super().__init__()
        net_cfg = net_cfg or NetworkConfig()
        iqn_cfg = iqn_cfg or IQNConfig()

        self.iqn_embedding_dim = iqn_cfg.iqn_embedding_dim
        self.action_dims = ACTION_DIMS  # [3, 3, 2, 2, 2]
        leaky_slope = net_cfg.leaky_relu_slope
        activation_gain = nn.init.calculate_gain("leaky_relu", leaky_slope)

        # CNN branch
        cnn_layers = []
        in_ch = 4  # frame_stack
        for out_ch, k, s in zip(net_cfg.cnn_channels, net_cfg.cnn_kernels, net_cfg.cnn_strides):
            cnn_layers.append(nn.Conv2d(in_ch, out_ch, k, s))
            cnn_layers.append(nn.LeakyReLU(leaky_slope, inplace=True))
            in_ch = out_ch
        cnn_layers.append(nn.Flatten())
        self.cnn = nn.Sequential(*cnn_layers)

        # Compute CNN output dim
        with torch.no_grad():
            dummy = torch.zeros(1, 4, 120, 160)
            cnn_out_dim = self.cnn(dummy).shape[1]

        # Float branch
        self.float_mlp = nn.Sequential(
            nn.Linear(float_input_dim, net_cfg.float_hidden_dim),
            nn.LeakyReLU(leaky_slope, inplace=True),
            nn.Linear(net_cfg.float_hidden_dim, net_cfg.float_hidden_dim),
            nn.LeakyReLU(leaky_slope, inplace=True),
        )

        self.dense_input_dim = cnn_out_dim + net_cfg.float_hidden_dim
        hidden_half = net_cfg.dense_hidden_dim // 2

        # IQN quantile embedding: cosine basis → linear → LeakyReLU
        self.iqn_fc = nn.Sequential(
            nn.Linear(self.iqn_embedding_dim, self.dense_input_dim),
            nn.LeakyReLU(leaky_slope, inplace=True),
        )

        # Shared value head
        self.V_head = nn.Sequential(
            nn.Linear(self.dense_input_dim, hidden_half),
            nn.LeakyReLU(leaky_slope, inplace=True),
            nn.Linear(hidden_half, 1),
        )

        # Branching advantage heads — one per action dimension
        self.A_heads = nn.ModuleList()
        for dim_size in self.action_dims:
            head = nn.Sequential(
                nn.Linear(self.dense_input_dim, hidden_half),
                nn.LeakyReLU(leaky_slope, inplace=True),
                nn.Linear(hidden_half, dim_size),
            )
            self.A_heads.append(head)

        self._init_weights(activation_gain)

    def _init_weights(self, activation_gain: float):
        for module in [self.cnn, self.float_mlp]:
            for m in module:
                _init_orthogonal(m, activation_gain)

        # IQN FC gets sqrt(2) * gain because cosine has variance 1/2
        _init_orthogonal(self.iqn_fc[0], math.sqrt(2) * activation_gain)

        # V and A hidden layers get activation gain, output layers get gain=1
        for m in self.V_head[:-1]:
            _init_orthogonal(m, activation_gain)
        _init_orthogonal(self.V_head[-1], 1.0)

        for head in self.A_heads:
            for m in head[:-1]:
                _init_orthogonal(m, activation_gain)
            _init_orthogonal(head[-1], 1.0)

    def forward(self, screen: torch.Tensor, float_inputs: torch.Tensor,
                num_quantiles: int, tau: torch.Tensor = None):
        """
        Forward pass through branching IQN.

        Args:
            screen: (batch_size, C, H, W) float32
            float_inputs: (batch_size, float_dim) float32
            num_quantiles: N or N' from IQN paper
            tau: optional (batch_size * num_quantiles, 1) quantile values

        Returns:
            Q_branches: list of 5 tensors, each (batch_size * num_quantiles, dim_i)
            tau: (batch_size * num_quantiles, 1) quantile values used
        """
        batch_size = screen.shape[0]
        device = screen.device

        # Feature extraction
        cnn_out = self.cnn(screen)
        float_out = self.float_mlp(float_inputs)
        features = torch.cat([cnn_out, float_out], dim=1)  # (B, dense_input_dim)

        # Sample tau if not provided (symmetric sampling like Linesight)
        if tau is None:
            half = num_quantiles // 2
            tau = (
                torch.arange(half, device=device, dtype=torch.float32)
                .repeat_interleave(batch_size).unsqueeze(1)
                + torch.rand(batch_size * half, 1, device=device)
            ) / num_quantiles
            tau = torch.cat([tau, 1 - tau], dim=0)  # symmetric

        # Cosine quantile embedding
        i_pi = torch.arange(1, self.iqn_embedding_dim + 1, device=device).float() * math.pi
        cos_embedding = torch.cos(tau * i_pi)  # (B*N, embedding_dim)
        quantile_features = self.iqn_fc(cos_embedding)  # (B*N, dense_input_dim)

        # Hadamard product of features and quantile embedding
        features_expanded = features.repeat(num_quantiles, 1)  # (B*N, dense_input_dim)
        combined = features_expanded * quantile_features  # (B*N, dense_input_dim)

        # Shared value stream
        V = self.V_head(combined)  # (B*N, 1)

        # Branching advantage streams
        Q_branches = []
        for head in self.A_heads:
            A = head(combined)  # (B*N, dim_i)
            Q = V + A - A.mean(dim=-1, keepdim=True)
            Q_branches.append(Q)

        return Q_branches, tau
