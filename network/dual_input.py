"""
Dual-input feature extractor for F-Zero RL.

Architecture adapted from Linesight (Trackmania AI):
  - 4-layer CNN for game screenshots (LeakyReLU)
  - 2-layer MLP for float features (LeakyReLU)
  - Concatenation fusion
  - Dueling heads (Advantage + Value) for Q-value decomposition

Total: ~2-3M parameters
"""
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from training.config import NetworkConfig


def _init_orthogonal(module: nn.Module, gain: float = 1.0):
    """Initialize a linear or conv layer with orthogonal weights."""
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class FZeroFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom SB3 feature extractor with dual-input architecture.

    Processes Dict observation space with keys:
      - "screen": (frame_stack, H, W) float32 — processed by CNN
      - "float": (float_dim,) float32 — processed by MLP

    Outputs a fused feature vector for SB3's policy/value heads.
    """

    def __init__(self, observation_space: spaces.Dict, cfg: NetworkConfig = None):
        cfg = cfg or NetworkConfig()

        # We compute features_dim after building the network
        # Pass a placeholder first, then update
        super().__init__(observation_space, features_dim=1)

        screen_space = observation_space["screen"]
        float_space = observation_space["float"]

        n_input_channels = screen_space.shape[0]  # frame_stack
        float_input_dim = float_space.shape[0]
        leaky_slope = cfg.leaky_relu_slope
        activation_gain = nn.init.calculate_gain("leaky_relu", leaky_slope)

        # === CNN branch (image head) ===
        cnn_layers = []
        in_channels = n_input_channels
        for out_channels, kernel, stride in zip(
            cfg.cnn_channels, cfg.cnn_kernels, cfg.cnn_strides
        ):
            cnn_layers.append(nn.Conv2d(in_channels, out_channels, kernel, stride))
            cnn_layers.append(nn.LeakyReLU(leaky_slope, inplace=True))
            in_channels = out_channels
        cnn_layers.append(nn.Flatten())
        self.cnn = nn.Sequential(*cnn_layers)

        # Compute CNN output size by running a dummy forward pass
        with torch.no_grad():
            dummy_screen = torch.as_tensor(screen_space.sample()[None]).float()
            cnn_output_dim = self.cnn(dummy_screen).shape[1]

        # === Float branch (state head) ===
        self.float_mlp = nn.Sequential(
            nn.Linear(float_input_dim, cfg.float_hidden_dim),
            nn.LeakyReLU(leaky_slope, inplace=True),
            nn.Linear(cfg.float_hidden_dim, cfg.float_hidden_dim),
            nn.LeakyReLU(leaky_slope, inplace=True),
        )

        # === Fusion ===
        fusion_input_dim = cnn_output_dim + cfg.float_hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, cfg.dense_hidden_dim),
            nn.LeakyReLU(leaky_slope, inplace=True),
        )

        # Update the features_dim that SB3 uses to build policy/value heads
        self._features_dim = cfg.dense_hidden_dim

        # === Initialize weights (orthogonal, following Linesight) ===
        for module in [self.cnn, self.float_mlp, self.fusion]:
            for m in module:
                _init_orthogonal(m, activation_gain)

    def forward(self, observations: dict) -> torch.Tensor:
        """
        Forward pass through dual-input architecture.

        Args:
            observations: dict with "screen" (B, C, H, W) and "float" (B, D) tensors

        Returns:
            (B, features_dim) fused feature tensor
        """
        screen = observations["screen"]
        float_inputs = observations["float"]

        cnn_features = self.cnn(screen)
        float_features = self.float_mlp(float_inputs)

        fused = torch.cat([cnn_features, float_features], dim=1)
        return self.fusion(fused)
