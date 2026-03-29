"""Tests for the dual-input neural network architecture."""
import pytest
import torch
import numpy as np
from gymnasium import spaces

from network.dual_input import FZeroFeatureExtractor
from training.config import NetworkConfig, EnvConfig


@pytest.fixture
def obs_space():
    """Create a Dict observation space matching FZeroEnv."""
    cfg = EnvConfig()
    from env.observations import FloatFeatureBuilder
    fb = FloatFeatureBuilder(cfg)
    return spaces.Dict({
        "screen": spaces.Box(
            low=0.0, high=1.0,
            shape=(cfg.frame_stack, cfg.screen_height, cfg.screen_width),
            dtype=np.float32,
        ),
        "float": spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(fb.dim,),
            dtype=np.float32,
        ),
    })


class TestFZeroFeatureExtractor:
    def test_creation(self, obs_space):
        extractor = FZeroFeatureExtractor(obs_space, cfg=NetworkConfig())
        assert extractor.features_dim > 0

    def test_features_dim_equals_dense_hidden(self, obs_space):
        cfg = NetworkConfig()
        extractor = FZeroFeatureExtractor(obs_space, cfg=cfg)
        assert extractor.features_dim == cfg.dense_hidden_dim

    def test_forward_pass_shape(self, obs_space):
        extractor = FZeroFeatureExtractor(obs_space)
        batch_size = 4

        dummy_obs = {
            "screen": torch.randn(batch_size, *obs_space["screen"].shape),
            "float": torch.randn(batch_size, *obs_space["float"].shape),
        }
        output = extractor(dummy_obs)
        assert output.shape == (batch_size, extractor.features_dim)

    def test_forward_no_nan(self, obs_space):
        extractor = FZeroFeatureExtractor(obs_space)
        dummy_obs = {
            "screen": torch.randn(1, *obs_space["screen"].shape),
            "float": torch.randn(1, *obs_space["float"].shape),
        }
        output = extractor(dummy_obs)
        assert not torch.any(torch.isnan(output))

    def test_param_count_reasonable(self, obs_space):
        extractor = FZeroFeatureExtractor(obs_space)
        total_params = sum(p.numel() for p in extractor.parameters())
        # Should be in the 1M-5M range based on design
        assert 500_000 < total_params < 10_000_000, f"Param count {total_params} outside expected range"

    def test_gradient_flows(self, obs_space):
        extractor = FZeroFeatureExtractor(obs_space)
        dummy_obs = {
            "screen": torch.randn(2, *obs_space["screen"].shape),
            "float": torch.randn(2, *obs_space["float"].shape),
        }
        output = extractor(dummy_obs)
        loss = output.sum()
        loss.backward()

        # Check that gradients exist for all parameters
        for name, param in extractor.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.any(torch.isnan(param.grad)), f"NaN gradient for {name}"
