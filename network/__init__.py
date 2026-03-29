"""
Network module — custom neural network architectures for F-Zero RL.

Public API:
  FZeroFeatureExtractor — SB3-compatible dual-input feature extractor
  IQNNetwork — Custom IQN with branching dueling heads
"""
from network.dual_input import FZeroFeatureExtractor
from network.iqn import IQNNetwork

__all__ = ["FZeroFeatureExtractor", "IQNNetwork"]
