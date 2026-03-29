"""
Centralized configuration for all hyperparameters, paths, and constants.
Every tunable value lives here — no magic numbers scattered across the codebase.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROM_DIR = PROJECT_ROOT / "roms"
RUNS_DIR = PROJECT_ROOT / "runs"  # all training outputs go here
INTEGRATION_DIR = PROJECT_ROOT / "env"  # contains FZero-Snes/ subdirectory

ROM_FILENAME = "F-Zero (USA).sfc"
GAME_NAME = "FZero-Snes"

# RAM addresses are defined in env/FZero-Snes/data.json (the source of truth
# for stable-retro). See SnesLab F-Zero RAM Map for documentation.

# ============================================================
# Environment Configuration
# ============================================================
@dataclass
class EnvConfig:
    n_envs: int = 16
    frameskip: int = 2              # 30 actions/sec — fine control for blast turning
    frame_stack: int = 4
    max_episode_steps: int = 36000  # 5 min at 60fps / frameskip=2

    # Frame processing — 160x120 matches Linesight, preserves SNES 256x224 detail
    screen_crop_top: int = 32
    screen_crop_bottom: int = 32
    screen_width: int = 160
    screen_height: int = 120

    # Track preview (upcoming checkpoints in float obs for cornering anticipation)
    n_preview_checkpoints: int = 10
    # Action history length
    n_action_history: int = 3


# ============================================================
# Reward Configuration
# ============================================================
@dataclass
class RewardConfig:
    # Track-centerline progress reward.
    progress_scale: float = 0.01
    # Exponential time penalty schedule: ramps from start to end over training.
    # Fixed per step (not proportional to speed) → clear gradient for speed optimization.
    time_penalty_start: float = 0.15        # gentle early (break-even at 15 u/step)
    time_penalty_end: float = 0.40          # aggressive late (break-even at 40 u/step)
    time_penalty_ramp_steps: int = 40_000_000  # ramp over ~250K per-env steps at 160 envs
    time_penalty_exp_k: float = 4.0         # exponential steepness
    # Stuck penalty: applied when episode is terminated for no progress
    stuck_penalty: float = 1.0
    stuck_timeout_steps: int = 150  # ~5 sec at 30fps (frameskip=2)
    # Removed: wall_penalty (walls self-punish via reduced progress)
    # Removed: speed_bonus (redundant with progress — faster = more progress)
    # Removed: lap_bonus / finish_bonus (progress already rewards completing laps)
    reward_clip_min: float = -2.0
    reward_clip_max: float = 2.0


# ============================================================
# Network Configuration
# ============================================================
@dataclass
class NetworkConfig:
    # CNN branch (image head)
    cnn_channels: list = field(default_factory=lambda: [16, 32, 64, 32])
    cnn_kernels: list = field(default_factory=lambda: [4, 4, 3, 3])
    cnn_strides: list = field(default_factory=lambda: [2, 2, 2, 1])

    # Float branch
    float_hidden_dim: int = 128

    # Fusion (output dim of the feature extractor, fed to SB3 policy/value heads)
    # Linesight uses 1024; matches our larger fusion input (5760 dims)
    dense_hidden_dim: int = 1024

    # Weight init
    leaky_relu_slope: float = 0.01


# ============================================================
# PPO Configuration
# ============================================================
@dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    n_steps: int = 512
    # batch_size should scale with n_envs: ~n_envs × n_steps / 20
    # With 80 envs: 2048. With 160 envs: 4096.
    batch_size: int = 4096
    n_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    total_timesteps: int = 100_000_000  # 2x more steps with frameskip=2


# ============================================================
# DQN Configuration
# ============================================================
@dataclass
class DQNConfig:
    learning_rate: float = 1e-4
    buffer_size: int = 100_000
    learning_starts: int = 10_000
    batch_size: int = 32
    gamma: float = 0.99
    target_update_interval: int = 1000
    exploration_fraction: float = 0.1
    exploration_final_eps: float = 0.05
    total_timesteps: int = 100_000_000


# ============================================================
# QR-DQN Configuration (distributional RL — closest to IQN in SB3)
# ============================================================
@dataclass
class QRDQNConfig:
    learning_rate: float = 5e-5
    buffer_size: int = 200_000
    learning_starts: int = 10_000
    batch_size: int = 64
    gamma: float = 0.99
    target_update_interval: int = 5000
    exploration_fraction: float = 0.1
    exploration_final_eps: float = 0.03
    n_quantiles: int = 64            # number of quantiles for return distribution
    total_timesteps: int = 100_000_000


# ============================================================
# IQN Configuration (custom implementation)
# ============================================================
@dataclass
class IQNConfig:
    learning_rate: float = 5e-5
    buffer_size: int = 200_000
    learning_starts: int = 10_000
    batch_size: int = 64
    gamma: float = 0.99
    target_update_interval: int = 5000
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.03
    exploration_fraction: float = 0.1
    iqn_embedding_dim: int = 64       # cosine basis dimension
    iqn_n: int = 8                    # quantiles for training
    iqn_k: int = 32                   # quantiles for action selection
    iqn_kappa: float = 1.0            # Huber loss threshold
    use_ddqn: bool = True             # double DQN target selection
    max_grad_norm: float = 10.0
    total_timesteps: int = 100_000_000


# ============================================================
# Training Configuration
# ============================================================
@dataclass
class TrainingConfig:
    use_wandb: bool = True
    wandb_project: str = "fzero-rl"
    video_record_freq: int = 200_000  # steps between video recordings (~50 videos per 10M steps)
    eval_freq: int = 10_000
    save_freq: int = 50_000
    log_interval: int = 1

    env: EnvConfig = field(default_factory=EnvConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    dqn: DQNConfig = field(default_factory=DQNConfig)
    qrdqn: QRDQNConfig = field(default_factory=QRDQNConfig)
    iqn: IQNConfig = field(default_factory=IQNConfig)


# ============================================================
# Game Constants (derived from F-Zero game data)
# ============================================================
MAX_POSITION_X = 8192
MAX_POSITION_Y = 4096
MAX_ENERGY = 2048  # Max power meter value (verified: starts at ~2048)
MAX_SPEED = 50.0  # Max position delta per frame (to be calibrated during M1)
N_ACTIONS_FLAT = 72   # 3*3*2*2*2 (for flat Discrete algorithms like DQN/QR-DQN)
TOTAL_LAPS = 5        # F-Zero has 5 laps
LAP_ZERO_INDEXED = True  # RAM lap counter is 0-indexed (0=lap1, 4=lap5)
BOOST_AVAILABLE_LAP = 1  # Boost available from lap index 1 (second lap)
