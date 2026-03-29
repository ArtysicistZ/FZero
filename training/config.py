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
    frameskip: int = 3              # 20 actions/sec (matches Linesight's 50ms per action)
    frame_stack: int = 4
    max_episode_steps: int = 24000  # 5 min at 60fps / frameskip=3

    # Frame processing — 96x84 (8:7 ratio = 256:224 native SNES, no crop)
    screen_crop_top: int = 0
    screen_crop_bottom: int = 0
    screen_width: int = 96
    screen_height: int = 84

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
    time_penalty_end: float = 0.30          # break-even at 30 u/step ≈ WR pace with frameskip=3
    time_penalty_ramp_steps: int = 30_000_000  # ramp over ~187K per-env steps at 160 envs (frameskip=3)
    time_penalty_exp_k: float = 4.0         # exponential steepness
    # Stuck penalty: applied when episode is terminated for no progress
    stuck_penalty: float = 1.0
    stuck_timeout_steps: int = 100  # ~5 sec at 20 actions/sec (frameskip=3)
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
    # Linesight uses 1024 (split 512+512 for dueling). For PPO this is a single layer.
    # For IQN, this is split into 512 V-head + 512 A-heads.
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
    gamma: float = 0.999             # long horizon for racing (~1000 step horizon)
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    total_timesteps: int = 50_000_000   # ~10h at 1400 fps with 160 envs


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
    # LR schedule: piecewise linear (Linesight-style)
    lr_schedule: list = field(default_factory=lambda: [
        (0,          1e-3),
        (3_000_000,  5e-5),
        (12_000_000, 5e-5),
        (15_000_000, 1e-5),
    ])
    # Gamma schedule: piecewise linear (Linesight: 0.999 → 1.0 for infinite horizon)
    gamma_schedule: list = field(default_factory=lambda: [
        (0,          0.999),
        (1_500_000,  0.999),
        (2_500_000,  1.0),
    ])
    # Epsilon schedule: piecewise linear (Linesight-style: hold → fast decay → slow decay)
    epsilon_schedule: list = field(default_factory=lambda: [
        (0,        1.0),
        (50_000,   1.0),    # pure exploration initially
        (300_000,  0.1),    # fast decay
        (3_000_000, 0.03),  # slow decay to final
    ])
    buffer_size: int = 200_000
    learning_starts: int = 10_000
    batch_size: int = 512             # Linesight uses 512
    gradient_steps_per_env_step: int = 5   # 5 steps × 512 batch / 80 envs = replay ratio 32 (Linesight)
    n_step_returns: int = 3           # multi-step Bellman (Linesight uses 3)
    target_update_tau: float = 0.02   # soft update (Linesight-style)
    target_update_interval: int = 4   # soft-update every 4 gradient steps
    iqn_embedding_dim: int = 64       # cosine basis dimension
    iqn_n: int = 8                    # quantiles for training
    iqn_k: int = 32                   # quantiles for action selection
    iqn_kappa: float = 0.005          # Huber loss threshold (Linesight: 5e-3, nearly L1)
    use_ddqn: bool = True             # double DQN target selection
    max_grad_norm: float = 30.0       # Linesight uses 30
    max_grad_value: float = 1000.0    # Linesight uses 1000
    use_amp: bool = True              # mixed precision training (fp16 on GPU)
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
