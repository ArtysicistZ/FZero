"""Quick pipeline test: 1000 steps with 8 envs, no W&B, no video."""
import os
import sys

# Load .env
env_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
if os.path.exists(env_file):
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.train import train
from training.config import TrainingConfig

if __name__ == "__main__":
    config = TrainingConfig()
    config.env.n_envs = 1  # Single env for quick test
    config.ppo.total_timesteps = 512
    config.ppo.n_steps = 128
    config.ppo.batch_size = 32
    config.use_wandb = False
    config.video_record_freq = 100000

    train(algo="ppo", config=config)
