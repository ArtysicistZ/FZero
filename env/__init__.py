"""
F-Zero RL environment module.

Public API:
  make_fzero_env(n_envs, render_mode, env_config, reward_config, flat_actions) -> VecEnv
"""
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from training.config import (
    EnvConfig, RewardConfig, INTEGRATION_DIR, GAME_NAME,
)


def _make_single_env(render_mode: str, env_config: EnvConfig, reward_config: RewardConfig,
                     flat_actions: bool = False):
    """Factory function that creates a single FZeroEnv. Used by VecEnv."""
    integration_dir_abs = str(INTEGRATION_DIR.resolve())
    game_name = GAME_NAME

    def _init():
        import stable_retro
        stable_retro.data.add_custom_integration(integration_dir_abs)
        from env.fzero_env import FZeroEnv, FlatDiscreteWrapper
        env = FZeroEnv(
            game_path=game_name,
            render_mode=render_mode,
            env_config=env_config,
            reward_config=reward_config,
        )
        if flat_actions:
            env = FlatDiscreteWrapper(env)
        return Monitor(env)
    return _init


def make_fzero_env(
    n_envs: int = 1,
    render_mode: str = "rgb_array",
    env_config: EnvConfig = None,
    reward_config: RewardConfig = None,
    flat_actions: bool = False,
):
    """
    Create a vectorized F-Zero environment.

    Args:
        n_envs: Number of parallel environments
        render_mode: "rgb_array" for training, "human" for visualization
        env_config: Environment configuration (uses defaults if None)
        reward_config: Reward configuration (uses defaults if None)
        flat_actions: If True, use Discrete(72) instead of MultiDiscrete([3,3,2,2,2]).
                      Required for DQN and QR-DQN.
    """
    cfg_env = env_config or EnvConfig()
    cfg_reward = reward_config or RewardConfig()

    env_fns = [_make_single_env(render_mode, cfg_env, cfg_reward, flat_actions)
               for _ in range(n_envs)]

    if n_envs == 1:
        return DummyVecEnv(env_fns)
    return SubprocVecEnv(env_fns)
