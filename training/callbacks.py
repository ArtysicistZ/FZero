"""
Custom SB3 callbacks for F-Zero RL training.

  - RewardLoggingCallback: logs per-component reward breakdown
  - BestLapCallback: saves model when best lap time improves
"""
import os
from stable_baselines3.common.callbacks import BaseCallback


class RewardLoggingCallback(BaseCallback):
    """
    Logs individual reward components to TensorBoard/W&B after each episode.
    Components: progress, speed, wall, time, lap, finish, stuck.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._episode_reward_sums = {}  # {env_idx: {key: sum}}
        self._episode_norm_reward_sums = {}  # {env_idx: float}

    def _on_step(self) -> bool:
        # self.locals["rewards"] contains the NORMALIZED rewards PPO trains on
        norm_rewards = self.locals.get("rewards", [])

        for i, info in enumerate(self.locals.get("infos", [])):
            if i not in self._episode_reward_sums:
                self._episode_reward_sums[i] = {}
                self._episode_norm_reward_sums[i] = 0.0

            # Accumulate raw reward components
            components = info.get("reward_components", {})
            for key, val in components.items():
                full_key = f"reward/{key}"
                self._episode_reward_sums[i][full_key] = (
                    self._episode_reward_sums[i].get(full_key, 0.0) + val
                )

            # Track latest time_penalty_value (snapshot, not sum)
            if "time_penalty_value" in info:
                self._episode_reward_sums[i]["reward/time_penalty_value"] = info["time_penalty_value"]

            # Accumulate normalized reward (what PPO actually trains on)
            if i < len(norm_rewards):
                self._episode_norm_reward_sums[i] += float(norm_rewards[i])

            # Check if episode ended
            if "episode" in info:
                # Log raw components
                for key, val in self._episode_reward_sums[i].items():
                    self.logger.record(key, val)
                # Log normalized episode return (what PPO actually sees)
                self.logger.record("reward/norm_episode_return",
                                   self._episode_norm_reward_sums[i])
                self._episode_reward_sums[i] = {}
                self._episode_norm_reward_sums[i] = 0.0

        return True


class BestLapCallback(BaseCallback):
    """
    Saves the model whenever a new best lap time is achieved.
    Tracks the best 5-lap race time across all environments.
    """

    def __init__(self, save_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self._best_race_time = float("inf")
        self._save_dir = save_dir

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            # race_time is set by fzero_env when race finishes (already BCD-decoded)
            race_time = info.get("race_time", None)
            if race_time is not None:
                if race_time < self._best_race_time:
                    self._best_race_time = race_time
                    save_path = os.path.join(self._save_dir, "best_model")
                    self.model.save(save_path)
                    self.logger.record("best/race_time", race_time)

                    try:
                        import wandb
                        if wandb.run is not None:
                            wandb.log({"best/race_time": race_time}, commit=False)
                    except ImportError:
                        pass

                    if self.verbose > 0:
                        print(f"New best race time: {race_time:.2f}s -> saved to {save_path}")

        return True
