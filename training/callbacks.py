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

    def _on_step(self) -> bool:
        for i, info in enumerate(self.locals.get("infos", [])):
            if i not in self._episode_reward_sums:
                self._episode_reward_sums[i] = {}

            components = info.get("reward_components", {})
            for key, val in components.items():
                full_key = f"reward/{key}"
                self._episode_reward_sums[i][full_key] = (
                    self._episode_reward_sums[i].get(full_key, 0.0) + val
                )

            # Track latest time_penalty_value (snapshot, not sum)
            if "time_penalty_value" in info:
                self._episode_reward_sums[i]["reward/time_penalty_value"] = info["time_penalty_value"]

            # Check if episode ended (SB3 wraps with Monitor, sets "episode" key)
            if "episode" in info:
                for key, val in self._episode_reward_sums[i].items():
                    self.logger.record(key, val)
                # SB3 logger handles aggregation; W&B syncs via sync_tensorboard
                self._episode_reward_sums[i] = {}

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
            # Check if race was completed (lap is 0-indexed: >= 5 means finished)
            lap = info.get("lap", 0)
            if lap >= 5:
                timer_min = info.get("race_timer_min", 0)
                timer_sec = info.get("race_timer_sec", 0)
                timer_csec = info.get("race_timer_csec", 0)
                race_time = timer_min * 60.0 + timer_sec + timer_csec / 100.0

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
