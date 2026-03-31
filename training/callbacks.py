"""
Custom SB3 callbacks for F-Zero RL training.

  - RewardLoggingCallback: logs per-component reward breakdown + speed metrics
  - BestLapCallback: saves model when best lap time improves
"""
import os
from stable_baselines3.common.callbacks import BaseCallback


class RewardLoggingCallback(BaseCallback):
    """
    Logs speed metrics and episode events to TensorBoard/W&B.

    Metrics:
      - speed/avg_delta: average track progress per step
      - speed/max_delta: peak speed in episode
      - episode/length: episode length in steps
      - episode/stuck: 1 if episode ended by stuck, 0 otherwise
      - race/time: every race completion time
      - race/best_time: best race time so far
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._episode_deltas = {}        # {env_idx: [delta, ...]}
        self._episode_steps = {}         # {env_idx: int}
        self._best_race_time = float("inf")

    def _on_step(self) -> bool:
        for i, info in enumerate(self.locals.get("infos", [])):
            if i not in self._episode_deltas:
                self._episode_deltas[i] = []
                self._episode_steps[i] = 0

            self._episode_steps[i] += 1

            # Track deltas
            components = info.get("reward_components", {})
            if "delta" in components:
                self._episode_deltas[i].append(components["delta"])

            # Check if episode ended
            if "episode" in info:
                # Log whether this episode ended by stuck (1) or not (0)
                ended_stuck = 1 if components.get("stuck", 0) != 0 else 0
                self.logger.record("episode/stuck", ended_stuck)

                # Log speed metrics from deltas
                deltas = self._episode_deltas[i]
                if deltas:
                    import numpy as np
                    deltas_arr = np.array(deltas)
                    self.logger.record("speed/avg_delta", np.mean(deltas_arr))
                    self.logger.record("speed/max_delta", np.max(deltas_arr))
                    self.logger.record("episode/length", self._episode_steps[i])

                # Log every race completion time + best
                race_time = info.get("race_time", None)
                if race_time is not None:
                    self.logger.record("race/time", race_time)
                    if race_time < self._best_race_time:
                        self._best_race_time = race_time
                    self.logger.record("race/best_time", self._best_race_time)
                    try:
                        import wandb
                        if wandb.run is not None:
                            wandb.log({"race/time": race_time,
                                       "race/best_time": self._best_race_time}, commit=False)
                    except ImportError:
                        pass

                # Reset
                self._episode_deltas[i] = []
                self._episode_steps[i] = 0

        return True


class AdaptiveLRCallback(BaseCallback):
    """
    Adjusts learning rate to keep approx_kl near target_kl.

    After each rollout update:
      - If KL > target: LR *= 0.5 (too aggressive, slow down)
      - If KL < target / 2: LR *= 1.5 (room to learn faster)
      - Clamp LR to [1e-6, 1e-3]
    """

    def __init__(self, target_kl: float = 0.015, verbose: int = 0):
        super().__init__(verbose)
        self._target_kl = target_kl

    def _on_rollout_end(self) -> None:
        # SB3 logs approx_kl after each train() call
        # Access it from the logger's name_to_value
        kl = self.logger.name_to_value.get("train/approx_kl", None)
        if kl is None:
            return

        optimizer = self.model.policy.optimizer
        old_lr = optimizer.param_groups[0]["lr"]

        if kl > self._target_kl:
            new_lr = max(old_lr * 0.5, 1e-6)
        elif kl < self._target_kl / 2:
            new_lr = min(old_lr * 1.5, 1e-3)
        else:
            new_lr = old_lr

        if new_lr != old_lr:
            for pg in optimizer.param_groups:
                pg["lr"] = new_lr
            if self.verbose > 0:
                print(f"AdaptiveLR: KL={kl:.4f}, LR {old_lr:.2e} -> {new_lr:.2e}")

        self.logger.record("train/adaptive_lr", new_lr)

    def _on_step(self) -> bool:
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
            race_time = info.get("race_time", None)
            if race_time is not None:
                if race_time < self._best_race_time:
                    self._best_race_time = race_time
                    save_path = os.path.join(self._save_dir, "best_model")
                    self.model.save(save_path)
                    if self.verbose > 0:
                        print(f"New best race time: {race_time:.2f}s -> saved to {save_path}")

        return True
