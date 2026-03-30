"""
Reward function for F-Zero RL agent.

Three reward components:
  1. Progress: distance advanced along track centerline (per step)
  2. Time penalty: fixed cost per step (drives speed optimization)
  3. PBRS: potential-based reward shaping for cornering (Andrew Ng 1999)
  4. Time bonus: quadratic bonus at race completion (strong speed signal)

Uses checkpoint coordinates from RAM to define the track path.
"""
import math

import numpy as np

from training.config import RewardConfig


class RewardCalculator:
    """Computes shaped rewards from F-Zero game state."""

    def __init__(self, cfg: RewardConfig):
        self.cfg = cfg
        self._stuck_timeout = cfg.stuck_timeout_steps
        self._steps_without_progress = 0
        self._prev_track_dist = None
        self._cumulative_dist = 0.0  # total distance covered (across laps)
        self._checkpoints = None   # (N, 2) array, loaded on first compute
        self._cum_dist = None      # (N,) cumulative distance along track
        self._total_length = 0.0

    def reset(self):
        """Reset internal state for a new episode."""
        self._steps_without_progress = 0
        self._prev_track_dist = None
        self._cumulative_dist = 0.0

    def _load_checkpoints(self, info: dict):
        """Load checkpoint coordinates from RAM info on first call."""
        n_cp = int(info.get("checkpoint_total", 0))
        if n_cp == 0:
            return

        points = []
        for i in range(n_cp):
            cx = float(info.get(f"cp_x_{i}", 0))
            cy = float(info.get(f"cp_y_{i}", 0))
            points.append((cx, cy))

        self._checkpoints = np.array(points, dtype=np.float64)

        # Compute cumulative distance along checkpoint path (closed loop)
        diffs = np.diff(self._checkpoints, axis=0)
        seg_lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
        close_diff = self._checkpoints[0] - self._checkpoints[-1]
        close_len = np.sqrt(np.sum(close_diff ** 2))
        self._cum_dist = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        self._total_length = self._cum_dist[-1] + close_len

    def _get_track_distance(self, px: float, py: float) -> float:
        """Project player onto track centerline. Returns distance along track."""
        if self._checkpoints is None or len(self._checkpoints) < 2:
            return 0.0

        n = len(self._checkpoints)
        best_track_dist = 0.0
        best_perp_dist_sq = float("inf")

        for i in range(n):
            next_i = (i + 1) % n
            ax, ay = self._checkpoints[i]
            bx, by = self._checkpoints[next_i]
            dx, dy = bx - ax, by - ay
            seg_len_sq = dx * dx + dy * dy
            if seg_len_sq < 1e-6:
                continue

            t = ((px - ax) * dx + (py - ay) * dy) / seg_len_sq
            t = max(0.0, min(1.0, t))

            proj_x = ax + t * dx
            proj_y = ay + t * dy
            perp_dist_sq = (px - proj_x) ** 2 + (py - proj_y) ** 2

            if perp_dist_sq < best_perp_dist_sq:
                best_perp_dist_sq = perp_dist_sq
                seg_len = math.sqrt(seg_len_sq)
                cum_at_i = self._cum_dist[i] if i < len(self._cum_dist) else self._cum_dist[-1]
                best_track_dist = cum_at_i + t * seg_len

        return best_track_dist

    def compute(self, info: dict, action: int = 0) -> tuple[float, dict, bool]:
        """
        Compute reward.

        Returns:
            (total_reward, component_dict, should_terminate_early)
        """
        if self._checkpoints is None:
            self._load_checkpoints(info)

        px = float(info.get("player_x_track", info.get("player_x", 0)))
        py = float(info.get("player_y_track", info.get("player_y", 0)))

        components = {}

        # 1. TRACK PROGRESS
        track_dist = self._get_track_distance(px, py)

        if self._prev_track_dist is None:
            delta = 0.0
        else:
            delta = track_dist - self._prev_track_dist
            if delta < -self._total_length / 2:
                delta += self._total_length
            elif delta > self._total_length / 2:
                delta -= self._total_length

        components["progress"] = delta * self.cfg.progress_scale

        # Accumulate total distance (across laps)
        self._cumulative_dist += max(0.0, delta)

        # 2. FIXED TIME PENALTY
        components["time"] = -self.cfg.time_penalty

        # 3. STUCK DETECTION
        if abs(delta) < 0.5:
            self._steps_without_progress += 1
        else:
            self._steps_without_progress = 0

        should_terminate = self._steps_without_progress >= self._stuck_timeout

        if should_terminate:
            components["stuck"] = -self.cfg.stuck_penalty
        else:
            components["stuck"] = 0.0

        # Update state
        self._prev_track_dist = track_dist

        total = sum(components.values())
        total = np.clip(total, self.cfg.reward_clip_min, self.cfg.reward_clip_max)

        # Expose track state for observation builder
        self.last_track_dist = self._cumulative_dist  # cumulative across laps
        self.last_total_length = self._total_length    # single lap length

        return float(total), components, should_terminate

    def get_nearest_checkpoint_index(self, px: float, py: float) -> int:
        """Get the index of the nearest checkpoint to the player position."""
        if self._checkpoints is None or len(self._checkpoints) < 2:
            return 0
        dists = np.sqrt(np.sum((self._checkpoints - np.array([px, py])) ** 2, axis=1))
        return int(np.argmin(dists))

    def compute_time_bonus(self, race_time: float) -> float:
        """Compute quadratic time bonus at race completion.

        Called by the environment when lap >= 5.
        Returns bonus reward: ((ref - time) / ref)^2 * scale
        """
        ref = self.cfg.time_bonus_reference
        if race_time >= ref:
            return 0.0
        return ((ref - race_time) / ref) ** 2 * self.cfg.time_bonus_scale
