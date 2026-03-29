"""
Reward function for F-Zero RL agent.

Linesight-style: reward = distance advanced along track centerline + time penalty.
Uses checkpoint coordinates from RAM ($7E:1200-15FF) to define the track path,
then projects the player's position onto this path to compute directional progress.
"""
import math

import numpy as np

from training.config import RewardConfig


class RewardCalculator:
    """Computes shaped rewards from F-Zero game state.

    Track progress is computed by projecting the player's position onto the
    checkpoint-defined track centerline. Only forward movement along the track
    counts — driving sideways, backward, or in circles gives zero progress.
    """

    def __init__(self, cfg: RewardConfig, n_envs: int = 1):
        self.cfg = cfg
        self._stuck_timeout = cfg.stuck_timeout_steps
        self._steps_without_progress = 0
        self._prev_track_dist = None
        self._total_steps = 0      # global step counter for penalty schedule
        self._per_env_ramp = cfg.time_penalty_ramp_steps / max(n_envs, 1)
        self._checkpoints = None   # (N, 2) array, loaded on first compute
        self._cum_dist = None      # (N,) cumulative distance along track
        self._total_length = 0.0

    def reset(self):
        """Reset internal state for a new episode."""
        self._steps_without_progress = 0
        self._prev_track_dist = None
        # Keep _total_steps across episodes — it drives the penalty schedule
        # Keep checkpoints — they don't change between episodes

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
        # Add closing segment (last -> first checkpoint)
        close_diff = self._checkpoints[0] - self._checkpoints[-1]
        close_len = np.sqrt(np.sum(close_diff ** 2))
        all_seg_lengths = np.concatenate([seg_lengths, [close_len]])
        self._cum_dist = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        self._total_length = self._cum_dist[-1] + close_len

    def _get_track_distance(self, px: float, py: float) -> float:
        """Compute player's distance along the track centerline.

        Projects the player position onto EVERY track segment and picks the
        closest one. This gives smooth, per-frame progress even when checkpoints
        are far apart.

        Returns a value in [0, total_track_length).
        """
        if self._checkpoints is None or len(self._checkpoints) < 2:
            return 0.0

        n = len(self._checkpoints)
        best_track_dist = 0.0
        best_perp_dist_sq = float("inf")

        # Include the closing segment (last checkpoint -> first checkpoint)
        for i in range(n):
            next_i = (i + 1) % n
            ax, ay = self._checkpoints[i]
            bx, by = self._checkpoints[next_i]
            dx, dy = bx - ax, by - ay
            seg_len_sq = dx * dx + dy * dy
            if seg_len_sq < 1e-6:
                continue

            # Project player onto this segment
            t = ((px - ax) * dx + (py - ay) * dy) / seg_len_sq
            t = max(0.0, min(1.0, t))

            # Closest point on segment
            proj_x = ax + t * dx
            proj_y = ay + t * dy
            perp_dist_sq = (px - proj_x) ** 2 + (py - proj_y) ** 2

            if perp_dist_sq < best_perp_dist_sq:
                best_perp_dist_sq = perp_dist_sq
                seg_len = math.sqrt(seg_len_sq)
                # For the closing segment (i = n-1), cum_dist[i] is the last checkpoint
                cum_at_i = self._cum_dist[i] if i < len(self._cum_dist) else self._cum_dist[-1]
                best_track_dist = cum_at_i + t * seg_len

        return best_track_dist

    def compute(self, info: dict, action: int = 0) -> tuple[float, dict, bool]:
        """
        Compute reward based on distance advanced along track centerline.

        Returns:
            (total_reward, component_dict, should_terminate_early)
        """
        # Load checkpoints on first call
        if self._checkpoints is None:
            self._load_checkpoints(info)

        # Use the correct player position addresses (0x0B70, 0x0B90)
        px = float(info.get("player_x_track", info.get("player_x", 0)))
        py = float(info.get("player_y_track", info.get("player_y", 0)))

        components = {}

        # 1. TRACK PROGRESS — distance advanced along centerline
        track_dist = self._get_track_distance(px, py)

        if self._prev_track_dist is None:
            delta = 0.0
        else:
            delta = track_dist - self._prev_track_dist
            # Handle lap wraparound (crossing start/finish line)
            if delta < -self._total_length / 2:
                delta += self._total_length
            elif delta > self._total_length / 2:
                delta -= self._total_length

        components["progress"] = delta * self.cfg.progress_scale

        # 2. EXPONENTIAL TIME PENALTY SCHEDULE
        # Fixed per step (not proportional to speed) → clear gradient for speed.
        # Ramps from start to end over training via normalized exponential.
        self._total_steps += 1
        frac = min(1.0, self._total_steps / self._per_env_ramp)
        k = self.cfg.time_penalty_exp_k
        t = (1 - math.exp(-k * frac)) / (1 - math.exp(-k))
        time_penalty = self.cfg.time_penalty_start + (self.cfg.time_penalty_end - self.cfg.time_penalty_start) * t
        components["time"] = -time_penalty

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

        return float(total), components, should_terminate
