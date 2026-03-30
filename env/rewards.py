"""
Reward function for F-Zero RL agent.

Reward: r = delta/ref + 0.5 * (delta/ref)^2

Delta = dot(movement, track_tangent) — the component of the car's movement
in the track-forward direction. This is stable regardless of lateral position
(track width) because it doesn't search for the nearest segment each step.

Why tangent projection is correct for racing:
  - A car taking a wide fast line at 30 u/step gets delta ≈ 28 (slight angle)
  - A car taking a tight slow line at 20 u/step gets delta ≈ 19
  - Wide fast turn gets MORE reward (28 > 19), correctly incentivizing speed
  - The slight underestimate on curves (~5-10%) doesn't change which line is faster
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
        self._prev_px = None
        self._prev_py = None
        self._cur_seg_idx = -1         # -1 = not initialized
        self._seg_progress = 0.0       # progress within current segment [0, seg_len)
        self._cumulative_dist = 0.0    # total distance covered (across laps)
        self._checkpoints = None       # (N, 2) array
        self._seg_lengths = None       # (N,) length of each segment
        self._seg_tangents = None      # (N, 2) unit tangent of each segment
        self._total_length = 0.0

    def reset(self):
        """Reset internal state for a new episode."""
        self._steps_without_progress = 0
        self._prev_px = None
        self._prev_py = None
        self._cur_seg_idx = -1
        self._seg_progress = 0.0
        self._cumulative_dist = 0.0

    def _load_checkpoints(self, info: dict):
        """Load checkpoint coordinates and precompute segment geometry."""
        n_cp = int(info.get("checkpoint_total", 0))
        if n_cp == 0:
            return

        points = []
        for i in range(n_cp):
            cx = float(info.get(f"cp_x_{i}", 0))
            cy = float(info.get(f"cp_y_{i}", 0))
            points.append((cx, cy))

        self._checkpoints = np.array(points, dtype=np.float64)
        n = len(self._checkpoints)

        # Precompute segment lengths and unit tangent vectors
        self._seg_lengths = np.zeros(n)
        self._seg_tangents = np.zeros((n, 2))
        for i in range(n):
            j = (i + 1) % n
            dx = self._checkpoints[j, 0] - self._checkpoints[i, 0]
            dy = self._checkpoints[j, 1] - self._checkpoints[i, 1]
            seg_len = math.sqrt(dx * dx + dy * dy)
            self._seg_lengths[i] = seg_len
            if seg_len > 1e-6:
                self._seg_tangents[i] = (dx / seg_len, dy / seg_len)

        self._total_length = float(np.sum(self._seg_lengths))

    def _init_position(self, px: float, py: float):
        """Global search to find starting segment and position within it."""
        n = len(self._checkpoints)
        best_perp_sq = float("inf")
        best_seg = 0
        best_t = 0.0

        for i in range(n):
            j = (i + 1) % n
            ax, ay = self._checkpoints[i]
            bx, by = self._checkpoints[j]
            dx, dy = bx - ax, by - ay
            seg_len_sq = dx * dx + dy * dy
            if seg_len_sq < 1e-6:
                continue

            t = ((px - ax) * dx + (py - ay) * dy) / seg_len_sq
            t = max(0.0, min(1.0, t))

            proj_x = ax + t * dx
            proj_y = ay + t * dy
            perp_sq = (px - proj_x) ** 2 + (py - proj_y) ** 2

            if perp_sq < best_perp_sq:
                best_perp_sq = perp_sq
                best_seg = i
                best_t = t

        self._cur_seg_idx = best_seg
        self._seg_progress = best_t * self._seg_lengths[best_seg]

    def compute(self, info: dict, action: int = 0) -> tuple[float, dict, bool]:
        """
        Compute reward: r = delta/ref + 0.5 * (delta/ref)^2

        Returns:
            (total_reward, component_dict, should_terminate_early)
        """
        if self._checkpoints is None:
            self._load_checkpoints(info)

        px = float(info.get("player_x_track", info.get("player_x", 0)))
        py = float(info.get("player_y_track", info.get("player_y", 0)))

        if self._cur_seg_idx == -1:
            # First call: initialize position via global search
            self._init_position(px, py)
            self._prev_px = px
            self._prev_py = py
            delta = 0.0
        else:
            # Project movement onto current segment's tangent direction
            move_x = px - self._prev_px
            move_y = py - self._prev_py
            tx, ty = self._seg_tangents[self._cur_seg_idx]
            delta = move_x * tx + move_y * ty

            # Advance segment tracking
            self._seg_progress += delta
            n = len(self._checkpoints)

            # Forward: crossed into next segment(s)
            while self._seg_progress >= self._seg_lengths[self._cur_seg_idx]:
                self._seg_progress -= self._seg_lengths[self._cur_seg_idx]
                self._cur_seg_idx = (self._cur_seg_idx + 1) % n

            # Backward: crossed into previous segment(s)
            while self._seg_progress < 0:
                self._cur_seg_idx = (self._cur_seg_idx - 1) % n
                self._seg_progress += self._seg_lengths[self._cur_seg_idx]

            self._prev_px = px
            self._prev_py = py

        # Accumulate total distance (across laps)
        self._cumulative_dist += max(0.0, delta)

        # Compute reward: linear + quadratic in normalized delta
        ref = self.cfg.speed_ref
        d_norm = delta / ref
        linear = self.cfg.linear_weight * d_norm
        quadratic = self.cfg.quadratic_weight * d_norm * d_norm

        # For negative delta, quadratic would reward going backward.
        # Use signed quadratic to preserve the penalty.
        if delta < 0:
            quadratic = -self.cfg.quadratic_weight * d_norm * d_norm

        components = {
            "linear": linear,
            "quadratic": quadratic,
            "delta": delta,
        }

        # Stuck detection
        if abs(delta) < 0.5:
            self._steps_without_progress += 1
        else:
            self._steps_without_progress = 0

        should_terminate = self._steps_without_progress >= self._stuck_timeout

        if should_terminate:
            components["stuck"] = -self.cfg.stuck_penalty
        else:
            components["stuck"] = 0.0

        total = linear + quadratic + components["stuck"]

        # Expose track state for observation builder
        self.last_track_dist = self._cumulative_dist
        self.last_total_length = self._total_length

        return float(total), components, should_terminate

    def get_nearest_checkpoint_index(self, px: float, py: float) -> int:
        """Get the index of the nearest checkpoint to the player position."""
        if self._checkpoints is None or len(self._checkpoints) < 2:
            return 0
        dists = np.sqrt(np.sum((self._checkpoints - np.array([px, py])) ** 2, axis=1))
        return int(np.argmin(dists))
