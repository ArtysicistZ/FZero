"""
Reward function for F-Zero RL agent.

Reward: r = delta/ref + 0.5 * (delta/ref)^2

Delta = track progress per step, computed by projecting the car onto a
smooth centerline (spline-interpolated from 58 RAM checkpoints → 660 pts).
Global search over all segments.
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
        self._prev_px = None
        self._prev_py = None
        self._cumulative_dist = 0.0
        self._checkpoints = None       # (N, 2) dense centerline
        self._cum_dist = None          # (N,) cumulative distance
        self._seg_lengths = None       # (N,) segment lengths
        self._total_length = 0.0

    def reset(self):
        """Reset internal state for a new episode."""
        self._steps_without_progress = 0
        self._prev_track_dist = None
        self._prev_px = None
        self._prev_py = None
        self._cumulative_dist = 0.0

    def _load_checkpoints(self, info: dict):
        """Build dense centerline by spline-interpolating 58 RAM checkpoints."""
        from scipy.interpolate import splprep, splev

        n_cp = int(info.get("checkpoint_total", 0))
        if n_cp == 0:
            return

        points = []
        for i in range(n_cp):
            cx = float(info.get(f"cp_x_{i}", 0))
            cy = float(info.get(f"cp_y_{i}", 0))
            points.append((cx, cy))
        cps = np.array(points, dtype=np.float64)

        # Spline interpolate → 660 dense points (smooth, no L-turns)
        tck, u = splprep([cps[:, 0], cps[:, 1]], s=50000, per=True, k=3)
        u_dense = np.linspace(0, 1, 660, endpoint=False)
        self._checkpoints = np.array(splev(u_dense, tck)).T

        # Remove degenerate closing point
        if len(self._checkpoints) >= 3:
            if np.allclose(self._checkpoints[-1], self._checkpoints[0], atol=0.01):
                self._checkpoints = self._checkpoints[:-1]

        n = len(self._checkpoints)

        # Precompute segment lengths
        self._seg_lengths = np.zeros(n)
        for i in range(n):
            j = (i + 1) % n
            dx = self._checkpoints[j, 0] - self._checkpoints[i, 0]
            dy = self._checkpoints[j, 1] - self._checkpoints[i, 1]
            self._seg_lengths[i] = math.sqrt(dx * dx + dy * dy)

        self._cum_dist = np.concatenate([[0.0], np.cumsum(self._seg_lengths[:-1])])
        self._total_length = float(np.sum(self._seg_lengths))

    def _project_onto_segment(self, px, py, seg_idx):
        """Project point (px, py) onto segment seg_idx → (seg_idx+1) % N.

        Returns (track_dist, perp_dist_sq) or (None, inf) if degenerate.
        """
        n = len(self._checkpoints)
        i = seg_idx % n
        j = (i + 1) % n
        ax, ay = self._checkpoints[i]
        bx, by = self._checkpoints[j]
        dx, dy = bx - ax, by - ay
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq < 1e-6:
            return None, float("inf")

        t = ((px - ax) * dx + (py - ay) * dy) / seg_len_sq
        t = max(0.0, min(1.0, t))

        proj_x = ax + t * dx
        proj_y = ay + t * dy
        perp_dist_sq = (px - proj_x) ** 2 + (py - proj_y) ** 2

        track_dist = self._cum_dist[i] + t * self._seg_lengths[i]
        return track_dist, perp_dist_sq

    def _get_track_distance(self, px: float, py: float) -> float:
        """Project car onto centerline via global search."""
        if self._checkpoints is None or len(self._checkpoints) < 2:
            return 0.0

        n = len(self._checkpoints)
        best_track_dist = 0.0
        best_perp_sq = float("inf")

        for seg_idx in range(n):
            td, perp_sq = self._project_onto_segment(px, py, seg_idx)
            if td is not None and perp_sq <= best_perp_sq:
                best_perp_sq = perp_sq
                best_track_dist = td

        return best_track_dist

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

        # Get track distance
        track_dist = self._get_track_distance(px, py)

        if self._prev_track_dist is None:
            delta = 0.0
        else:
            raw_delta = track_dist - self._prev_track_dist
            L = self._total_length

            if (self._prev_track_dist > L * 0.75) and (track_dist < L * 0.25):
                # Forward lap crossing
                delta = (L - self._prev_track_dist) + track_dist
            elif (self._prev_track_dist < L * 0.25) and (track_dist > L * 0.75):
                # Backward lap crossing
                delta = -(self._prev_track_dist + (L - track_dist))
            else:
                delta = raw_delta

        # Euclidean movement for stuck detection
        if self._prev_px is not None:
            move_x = px - self._prev_px
            move_y = py - self._prev_py
            euclid_move = math.sqrt(move_x * move_x + move_y * move_y)
        else:
            euclid_move = 1.0

        self._prev_track_dist = track_dist
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
        if delta < 0:
            quadratic = -self.cfg.quadratic_weight * d_norm * d_norm

        components = {
            "linear": linear,
            "quadratic": quadratic,
            "delta": delta,
        }

        # Stuck detection
        if euclid_move < 0.5:
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
