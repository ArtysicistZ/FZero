"""Tests for the F-Zero reward computation module."""
import pytest
import numpy as np

from env.rewards import RewardCalculator
from training.config import RewardConfig


@pytest.fixture
def calc():
    return RewardCalculator(RewardConfig())


def _make_info(player_x_track=2200, player_y_track=368, **overrides):
    """Create info dict with checkpoint data for track projection."""
    defaults = {
        "player_x_track": player_x_track,
        "player_y_track": player_y_track,
        "player_x": 1663,
        "player_y": 3919,
        "energy": 200,
        "lap": 0,
        "checkpoint_total": 4,
        "checkpoint_facing": 0,
        # Simple square track: 4 checkpoints forming a loop
        "cp_x_0": 1000, "cp_y_0": 0,
        "cp_x_1": 2000, "cp_y_1": 0,
        "cp_x_2": 2000, "cp_y_2": 1000,
        "cp_x_3": 1000, "cp_y_3": 1000,
    }
    defaults.update(overrides)
    return defaults


class TestRewardCalculator:
    def test_first_step_returns_valid_reward(self, calc):
        calc.reset()
        reward, components, stuck = calc.compute(
            _make_info(player_x_track=1500, player_y_track=0)
        )
        assert isinstance(reward, float)
        assert not np.isnan(reward)
        assert "progress" in components
        assert "time" in components

    def test_first_step_progress_is_zero(self, calc):
        calc.reset()
        _, components, _ = calc.compute(
            _make_info(player_x_track=1500, player_y_track=0)
        )
        assert components["progress"] == 0.0

    def test_progress_positive_on_forward_movement(self, calc):
        calc.reset()
        # Start at x=1200 (on segment CP0→CP1, along x-axis)
        calc.compute(_make_info(player_x_track=1200, player_y_track=0))
        # Move forward to x=1500
        _, components, _ = calc.compute(
            _make_info(player_x_track=1500, player_y_track=0)
        )
        assert components["progress"] > 0

    def test_progress_negative_on_backward_movement(self, calc):
        calc.reset()
        calc.compute(_make_info(player_x_track=1500, player_y_track=0))
        # Move backward to x=1200
        _, components, _ = calc.compute(
            _make_info(player_x_track=1200, player_y_track=0)
        )
        assert components["progress"] < 0

    def test_time_penalty_is_negative(self, calc):
        calc.reset()
        _, components, _ = calc.compute(
            _make_info(player_x_track=1500, player_y_track=0)
        )
        assert components["time"] < 0

    def test_reward_clipped_within_bounds(self, calc):
        cfg = RewardConfig()
        calc.reset()
        reward, _, _ = calc.compute(
            _make_info(player_x_track=1500, player_y_track=0)
        )
        assert cfg.reward_clip_min <= reward <= cfg.reward_clip_max

    def test_stuck_detection_after_timeout(self, calc):
        calc.reset()
        # Stay at same position
        info = _make_info(player_x_track=1500, player_y_track=0)
        stuck = False
        for _ in range(calc._stuck_timeout + 1):
            _, _, stuck = calc.compute(info)
        assert stuck is True

    def test_not_stuck_when_making_progress(self, calc):
        calc.reset()
        # Move forward each step along CP0→CP1 segment (x-axis)
        for i in range(100):
            x = 1000 + (i % 900)  # cycle within segment
            _, _, stuck = calc.compute(
                _make_info(player_x_track=x, player_y_track=0)
            )
        assert stuck is False

    def test_no_wall_penalty_component(self, calc):
        calc.reset()
        calc.compute(_make_info(player_x_track=1500, player_y_track=0))
        _, components, _ = calc.compute(
            _make_info(player_x_track=1500, player_y_track=0)
        )
        assert "wall" not in components

    def test_time_penalty_increases_over_steps(self, calc):
        """Time penalty should increase over training steps (exponential schedule)."""
        calc.reset()
        penalties = []
        for i in range(100):
            x = 1000 + (i % 900)
            _, components, _ = calc.compute(
                _make_info(player_x_track=x, player_y_track=0)
            )
            penalties.append(-components["time"])
        # Penalty should increase from start toward end
        assert penalties[-1] > penalties[0]

    def test_time_penalty_starts_at_configured_value(self, calc):
        """Time penalty should start near time_penalty_start."""
        calc.reset()
        _, components, _ = calc.compute(
            _make_info(player_x_track=1500, player_y_track=0)
        )
        cfg = RewardConfig()
        assert abs(-components["time"] - cfg.time_penalty_start) < 0.01
