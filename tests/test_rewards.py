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
        calc.compute(_make_info(player_x_track=1200, player_y_track=0))
        _, components, _ = calc.compute(
            _make_info(player_x_track=1500, player_y_track=0)
        )
        assert components["progress"] > 0

    def test_progress_negative_on_backward_movement(self, calc):
        calc.reset()
        calc.compute(_make_info(player_x_track=1500, player_y_track=0))
        _, components, _ = calc.compute(
            _make_info(player_x_track=1200, player_y_track=0)
        )
        assert components["progress"] < 0

    def test_time_penalty_is_fixed(self, calc):
        calc.reset()
        _, components, _ = calc.compute(
            _make_info(player_x_track=1500, player_y_track=0)
        )
        assert components["time"] == -RewardConfig().time_penalty

    def test_reward_clipped_within_bounds(self, calc):
        cfg = RewardConfig()
        calc.reset()
        reward, _, _ = calc.compute(
            _make_info(player_x_track=1500, player_y_track=0)
        )
        assert cfg.reward_clip_min <= reward <= cfg.reward_clip_max

    def test_stuck_detection_after_timeout(self, calc):
        calc.reset()
        info = _make_info(player_x_track=1500, player_y_track=0)
        stuck = False
        for _ in range(calc._stuck_timeout + 1):
            _, _, stuck = calc.compute(info)
        assert stuck is True

    def test_not_stuck_when_making_progress(self, calc):
        calc.reset()
        for i in range(100):
            x = 1000 + (i % 900)
            _, _, stuck = calc.compute(
                _make_info(player_x_track=x, player_y_track=0)
            )
        assert stuck is False

    def test_no_pbrs_in_components(self, calc):
        calc.reset()
        calc.compute(_make_info(player_x_track=1500, player_y_track=0))
        _, components, _ = calc.compute(
            _make_info(player_x_track=1600, player_y_track=0)
        )
        assert "pbrs" not in components

    def test_time_bonus_positive_for_fast_race(self, calc):
        bonus = calc.compute_time_bonus(120.0)
        assert bonus > 0

    def test_time_bonus_zero_for_slow_race(self, calc):
        cfg = RewardConfig()
        bonus = calc.compute_time_bonus(cfg.time_bonus_reference + 10)
        assert bonus == 0.0

    def test_time_bonus_increases_with_speed(self, calc):
        bonus_slow = calc.compute_time_bonus(160.0)
        bonus_fast = calc.compute_time_bonus(120.0)
        assert bonus_fast > bonus_slow

    def test_time_bonus_quadratic(self, calc):
        """Marginal reward should increase as race time decreases."""
        bonus_162 = calc.compute_time_bonus(162.0)
        bonus_161 = calc.compute_time_bonus(161.0)
        bonus_120 = calc.compute_time_bonus(120.0)
        bonus_119 = calc.compute_time_bonus(119.0)
        marginal_easy = bonus_161 - bonus_162
        marginal_hard = bonus_119 - bonus_120
        assert marginal_hard > marginal_easy
