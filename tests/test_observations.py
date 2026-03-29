"""Tests for the F-Zero observation processing module."""
import pytest
import numpy as np

from env.observations import FrameProcessor, FloatFeatureBuilder
from training.config import EnvConfig


@pytest.fixture
def frame_proc():
    return FrameProcessor(EnvConfig())


@pytest.fixture
def float_builder():
    return FloatFeatureBuilder(EnvConfig())


class TestFrameProcessor:
    def test_output_shape(self, frame_proc):
        frame_proc.reset()
        # SNES native resolution: 256x224 RGB
        raw = np.random.randint(0, 255, (224, 256, 3), dtype=np.uint8)
        result = frame_proc.process_frame(raw)
        cfg = EnvConfig()
        assert result.shape == (cfg.frame_stack, cfg.screen_height, cfg.screen_width)

    def test_output_range(self, frame_proc):
        frame_proc.reset()
        raw = np.random.randint(0, 255, (224, 256, 3), dtype=np.uint8)
        result = frame_proc.process_frame(raw)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_output_dtype(self, frame_proc):
        frame_proc.reset()
        raw = np.random.randint(0, 255, (224, 256, 3), dtype=np.uint8)
        result = frame_proc.process_frame(raw)
        assert result.dtype == np.float32

    def test_frame_stack_initialized_on_first_call(self, frame_proc):
        frame_proc.reset()
        raw = np.zeros((224, 256, 3), dtype=np.uint8)
        result = frame_proc.process_frame(raw)
        cfg = EnvConfig()
        # All 4 frames should be identical on first call
        for i in range(1, cfg.frame_stack):
            np.testing.assert_array_equal(result[0], result[i])

    def test_frame_stack_shifts_on_subsequent_calls(self, frame_proc):
        frame_proc.reset()
        black = np.zeros((224, 256, 3), dtype=np.uint8)
        white = np.ones((224, 256, 3), dtype=np.uint8) * 255

        frame_proc.process_frame(black)
        result = frame_proc.process_frame(white)

        # Last frame should be white (1.0), earlier frames should be black (0.0)
        assert result[-1].mean() > 0.9
        assert result[0].mean() < 0.1

    def test_reset_clears_stack(self, frame_proc):
        raw = np.random.randint(0, 255, (224, 256, 3), dtype=np.uint8)
        frame_proc.reset()
        frame_proc.process_frame(raw)
        frame_proc.reset()
        # After reset, next call should reinitialize the stack
        result = frame_proc.process_frame(np.zeros((224, 256, 3), dtype=np.uint8))
        # All frames should be identical (zeros)
        for i in range(1, EnvConfig().frame_stack):
            np.testing.assert_array_equal(result[0], result[i])


class TestFloatFeatureBuilder:
    def test_output_dim_matches_declared(self, float_builder):
        float_builder.reset()
        info = {"player_x": 100, "player_y": 100, "energy": 200,
                "lap": 1, "checkpoint_total": 20, "checkpoint_facing": 5}
        checkpoints = np.random.rand(20, 2) * 1000
        result = float_builder.build(info, checkpoints, action=0)
        assert result.shape == (float_builder.dim,)

    def test_output_dtype(self, float_builder):
        float_builder.reset()
        info = {"player_x": 100, "player_y": 100, "energy": 200,
                "lap": 1, "checkpoint_total": 20, "checkpoint_facing": 5}
        result = float_builder.build(info, np.zeros((20, 2)), action=0)
        assert result.dtype == np.float32

    def test_no_nan_values(self, float_builder):
        float_builder.reset()
        info = {"player_x": 0, "player_y": 0, "energy": 0,
                "lap": 0, "checkpoint_total": 0, "checkpoint_facing": 0}
        result = float_builder.build(info, None, action=0)
        assert not np.any(np.isnan(result))

    def test_handles_none_checkpoints(self, float_builder):
        float_builder.reset()
        info = {"player_x": 100, "player_y": 100, "energy": 200,
                "lap": 1, "checkpoint_total": 20, "checkpoint_facing": 5}
        result = float_builder.build(info, None, action=0)
        assert result.shape == (float_builder.dim,)

    def test_action_history_accumulates(self, float_builder):
        float_builder.reset()
        info = {"player_x": 100, "player_y": 100, "energy": 200,
                "lap": 1, "checkpoint_total": 20, "checkpoint_facing": 5}
        cp = np.random.rand(20, 2) * 1000

        r1 = float_builder.build(info, cp, action=0)
        r2 = float_builder.build(info, cp, action=1)
        # Action history portion should differ
        assert not np.array_equal(r1, r2)
