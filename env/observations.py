"""
Observation construction for F-Zero environment.

Builds a dual-input observation:
  1. Screen: cropped, grayscale, resized game frame (4 x 84 x 84)
  2. Float: rich game state vector (~50 dims) including track preview and action history
"""
import numpy as np
import cv2

from training.config import (
    EnvConfig, MAX_POSITION_X, MAX_POSITION_Y, MAX_ENERGY, MAX_SPEED,
)


class FrameProcessor:
    """Processes raw SNES frames into CNN-ready observations."""

    def __init__(self, cfg: EnvConfig):
        self.crop_top = cfg.screen_crop_top
        self.crop_bottom = cfg.screen_crop_bottom
        self.width = cfg.screen_width
        self.height = cfg.screen_height
        self.stack_size = cfg.frame_stack
        self._frame_stack = None

    def reset(self):
        """Clear the frame stack (call on episode reset)."""
        self._frame_stack = None

    def process_frame(self, raw_frame: np.ndarray) -> np.ndarray:
        """
        Process a single raw SNES frame and return the stacked observation.

        Args:
            raw_frame: (H, W, 3) uint8 RGB frame from the emulator

        Returns:
            (frame_stack, H, W) float32 array in [0, 1]
        """
        h = raw_frame.shape[0]
        bottom = h - self.crop_bottom
        cropped = raw_frame[self.crop_top:bottom, :, :]

        gray = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (self.width, self.height), interpolation=cv2.INTER_AREA)

        # Normalize to [0, 1] float32
        frame = resized.astype(np.float32) / 255.0

        if self._frame_stack is None:
            # Initialize stack with copies of the first frame
            self._frame_stack = np.stack([frame] * self.stack_size, axis=0)
        else:
            # Shift stack and add new frame
            self._frame_stack = np.roll(self._frame_stack, shift=-1, axis=0)
            self._frame_stack[-1] = frame

        return self._frame_stack.copy()


class FloatFeatureBuilder:
    """Builds the float feature vector from RAM state and action history."""

    def __init__(self, cfg: EnvConfig):
        self.n_preview = cfg.n_preview_checkpoints
        self.n_history = cfg.n_action_history
        # Core state (5) + track preview (n_preview * 3) + action history
        # Action encoding: [straight, left, right, shoulder_l, shoulder_r, accel, brake, boost] = 8 dims
        self._action_encoding_dim = 8
        self.dim = 5 + self.n_preview * 3 + self.n_history * self._action_encoding_dim
        self._action_history = []
        self._prev_x = 0.0
        self._prev_y = 0.0

    def reset(self):
        """Clear action history (call on episode reset)."""
        self._action_history = [np.zeros(5, dtype=np.int64)] * self.n_history
        self._prev_x = 0.0
        self._prev_y = 0.0

    def build(self, info: dict, checkpoints: np.ndarray, action: np.ndarray = None) -> np.ndarray:
        """
        Build the float feature vector from RAM info and track data.

        Args:
            info: dict of RAM variables from stable-retro step
            checkpoints: (N, 2) array of track checkpoint (x, y) positions
            action: MultiDiscrete action array (5,) or None

        Returns:
            (dim,) float32 array
        """
        if action is None:
            action = np.zeros(5, dtype=np.int64)
        x = float(info.get("player_x", 0))
        y = float(info.get("player_y", 0))
        energy = float(info.get("energy", 0))
        lap = float(info.get("lap", 1))
        cp_total = float(info.get("checkpoint_total", 1))
        cp_facing = float(info.get("checkpoint_facing", 0))

        # Derive speed from position delta
        dx = x - self._prev_x
        dy = y - self._prev_y
        speed = np.sqrt(dx * dx + dy * dy)
        self._prev_x = x
        self._prev_y = y

        # Core state (5 dims) — lap is 0-indexed (0=first lap)
        core = np.array([
            speed / MAX_SPEED,
            energy / MAX_ENERGY,
            lap / 4.0,                       # normalize: 0..4 -> 0..1
            1.0 if lap >= 1 else 0.0,        # boost available from lap index 1
            cp_facing / max(cp_total, 1.0),
        ], dtype=np.float32)

        # Track preview: next N checkpoints relative to player (N * 3 dims)
        preview = self._build_track_preview(x, y, cp_facing, checkpoints)

        # Action history (n_history * action_encoding_dim dims)
        self._action_history.append(action)
        if len(self._action_history) > self.n_history:
            self._action_history = self._action_history[-self.n_history:]
        history = self._encode_action_history()

        return np.concatenate([core, preview, history]).astype(np.float32)

    def _build_track_preview(
        self, player_x: float, player_y: float, cp_idx: float,
        checkpoints: np.ndarray,
    ) -> np.ndarray:
        """
        Build relative positions of upcoming checkpoints.
        Returns (n_preview * 3,) array: [dx, dy, dist] for each upcoming checkpoint.
        """
        if checkpoints is None or len(checkpoints) == 0:
            return np.zeros(self.n_preview * 3, dtype=np.float32)

        n_cp = len(checkpoints)
        start_idx = int(cp_idx) % n_cp
        preview = np.zeros(self.n_preview * 3, dtype=np.float32)

        for i in range(self.n_preview):
            idx = (start_idx + i + 1) % n_cp
            cp_x, cp_y = checkpoints[idx]
            dx = (cp_x - player_x) / MAX_POSITION_X
            dy = (cp_y - player_y) / MAX_POSITION_Y
            dist = np.sqrt(dx * dx + dy * dy)
            preview[i * 3] = dx
            preview[i * 3 + 1] = dy
            preview[i * 3 + 2] = dist

        return preview

    def _encode_action_history(self) -> np.ndarray:
        """Encode recent actions as binary feature vectors.

        Action is MultiDiscrete([3,3,2,2,2]) stored as numpy array:
          [steer, shoulder, accel, brake, boost]

        Encoding: [straight, left, right, shoulder_l, shoulder_r, accel, brake, boost] = 8 dims
        """
        encoded = np.zeros(self.n_history * self._action_encoding_dim, dtype=np.float32)
        for i, act in enumerate(self._action_history[-self.n_history:]):
            offset = i * self._action_encoding_dim
            act = np.asarray(act)
            steer = int(act[0]) if act.ndim > 0 else 0
            shoulder = int(act[1]) if act.ndim > 0 else 0
            accel = int(act[2]) if act.ndim > 0 else 0
            brake = int(act[3]) if act.ndim > 0 else 0
            boost = int(act[4]) if act.ndim > 0 else 0

            if steer == 0:
                encoded[offset] = 1.0         # straight
            elif steer == 1:
                encoded[offset + 1] = 1.0     # left
            else:
                encoded[offset + 2] = 1.0     # right

            if shoulder == 1:
                encoded[offset + 3] = 1.0     # shoulder L
            elif shoulder == 2:
                encoded[offset + 4] = 1.0     # shoulder R

            if accel:
                encoded[offset + 5] = 1.0     # accelerate
            if brake:
                encoded[offset + 6] = 1.0     # brake
            if boost:
                encoded[offset + 7] = 1.0     # boost

        return encoded
