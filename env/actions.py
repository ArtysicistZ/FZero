"""
F-Zero SNES action mapping.

Supports two action space formats:
  - MultiDiscrete([3, 3, 2, 2, 2]) for PPO — each dimension learned independently
  - Discrete(72) for DQN/QR-DQN/IQN — flat enumeration

Dimensions:
  [0] Steer:    0=straight, 1=left, 2=right
  [1] Shoulder: 0=none, 1=L, 2=R
  [2] Accel:    0=no, 1=yes  (releasing enables blast turning)
  [3] Brake:    0=no, 1=yes  (SNES Y button)
  [4] Boost:    0=no, 1=yes  (SNES A button = Super Jet)
"""
import numpy as np
from dataclasses import dataclass

# Sub-action dimensions: [steer(3), shoulder(3), accel(2), brake(2), boost(2)]
ACTION_DIMS = [3, 3, 2, 2, 2]
N_ACTIONS_FLAT = 72  # product of ACTION_DIMS


@dataclass(frozen=True)
class ButtonCombo:
    """Represents a set of SNES buttons to press simultaneously."""
    left: bool = False
    right: bool = False
    up: bool = False
    down: bool = False
    a: bool = False      # Super Jet / Boost (SNES A button)
    b: bool = False      # Accelerate (SNES B button)
    l: bool = False      # L-shoulder lean
    r: bool = False      # R-shoulder lean
    x: bool = False      # Brake (SNES X button, duplicate of Y)
    y: bool = False      # Brake (SNES Y button)
    start: bool = False
    select: bool = False

    def to_array(self):
        """Convert to the button array format expected by stable-retro.
        Order matches SNES button ordering in stable-retro:
        B, Y, SELECT, START, UP, DOWN, LEFT, RIGHT, A, X, L, R
        """
        return [
            self.b, self.y, self.select, self.start,
            self.up, self.down, self.left, self.right,
            self.a, self.x, self.l, self.r,
        ]


def multi_discrete_to_buttons(action: np.ndarray) -> list:
    """Convert a MultiDiscrete([3,3,2,2,2]) action to a stable-retro button array.

    Args:
        action: numpy array of shape (5,) with values [steer, shoulder, accel, brake, boost]
    """
    steer, shoulder, accel, brake, boost = int(action[0]), int(action[1]), int(action[2]), int(action[3]), int(action[4])
    combo = ButtonCombo(
        left=(steer == 1),
        right=(steer == 2),
        l=(shoulder == 1),
        r=(shoulder == 2),
        b=bool(accel),
        y=bool(brake),
        a=bool(boost),
    )
    return combo.to_array()


def flat_to_multi(action_idx: int) -> np.ndarray:
    """Convert Discrete(72) index to MultiDiscrete([3,3,2,2,2]) array."""
    steer = action_idx // 24
    shoulder = (action_idx % 24) // 8
    accel = (action_idx % 8) // 4
    brake = (action_idx % 4) // 2
    boost = action_idx % 2
    return np.array([steer, shoulder, accel, brake, boost], dtype=np.int64)


def multi_to_flat(action: np.ndarray) -> int:
    """Convert MultiDiscrete([3,3,2,2,2]) array to Discrete(72) index."""
    return int(action[0]) * 24 + int(action[1]) * 8 + int(action[2]) * 4 + int(action[3]) * 2 + int(action[4])


# Legacy support: flat action table for Discrete(72) algorithms
def _build_action_table():
    """Build the flat action table: 72 discrete actions."""
    table = []
    for idx in range(N_ACTIONS_FLAT):
        multi = flat_to_multi(idx)
        combo = ButtonCombo(
            left=(multi[0] == 1), right=(multi[0] == 2),
            l=(multi[1] == 1), r=(multi[1] == 2),
            b=bool(multi[2]), y=bool(multi[3]), a=bool(multi[4]),
        )
        table.append(combo)
    return table


ACTION_TABLE = _build_action_table()


def action_to_buttons(action_idx: int) -> list:
    """Convert a Discrete(72) action index to a stable-retro button array."""
    return ACTION_TABLE[action_idx].to_array()


def action_to_description(action_idx: int) -> str:
    """Human-readable description of a flat action for debug overlay."""
    combo = ACTION_TABLE[action_idx]
    parts = []
    if combo.left:
        parts.append("L")
    elif combo.right:
        parts.append("R")
    else:
        parts.append("-")
    if combo.l:
        parts.append("+SL")
    elif combo.r:
        parts.append("+SR")
    if not combo.b:
        parts.append("+NoAcc")
    if combo.y:
        parts.append("+Brake")
    if combo.a:
        parts.append("+Boost")
    return "".join(parts)
