"""
Debug overlay for F-Zero gameplay visualization.

Renders game state information on top of captured game frames
for debugging reward functions and agent behavior.
"""
import cv2
import numpy as np

from env.actions import action_to_description


def draw_overlay(frame: np.ndarray, info: dict) -> np.ndarray:
    """
    Draw debug information overlay on a game frame.

    Args:
        frame: (H, W, 3) uint8 RGB frame
        info: dict containing RAM variables and reward components

    Returns:
        (H, W, 3) uint8 frame with overlay text
    """
    frame = frame.copy()
    h, w = frame.shape[:2]

    # Prepare text lines
    energy = info.get("energy", 0)
    lap = info.get("lap", 1)
    cp_facing = info.get("checkpoint_facing", 0)
    cp_total = info.get("checkpoint_total", 0)
    action = info.get("action", 0)
    step = info.get("step_count", 0)

    components = info.get("reward_components", {})
    total_reward = sum(components.values()) if components else 0.0

    timer_min = info.get("race_timer_min", 0)
    # BCD decode timer fields
    _raw_sec = info.get("race_timer_sec", 0)
    _raw_csec = info.get("race_timer_csec", 0)
    timer_sec = ((_raw_sec >> 4) & 0xF) * 10 + (_raw_sec & 0xF)
    timer_csec = ((_raw_csec >> 4) & 0xF) * 10 + (_raw_csec & 0xF)

    lines = [
        f"NRG: {energy:4d}  LAP: {lap}/5",
        f"CP: {cp_facing}/{cp_total}",
        f"TIME: {timer_min}:{timer_sec:02d}.{timer_csec:02d}",
        f"ACT: {action_to_description(action)}",
        f"RWD: {total_reward:+.2f}",
        f"Step: {step}",
    ]

    # Draw semi-transparent background
    overlay_h = len(lines) * 18 + 10
    cv2.rectangle(frame, (0, 0), (w, overlay_h), (0, 0, 0), -1)
    # Blend for semi-transparency would require alpha — keep opaque for simplicity

    # Draw text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    color = (0, 255, 0)
    for i, line in enumerate(lines):
        y = 15 + i * 18
        cv2.putText(frame, line, (5, y), font, font_scale, color, 1, cv2.LINE_AA)

    return frame
