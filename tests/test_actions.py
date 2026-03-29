"""Tests for the F-Zero action mapping module."""
import pytest

from env.actions import ACTION_TABLE, action_to_buttons, action_to_description


def test_action_table_has_18_entries():
    assert len(ACTION_TABLE) == 18


def test_action_table_all_accelerate():
    """Every action should have the B button (accelerate) pressed."""
    for i, combo in enumerate(ACTION_TABLE):
        assert combo.b, f"Action {i} does not hold accelerate"


def test_action_to_buttons_returns_12_elements():
    """SNES has 12 buttons in stable-retro format."""
    for i in range(18):
        buttons = action_to_buttons(i)
        assert len(buttons) == 12, f"Action {i} returned {len(buttons)} buttons"


def test_action_to_buttons_first_is_straight():
    """Action 0 should be straight (no steer, no shoulder, no boost)."""
    buttons = action_to_buttons(0)
    # B=True, Y=False, SELECT=False, START=False,
    # UP=False, DOWN=False, LEFT=False, RIGHT=False,
    # A=False, X=False, L=False, R=False
    assert buttons[0] is True   # B (accelerate)
    assert buttons[6] is False  # LEFT
    assert buttons[7] is False  # RIGHT
    assert buttons[10] is False  # L
    assert buttons[11] is False  # R
    assert buttons[1] is False  # Y (boost)


def test_action_descriptions_not_empty():
    for i in range(18):
        desc = action_to_description(i)
        assert isinstance(desc, str)
        assert len(desc) > 0


def test_no_duplicate_actions():
    """All 18 actions should produce unique button combinations."""
    seen = set()
    for i in range(18):
        buttons = tuple(action_to_buttons(i))
        assert buttons not in seen, f"Action {i} is a duplicate"
        seen.add(buttons)
