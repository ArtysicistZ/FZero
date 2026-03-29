"""
Navigate F-Zero SNES menus to reach the Mute City I race start,
then save the emulator state (gzip compressed) for training.
"""
import gzip
import os

import cv2
import numpy as np
import stable_retro


def main():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_dir = os.path.join(project_dir, "env")
    video_dir = os.path.join(project_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)

    stable_retro.data.add_custom_integration(env_dir)

    env = stable_retro.make(
        game="FZero-Snes",
        state=stable_retro.State.NONE,
        inttype=stable_retro.data.Integrations.CUSTOM,
        use_restricted_actions=stable_retro.Actions.ALL,
        render_mode="rgb_array",
    )

    obs, info = env.reset()

    # SNES buttons: B, Y, SELECT, START, UP, DOWN, LEFT, RIGHT, A, X, L, R
    NONE  = [0]*12
    START = [0,0,0,1, 0,0,0,0, 0,0,0,0]
    B_BTN = [1,0,0,0, 0,0,0,0, 0,0,0,0]
    A_BTN = [0,0,0,0, 0,0,0,0, 1,0,0,0]
    RIGHT = [0,0,0,0, 0,0,0,1, 0,0,0,0]
    DOWN  = [0,0,0,0, 0,1,0,0, 0,0,0,0]

    def step_n(action, n):
        nonlocal obs, info
        for _ in range(n):
            obs, _, _, _, info = env.step(action)

    def wait(n=60):
        step_n(NONE, n)

    def press(btn, hold=5, pause=30):
        step_n(btn, hold)
        wait(pause)

    def save_frame(name):
        path = os.path.join(video_dir, name)
        cv2.imwrite(path, cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
        print(f"  Saved: {name}")

    def ram_str():
        keys = ["player_x", "player_y", "energy", "lap", "checkpoint_facing", "checkpoint_total"]
        return " ".join(f"{k}={info.get(k, '?')}" for k in keys)

    print("=== F-Zero Save State Creator ===")
    print()

    # Title screen
    print("1. Waiting for title screen...")
    wait(200)
    save_frame("01_title.png")

    # Press START at title
    print("2. Press START...")
    press(START)
    save_frame("02_after_start.png")

    # Grand Prix mode select - press B to confirm
    print("3. Select Grand Prix mode...")
    press(B_BTN)
    save_frame("03_gp_select.png")

    # Difficulty/class select - press B
    print("4. Select difficulty...")
    press(B_BTN)
    save_frame("04_difficulty.png")

    # Car select - press B
    print("5. Select car...")
    press(B_BTN)
    save_frame("05_car.png")

    # More confirmations
    print("6. Confirming selections...")
    for i in range(5):
        press(B_BTN, hold=3, pause=40)
        print(f"   {ram_str()}")

    save_frame("06_loading.png")

    # Wait for race to load
    print("7. Waiting for race to load...")
    wait(300)
    print(f"   {ram_str()}")
    save_frame("07_race_loading.png")

    # If there's a "GIVE UP" dialog, press B on "NO"
    # First move to NO (right), then press B
    print("8. Dismissing any dialogs...")
    press(RIGHT, hold=3, pause=10)
    press(B_BTN, hold=3, pause=30)
    press(A_BTN, hold=3, pause=30)
    print(f"   {ram_str()}")
    save_frame("08_after_dialog.png")

    # Wait for countdown (3, 2, 1, GO)
    print("9. Waiting for race countdown...")
    wait(300)
    print(f"   {ram_str()}")
    save_frame("09_race_ready.png")

    # Hold accelerate briefly to confirm we're racing
    print("10. Testing controls...")
    step_n(B_BTN, 30)
    print(f"   {ram_str()}")
    save_frame("10_racing.png")

    # Save state as gzip (stable-retro expects gzip format)
    state_data = env.em.get_state()
    state_path = os.path.join(env_dir, "FZero-Snes", "MuteCity1.state")
    with gzip.open(state_path, "wb") as f:
        f.write(state_data)
    print(f"\nState saved: {state_path} ({len(state_data)} bytes, gzipped)")

    # Close first env before creating second (retro limitation: one emulator per process)
    env.close()

    # Verify state loads
    print("\nVerifying state loads...")
    env2 = stable_retro.make(
        game="FZero-Snes",
        state="MuteCity1",
        inttype=stable_retro.data.Integrations.CUSTOM,
        use_restricted_actions=stable_retro.Actions.ALL,
        render_mode="rgb_array",
    )
    obs2, info2 = env2.reset()
    keys = ["player_x", "player_y", "energy", "lap", "checkpoint_facing"]
    print("  Loaded! " + " ".join(f"{k}={info2.get(k, '?')}" for k in keys))

    cv2.imwrite(
        os.path.join(video_dir, "11_verified_state.png"),
        cv2.cvtColor(obs2, cv2.COLOR_RGB2BGR),
    )
    print("  Saved: 11_verified_state.png")
    env2.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
