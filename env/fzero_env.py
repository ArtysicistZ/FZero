"""
F-Zero SNES Gymnasium environment wrapper.

Wraps a Stable-Retro RetroEnv to provide:
  - Dual-input observation space (screen + float features)
  - Dense shaped reward from RAM state
  - Episode termination on death, timeout, or stuck
  - Frame stacking for motion perception

Action space is MultiDiscrete([3, 3, 2, 2, 2]):
  [steer, shoulder, accel, brake, boost]
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env.actions import multi_discrete_to_buttons, multi_to_flat, flat_to_multi, ACTION_DIMS, N_ACTIONS_FLAT
from env.observations import FrameProcessor, FloatFeatureBuilder
from env.rewards import RewardCalculator
from training.config import EnvConfig, RewardConfig, INTEGRATION_DIR


class FZeroEnv(gym.Env):
    """
    Custom Gymnasium wrapper for F-Zero SNES via Stable-Retro.

    Observation space: Dict with
      - "screen": (frame_stack, 120, 160) float32
      - "float": (float_dim,) float32
    Action space: MultiDiscrete([3, 3, 2, 2, 2])
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, game_path: str, render_mode: str = "rgb_array",
                 env_config: EnvConfig = None, reward_config: RewardConfig = None):
        super().__init__()

        self._env_config = env_config or EnvConfig()
        self._reward_config = reward_config or RewardConfig()

        # Create the underlying Stable-Retro environment
        import stable_retro
        self._retro_env = stable_retro.make(
            game=game_path,
            state=stable_retro.State.NONE,
            inttype=stable_retro.data.Integrations.CUSTOM,
            use_restricted_actions=stable_retro.Actions.ALL,
            obs_type=stable_retro.Observations.IMAGE,
            render_mode=render_mode,
        )

        # Load save state data for manual restore on reset
        import gzip
        state_file = INTEGRATION_DIR / "FZero-Snes" / "MuteCity1.state"
        with gzip.open(str(state_file.resolve()), "rb") as f:
            self._initial_state_data = f.read()

        # Submodules
        self._frame_processor = FrameProcessor(self._env_config)
        self._float_builder = FloatFeatureBuilder(self._env_config)
        self._reward_calc = RewardCalculator(self._reward_config)

        # Track checkpoint data (loaded once, used for track preview)
        self._checkpoints = None

        # Define observation space
        screen_shape = (
            self._env_config.frame_stack,
            self._env_config.screen_height,
            self._env_config.screen_width,
        )
        self.observation_space = spaces.Dict({
            "screen": spaces.Box(low=0.0, high=1.0, shape=screen_shape, dtype=np.float32),
            "float": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self._float_builder.dim,), dtype=np.float32,
            ),
        })

        # Action space: MultiDiscrete — each dimension learned independently
        self.action_space = spaces.MultiDiscrete(ACTION_DIMS)

        # Internal state
        self._step_count = 0
        self._last_info = {}
        self._last_reward_components = {}
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        """Reset the environment and return initial observation."""
        super().reset(seed=seed)

        self._retro_env.reset()
        self._retro_env.em.set_state(self._initial_state_data)
        obs_raw, _, _, _, info = self._retro_env.step([0]*12)

        self._frame_processor.reset()
        self._float_builder.reset()
        self._reward_calc.reset()
        self._step_count = 0

        # Load checkpoint coordinates from RAM (once)
        if self._checkpoints is None:
            n_cp = int(info.get("checkpoint_total", 0))
            if n_cp > 0:
                pts = []
                for i in range(n_cp):
                    cx = float(info.get(f"cp_x_{i}", 0))
                    cy = float(info.get(f"cp_y_{i}", 0))
                    pts.append((cx, cy))
                self._checkpoints = np.array(pts, dtype=np.float64)

        screen = self._frame_processor.process_frame(obs_raw)
        float_obs = self._float_builder.build(info, self._checkpoints, action=np.zeros(5, dtype=np.int64))

        self._last_info = info
        observation = {"screen": screen, "float": float_obs}
        return observation, info

    def step(self, action):
        """
        Take one step in the environment.

        Args:
            action: numpy array of shape (5,) — MultiDiscrete([3,3,2,2,2])

        Returns:
            observation, reward, terminated, truncated, info
        """
        action = np.asarray(action, dtype=np.int64)
        buttons = multi_discrete_to_buttons(action)

        # Step the emulator with frameskip
        obs_raw = None
        info = {}
        for _ in range(self._env_config.frameskip):
            obs_raw, r, terminated_retro, truncated_retro, info = self._retro_env.step(buttons)
            if terminated_retro or truncated_retro:
                break

        self._step_count += 1

        # Build observation
        screen = self._frame_processor.process_frame(obs_raw)
        float_obs = self._float_builder.build(info, self._checkpoints, action=action)
        observation = {"screen": screen, "float": float_obs}

        # Compute shaped reward
        flat_action = multi_to_flat(action)
        reward, components, stuck = self._reward_calc.compute(info, action=flat_action)

        # Determine termination
        energy = info.get("energy", 0)
        lap = info.get("lap", 0)
        race_finished = lap >= 5
        terminated = energy <= 0 or race_finished

        # Time bonus — awarded at episode end based on speed
        # Full bonus for completing 5 laps; partial bonus for dying mid-race.
        # Partial bonus encourages boost exploration: dying fast on lap 4
        # is rewarded more than safe slow driving without boost.
        if terminated and lap > 0:
            t_min = info.get("race_timer_min", 0)
            t_sec = info.get("race_timer_sec", 0)
            t_csec = info.get("race_timer_csec", 0)
            elapsed = t_min * 60.0 + t_sec + t_csec / 100.0
            laps_done = min(lap, 5)

            if race_finished:
                bonus = self._reward_calc.compute_time_bonus(elapsed)
                info["race_time"] = elapsed
            else:
                projected_time = elapsed * 5.0 / laps_done
                bonus = self._reward_calc.compute_time_bonus(projected_time) * (laps_done / 5.0)

            components["time_bonus"] = bonus
            reward += bonus

        self._last_reward_components = components

        # Determine truncation
        truncated = (
            self._step_count >= self._env_config.max_episode_steps
            or stuck
        )

        # Enrich info dict for logging
        info["reward_components"] = components
        info["time_penalty_value"] = self._reward_config.time_penalty
        info["step_count"] = self._step_count
        info["action"] = flat_action
        self._last_info = info

        return observation, reward, terminated, truncated, info

    def render(self):
        return self._retro_env.render()

    def close(self):
        self._retro_env.close()

    @property
    def last_reward_components(self) -> dict:
        return self._last_reward_components


class FlatDiscreteWrapper(gym.ActionWrapper):
    """Wraps MultiDiscrete([3,3,2,2,2]) as Discrete(72) for DQN/QR-DQN.

    Converts flat action index to MultiDiscrete array before passing to env.
    """

    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Discrete(N_ACTIONS_FLAT)

    def action(self, action):
        return flat_to_multi(int(action))
