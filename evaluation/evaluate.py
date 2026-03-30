"""
Evaluate a trained F-Zero RL model.

Runs the model for N episodes and collects metrics:
  - Average/best/worst race time
  - Lap completion rate
  - Average reward per episode
"""
from stable_baselines3 import PPO, DQN

from env import make_fzero_env
from training.config import TrainingConfig


def evaluate(model_path: str, algo: str = "ppo", n_episodes: int = 10,
             config: TrainingConfig = None):
    """
    Evaluate a trained model on F-Zero.

    Args:
        model_path: Path to the saved model file
        algo: "ppo" or "dqn"
        n_episodes: Number of evaluation episodes
        config: Training config (uses defaults if None)

    Returns:
        dict with evaluation metrics
    """
    config = config or TrainingConfig()

    env = make_fzero_env(
        n_envs=1,
        render_mode="rgb_array",
        env_config=config.env,
        reward_config=config.reward,
    )

    # Load model
    ModelClass = PPO if algo == "ppo" else DQN
    model = ModelClass.load(model_path, env=env)

    results = {
        "race_times": [],
        "total_rewards": [],
        "laps_completed": [],
        "termination_reasons": [],
    }

    for ep in range(n_episodes):
        obs = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = env.step(action)
            total_reward += reward[0]
            done = dones[0]

            if done:
                info = infos[0]
                lap = info.get("lap", 1)
                # Use BCD-decoded race_time from fzero_env if available
                race_time = info.get("race_time", None)
                if race_time is None:
                    # Fallback: BCD decode here
                    def _bcd(v): return ((v >> 4) & 0xF) * 10 + (v & 0xF)
                    race_time = info.get("race_timer_min", 0) * 60.0 + _bcd(info.get("race_timer_sec", 0)) + _bcd(info.get("race_timer_csec", 0)) / 100.0

                results["race_times"].append(race_time)
                results["total_rewards"].append(total_reward)
                results["laps_completed"].append(min(lap, 5))

                if lap > 5:
                    results["termination_reasons"].append("finished")
                elif info.get("energy", 0) <= 0:
                    results["termination_reasons"].append("death")
                else:
                    results["termination_reasons"].append("timeout/stuck")

    env.close()

    # Compute summary stats
    if results["race_times"]:
        results["best_race_time"] = min(results["race_times"])
        results["avg_race_time"] = sum(results["race_times"]) / len(results["race_times"])
        results["avg_reward"] = sum(results["total_rewards"]) / len(results["total_rewards"])
        results["completion_rate"] = (
            results["termination_reasons"].count("finished") / n_episodes
        )

    return results
