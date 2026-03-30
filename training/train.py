"""
Main training entrypoint for F-Zero RL.

Usage:
    python -m training.train --algo ppo --timesteps 10000000
    python -m training.train --algo dqn --timesteps 10000000 --n-envs 4
    python -m training.train --algo qrdqn --n-envs 1
    python -m training.train --algo iqn --n-envs 80
"""
import argparse
import os
import sys

from datetime import datetime

from training.config import (
    TrainingConfig, RUNS_DIR, ROM_DIR, ROM_FILENAME,
    PROJECT_ROOT,
)


def _load_env_file():
    """Load .env file if it exists (for WANDB_API_KEY etc.)."""
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())


def _check_rom():
    """Verify the F-Zero ROM file exists before starting training."""
    rom_path = ROM_DIR / ROM_FILENAME
    if not rom_path.exists():
        print(f"ERROR: ROM file not found at {rom_path}")
        print(f"Please place '{ROM_FILENAME}' in the '{ROM_DIR}' directory.")
        print("You can obtain the ROM from online ROM archive sites.")
        sys.exit(1)


def _setup_run_dir(algo: str, config: TrainingConfig):
    """Create a timestamped run directory with all output subdirs.

    Structure:
      runs/
        ppo-80env-bs1024-20260329-053200/
          checkpoints/    ← periodic model saves
          best/           ← best lap time model
          final/          ← final model after training
          videos/         ← gameplay recordings
          logs/           ← tensorboard logs
    """
    algo_cfg = getattr(config, algo)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{algo}-{config.env.n_envs}env-bs{algo_cfg.batch_size}-{timestamp}"
    run_dir = RUNS_DIR / run_name

    dirs = {
        "run": run_dir,
        "checkpoints": run_dir / "checkpoints",
        "best": run_dir / "best",
        "final": run_dir / "final",
        "videos": run_dir / "videos",
        "logs": run_dir / "logs",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    return run_name, dirs


def train(algo: str = "ppo", config: TrainingConfig = None, load_path: str = None):
    """
    Train an RL agent on F-Zero.

    Args:
        algo: "ppo", "dqn", "qrdqn", or "iqn"
        config: Training configuration (uses defaults if None)
        load_path: path to a saved model to resume training from

    Returns:
        Path to the saved model
    """
    config = config or TrainingConfig()

    _load_env_file()
    _check_rom()
    run_name, dirs = _setup_run_dir(algo, config)

    # Import here to avoid slow imports when just checking CLI args
    from stable_baselines3 import PPO, DQN
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    from stable_baselines3.common.vec_env import VecVideoRecorder
    if algo == "qrdqn":
        from sb3_contrib import QRDQN

    from env import make_fzero_env
    from network import FZeroFeatureExtractor
    from training.callbacks import RewardLoggingCallback, BestLapCallback

    # Create vectorized environment
    # DQN and QR-DQN require Discrete action space (flat_actions=True)
    flat_actions = algo in ("dqn", "qrdqn")
    env = make_fzero_env(
        n_envs=config.env.n_envs,
        render_mode="rgb_array",
        env_config=config.env,
        reward_config=config.reward,
        flat_actions=flat_actions,
    )

    # Determine total timesteps for this algorithm
    algo_cfg = getattr(config, algo)
    total_timesteps = algo_cfg.total_timesteps

    # Wrap with video recorder (skip if freq is very high — effectively disabled)
    if config.video_record_freq < total_timesteps:
        env = VecVideoRecorder(
            env,
            video_folder=str(dirs["videos"]),
            record_video_trigger=lambda step: step % config.video_record_freq == 0,
            video_length=2000,  # ~2 min of gameplay at 15fps
        )

    # Policy kwargs: use our custom feature extractor
    policy_kwargs = {
        "features_extractor_class": FZeroFeatureExtractor,
        "features_extractor_kwargs": {"cfg": config.network},
    }

    # Create the RL model (or load from checkpoint)
    if algo == "ppo":
        if load_path:
            print(f"Loading model from {load_path}")
            model = PPO.load(
                load_path, env=env,
                learning_rate=config.ppo.learning_rate,
                gamma=config.ppo.gamma,
                clip_range=config.ppo.clip_range,
                ent_coef=config.ppo.ent_coef,
                n_steps=config.ppo.n_steps,
                batch_size=config.ppo.batch_size,
                n_epochs=config.ppo.n_epochs,
                tensorboard_log=str(dirs["logs"]),
            )
        else:
            model = PPO(
                "MultiInputPolicy",
                env,
                learning_rate=config.ppo.learning_rate,
                n_steps=config.ppo.n_steps,
                batch_size=config.ppo.batch_size,
                n_epochs=config.ppo.n_epochs,
                gamma=config.ppo.gamma,
                gae_lambda=config.ppo.gae_lambda,
                clip_range=config.ppo.clip_range,
                ent_coef=config.ppo.ent_coef,
                vf_coef=config.ppo.vf_coef,
                max_grad_norm=config.ppo.max_grad_norm,
                policy_kwargs=policy_kwargs,
                verbose=1,
                tensorboard_log=str(dirs["logs"]),
            )
    elif algo == "dqn":
        model = DQN(
            "MultiInputPolicy",
            env,
            learning_rate=config.dqn.learning_rate,
            buffer_size=config.dqn.buffer_size,
            learning_starts=config.dqn.learning_starts,
            batch_size=config.dqn.batch_size,
            gamma=config.dqn.gamma,
            target_update_interval=config.dqn.target_update_interval,
            exploration_fraction=config.dqn.exploration_fraction,
            exploration_final_eps=config.dqn.exploration_final_eps,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=str(dirs["logs"]),
        )
    elif algo == "qrdqn":
        qrdqn_policy_kwargs = {
            **policy_kwargs,
            "n_quantiles": config.qrdqn.n_quantiles,
        }
        model = QRDQN(
            "MultiInputPolicy",
            env,
            learning_rate=config.qrdqn.learning_rate,
            buffer_size=config.qrdqn.buffer_size,
            learning_starts=config.qrdqn.learning_starts,
            batch_size=config.qrdqn.batch_size,
            gamma=config.qrdqn.gamma,
            target_update_interval=config.qrdqn.target_update_interval,
            exploration_fraction=config.qrdqn.exploration_fraction,
            exploration_final_eps=config.qrdqn.exploration_final_eps,
            policy_kwargs=qrdqn_policy_kwargs,
            verbose=1,
            tensorboard_log=str(dirs["logs"]),
        )
    elif algo == "iqn":
        # IQN uses its own training loop, not SB3
        from training.iqn_trainer import IQNTrainer

        # Setup W&B before training
        if config.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=config.wandb_project,
                    name=run_name,
                    config={
                        "algorithm": "iqn",
                        "env": vars(config.env),
                        "reward": vars(config.reward),
                        "network": vars(config.network),
                        "iqn": vars(config.iqn),
                    },
                )
            except ImportError:
                print("WARNING: wandb not installed, skipping W&B logging.")

        trainer = IQNTrainer(env, config, run_name, dirs)
        trainer.train()

        env.close()
        if config.use_wandb:
            try:
                wandb.finish()
            except Exception:
                pass
        return str(dirs["final"] / "fzero_iqn_final.pt")
    else:
        raise ValueError(f"Unknown algorithm: {algo}. Choose 'ppo', 'dqn', 'qrdqn', or 'iqn'.")

    # Setup W&B if enabled (SB3 algorithms only — IQN handles its own W&B above)
    if config.use_wandb:
        try:
            import wandb
            from wandb.integration.sb3 import WandbCallback
            wandb.init(
                project=config.wandb_project,
                name=run_name,
                config={
                    "algorithm": algo,
                    "env": vars(config.env),
                    "reward": vars(config.reward),
                    "network": vars(config.network),
                    algo: vars(getattr(config, algo)),
                },
                sync_tensorboard=True,
            )
            wandb_callback = WandbCallback(
                verbose=1,
                model_save_path=str(dirs["checkpoints"]),
                model_save_freq=config.save_freq,
            )
        except ImportError:
            print("WARNING: wandb not installed, skipping W&B logging.")
            wandb_callback = None
    else:
        wandb_callback = None

    # Assemble callbacks
    callbacks = [
        RewardLoggingCallback(verbose=1),
        BestLapCallback(save_dir=str(dirs["best"]), verbose=1),
        CheckpointCallback(
            save_freq=config.save_freq,
            save_path=str(dirs["checkpoints"]),
            name_prefix=f"fzero_{algo}",
        ),
    ]
    if wandb_callback:
        callbacks.append(wandb_callback)

    # Train
    print(f"Starting {algo.upper()} training for {total_timesteps} timesteps...")
    print(f"  Run: {run_name}")
    print(f"  Run dir: {dirs['run']}")
    print(f"  Environments: {config.env.n_envs}")
    print(f"  Frameskip: {config.env.frameskip}")

    model.learn(
        total_timesteps=total_timesteps,
        callback=CallbackList(callbacks),
        log_interval=config.log_interval,
    )

    # Save final model
    final_path = str(dirs["final"] / f"fzero_{algo}_final")
    model.save(final_path)
    print(f"Training complete. Final model saved to {final_path}")
    print(f"All outputs in: {dirs['run']}")

    env.close()
    if config.use_wandb:
        try:
            wandb.finish()
        except Exception:
            pass

    return final_path


def main():
    parser = argparse.ArgumentParser(description="Train F-Zero RL agent")
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "dqn", "qrdqn", "iqn"],
                        help="RL algorithm to use")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Total training timesteps (overrides config)")
    parser.add_argument("--n-envs", type=int, default=None,
                        help="Number of parallel environments (overrides config)")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable W&B logging")
    parser.add_argument("--load", type=str, default=None,
                        help="Path to saved model to resume training from")
    args = parser.parse_args()

    config = TrainingConfig()
    if args.timesteps:
        getattr(config, args.algo).total_timesteps = args.timesteps
    if args.n_envs:
        config.env.n_envs = args.n_envs
    if args.no_wandb:
        config.use_wandb = False

    train(algo=args.algo, config=config, load_path=args.load)


if __name__ == "__main__":
    main()
