"""E2E test for IQN trainer with all features (n-step, AMP, schedules)."""
from env import make_fzero_env
from training.config import TrainingConfig
from training.iqn_trainer import IQNTrainer, _piecewise_linear
from pathlib import Path

cfg = TrainingConfig().iqn

print("LR Schedule:")
for s in [0, 1000000, 3000000, 15000000]:
    print(f"  step={s} lr={_piecewise_linear(cfg.lr_schedule, s)}")

print("Gamma Schedule:")
for s in [0, 1500000, 2500000, 5000000]:
    print(f"  step={s} gamma={_piecewise_linear(cfg.gamma_schedule, s)}")

print("Epsilon Schedule:")
for s in [0, 50000, 300000, 3000000]:
    print(f"  step={s} eps={_piecewise_linear(cfg.epsilon_schedule, s)}")

config = TrainingConfig()
config.env.n_envs = 1
config.iqn.learning_starts = 10
config.iqn.batch_size = 4
config.iqn.gradient_steps_per_env_step = 2

env = make_fzero_env(n_envs=1, env_config=config.env, reward_config=config.reward)
dirs = {k: Path("/tmp/iqn_e2e/" + k) for k in ["run", "checkpoints", "best", "final", "videos", "logs"]}
for d in dirs.values():
    d.mkdir(parents=True, exist_ok=True)

trainer = IQNTrainer(env, config, "test", dirs)
n_params = sum(p.numel() for p in trainer.online_net.parameters())
print(f"Params: {n_params}")
print(f"AMP: {trainer.use_amp}")
print(f"N-step: {config.iqn.n_step_returns}")

obs = env.reset()
for step in range(30):
    eps = trainer._get_schedule_value(cfg.epsilon_schedule, step)
    actions = trainer._select_actions(obs, eps)
    next_obs, rewards, dones, infos = env.step(actions)

    trainer.n_step_buffers[0].add(
        obs["screen"][0], obs["float"][0], actions[0], rewards[0], dones[0]
    )
    if trainer.n_step_buffers[0].is_ready():
        t = trainer.n_step_buffers[0].pop(
            next_obs["screen"][0], next_obs["float"][0], dones[0]
        )
        trainer.buffer.add(*t)
    if dones[0]:
        for t in trainer.n_step_buffers[0].flush():
            trainer.buffer.add(*t)

    obs = next_obs

    if trainer.buffer.size >= 10:
        gamma = trainer._update_gamma(step)
        trainer._update_lr(step)
        loss = trainer._train_step(gamma)
        if step == 29:
            print(f"Step {step}: loss={loss:.4f}, buf_size={trainer.buffer.size}")

env.close()
print("ALL TESTS PASSED")
