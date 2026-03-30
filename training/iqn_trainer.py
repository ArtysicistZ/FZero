"""
IQN training loop for F-Zero RL.

Self-contained trainer that handles:
  - Replay buffer with n-step returns (uniform sampling, float16 storage)
  - IQN quantile regression loss (Huber pinball, branching dueling)
  - Soft target network sync (Polyak averaging)
  - Piecewise linear schedules for epsilon, LR, gamma (Linesight-style)
  - Mixed precision training (AMP)
  - Integration with VecEnv and W&B logging
"""
import copy
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from training.config import TrainingConfig
from network.iqn import IQNNetwork
from env.actions import ACTION_DIMS


# ============================================================
# Piecewise Linear Schedule
# ============================================================
def _piecewise_linear(schedule: list, step: int) -> float:
    """Interpolate a piecewise linear schedule [(step, value), ...]."""
    if step >= schedule[-1][0]:
        return schedule[-1][1]
    for i in range(len(schedule) - 1):
        s0, v0 = schedule[i]
        s1, v1 = schedule[i + 1]
        if s0 <= step < s1:
            frac = (step - s0) / (s1 - s0)
            return v0 + frac * (v1 - v0)
    return schedule[0][1]


# ============================================================
# Replay Buffer with N-Step Returns
# ============================================================
class NStepBuffer:
    """Per-env transition accumulator for n-step returns."""

    def __init__(self, n_steps: int, gamma: float):
        self.n_steps = n_steps
        self.gamma = gamma
        self.states_screen = []
        self.states_float = []
        self.actions = []
        self.rewards = []
        self.dones = []

    def add(self, screen, floats, action, reward, done):
        self.states_screen.append(screen)
        self.states_float.append(floats)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

    def is_ready(self):
        return len(self.rewards) >= self.n_steps

    def pop(self, next_screen, next_floats, next_done):
        """Compute n-step return and return the transition.

        Returns:
            (screen_0, float_0, action_0, n_step_reward, next_screen, next_float, gamma_n_done)
            gamma_n_done: gamma^n * (1 - any_done_in_window)
        """
        n_step_reward = 0.0
        gamma_power = 1.0
        any_done = False
        for i in range(self.n_steps):
            n_step_reward += gamma_power * self.rewards[i]
            if self.dones[i]:
                any_done = True
                break
            gamma_power *= self.gamma

        # If episode ended within n steps, next_state is the terminal state's next
        # (which VecEnv auto-resets to new episode — masked by done=True anyway)
        if any_done:
            # Find the done step and use its next state
            for i in range(self.n_steps):
                if self.dones[i]:
                    if i + 1 < len(self.states_screen):
                        next_screen = self.states_screen[i + 1]
                        next_floats = self.states_float[i + 1]
                    break
            gamma_n_done = 0.0  # terminal — no bootstrap
        else:
            gamma_n_done = gamma_power  # gamma^n

        screen_0 = self.states_screen[0]
        float_0 = self.states_float[0]
        action_0 = self.actions[0]

        # Pop first element
        self.states_screen.pop(0)
        self.states_float.pop(0)
        self.actions.pop(0)
        self.rewards.pop(0)
        self.dones.pop(0)

        return screen_0, float_0, action_0, n_step_reward, next_screen, next_floats, gamma_n_done

    def flush(self):
        """Flush remaining transitions on episode end (shorter than n_steps)."""
        results = []
        while len(self.rewards) > 0:
            n_step_reward = 0.0
            gamma_power = 1.0
            for i in range(len(self.rewards)):
                n_step_reward += gamma_power * self.rewards[i]
                if self.dones[i]:
                    break
                gamma_power *= self.gamma

            # Last stored screen/float as next_state (masked by done anyway)
            last_screen = self.states_screen[-1]
            last_float = self.states_float[-1]

            results.append((
                self.states_screen[0], self.states_float[0], self.actions[0],
                n_step_reward, last_screen, last_float, 0.0  # terminal
            ))
            self.states_screen.pop(0)
            self.states_float.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.dones.pop(0)

        return results


class ReplayBuffer:
    """Uniform replay buffer with float16 storage."""

    def __init__(self, capacity: int, screen_shape: tuple, float_dim: int, n_action_dims: int):
        self.capacity = capacity
        self.pos = 0
        self.size = 0

        self.screens = np.zeros((capacity, *screen_shape), dtype=np.float16)
        self.floats = np.zeros((capacity, float_dim), dtype=np.float16)
        self.actions = np.zeros((capacity, n_action_dims), dtype=np.int8)
        self.rewards = np.zeros(capacity, dtype=np.float32)  # keep float32 for n-step sums
        self.next_screens = np.zeros((capacity, *screen_shape), dtype=np.float16)
        self.next_floats = np.zeros((capacity, float_dim), dtype=np.float16)
        self.gamma_n_dones = np.zeros(capacity, dtype=np.float32)  # gamma^n * (1-done)

    def add(self, screen, floats, action, reward, next_screen, next_floats, gamma_n_done):
        self.screens[self.pos] = screen
        self.floats[self.pos] = floats
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_screens[self.pos] = next_screen
        self.next_floats[self.pos] = next_floats
        self.gamma_n_dones[self.pos] = gamma_n_done
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.screens[idxs].astype(np.float32)),
            torch.from_numpy(self.floats[idxs].astype(np.float32)),
            torch.from_numpy(self.actions[idxs].astype(np.int64)),
            torch.from_numpy(self.rewards[idxs]),
            torch.from_numpy(self.next_screens[idxs].astype(np.float32)),
            torch.from_numpy(self.next_floats[idxs].astype(np.float32)),
            torch.from_numpy(self.gamma_n_dones[idxs]),
        )


# ============================================================
# IQN Quantile Regression Loss (Branching)
# ============================================================
def iqn_loss_branching(targets_list, outputs_list, tau_outputs, iqn_n, batch_size, kappa):
    """Compute IQN quantile regression loss for branching architecture."""
    tau = tau_outputs.reshape(iqn_n, batch_size, 1).transpose(0, 1)  # (B, N, 1)
    tau_expanded = tau[:, None, :, :].expand(-1, iqn_n, -1, -1)  # (B, N_tgt, N_out, 1)

    total_loss = torch.zeros(batch_size, device=targets_list[0].device)
    for targets, outputs in zip(targets_list, outputs_list):
        td_error = targets[:, :, None, :] - outputs[:, None, :, :]  # (B, N, N, 1)
        huber = torch.where(
            td_error.abs() < kappa,
            0.5 / kappa * td_error ** 2,
            td_error.abs() - 0.5 * kappa,
        )
        pinball = torch.where(td_error < 0, 1 - tau_expanded, tau_expanded) * huber
        total_loss = total_loss + pinball.sum(dim=2).mean(dim=1)[:, 0]

    return total_loss / len(targets_list)


# ============================================================
# IQN Trainer
# ============================================================
class IQNTrainer:
    """Complete IQN training loop with VecEnv integration."""

    def __init__(self, env, config: TrainingConfig, run_name: str, dirs: dict):
        self.env = env
        self.config = config
        self.cfg = config.iqn
        self.dirs = dirs
        self.run_name = run_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        screen_shape = env.observation_space["screen"].shape
        float_dim = env.observation_space["float"].shape[0]
        self.n_envs = env.num_envs

        # Networks
        self.online_net = IQNNetwork(
            float_input_dim=float_dim,
            screen_shape=screen_shape,
            net_cfg=config.network,
            iqn_cfg=self.cfg,
        ).to(self.device)
        self.target_net = copy.deepcopy(self.online_net).to(self.device)
        self.target_net.eval()

        initial_lr = self.cfg.lr_schedule[0][1]
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=initial_lr)

        # Mixed precision
        self.use_amp = self.cfg.use_amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Replay buffer
        self.buffer = ReplayBuffer(
            capacity=self.cfg.buffer_size,
            screen_shape=screen_shape,
            float_dim=float_dim,
            n_action_dims=len(ACTION_DIMS),
        )

        # N-step transition buffers (one per env)
        initial_gamma = self.cfg.gamma_schedule[0][1]
        self.n_step_buffers = [
            NStepBuffer(self.cfg.n_step_returns, initial_gamma)
            for _ in range(self.n_envs)
        ]

        # Logging
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self._ep_reward_accum = np.zeros(self.n_envs)
        self._ep_len_accum = np.zeros(self.n_envs, dtype=np.int64)
        self._ep_component_accum = [{} for _ in range(self.n_envs)]
        self._best_race_time = float("inf")

    def _get_schedule_value(self, schedule, step):
        return _piecewise_linear(schedule, step)

    def _update_lr(self, step: int):
        lr = self._get_schedule_value(self.cfg.lr_schedule, step)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr

    def _update_gamma(self, step: int):
        """Update gamma for n-step buffers."""
        gamma = self._get_schedule_value(self.cfg.gamma_schedule, step)
        for buf in self.n_step_buffers:
            buf.gamma = gamma
        return gamma

    @torch.no_grad()
    def _select_actions(self, obs: dict, epsilon: float) -> np.ndarray:
        """Epsilon-greedy action selection over branching heads."""
        actions = np.zeros((self.n_envs, len(ACTION_DIMS)), dtype=np.int64)

        screen = torch.from_numpy(obs["screen"]).to(self.device)
        floats = torch.from_numpy(obs["float"]).to(self.device)

        with torch.amp.autocast("cuda", enabled=self.use_amp):
            Q_branches, _ = self.online_net(screen, floats, self.cfg.iqn_k)

        for d, Q in enumerate(Q_branches):
            Q_mean = Q.float().reshape(self.cfg.iqn_k, self.n_envs, -1).mean(dim=0)
            greedy = Q_mean.argmax(dim=1).cpu().numpy()
            actions[:, d] = greedy

        random_mask = np.random.random(self.n_envs) < epsilon
        for i in range(self.n_envs):
            if random_mask[i]:
                for d, dim_size in enumerate(ACTION_DIMS):
                    actions[i, d] = np.random.randint(dim_size)

        return actions

    def _train_step(self, current_gamma: float):
        """One gradient step with AMP."""
        screens, floats, actions, rewards, next_screens, next_floats, gamma_n_dones = \
            self.buffer.sample(self.cfg.batch_size)

        screens = screens.to(self.device)
        floats = floats.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_screens = next_screens.to(self.device)
        next_floats = next_floats.to(self.device)
        gamma_n_dones = gamma_n_dones.to(self.device)

        batch_size = self.cfg.batch_size
        iqn_n = self.cfg.iqn_n

        self.optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=self.use_amp):
            with torch.no_grad():
                Q_target_branches, _ = self.target_net(next_screens, next_floats, iqn_n)

                if self.cfg.use_ddqn:
                    Q_online_branches, _ = self.online_net(next_screens, next_floats, iqn_n)

                targets_list = []
                for d in range(len(ACTION_DIMS)):
                    Q_t = Q_target_branches[d]

                    if self.cfg.use_ddqn:
                        Q_o = Q_online_branches[d]
                        best_a = Q_o.reshape(iqn_n, batch_size, -1).mean(0).argmax(1, keepdim=True)
                        best_a = best_a.repeat(iqn_n, 1)
                        Q_next = Q_t.gather(1, best_a)
                    else:
                        Q_next = Q_t.max(dim=1, keepdim=True)[0]

                    # N-step Bellman: r + gamma^n * (1-done) * Q_next
                    # gamma_n_dones already encodes gamma^n * (1-done)
                    r = rewards.unsqueeze(-1).repeat(iqn_n, 1)
                    gnd = gamma_n_dones.unsqueeze(-1).repeat(iqn_n, 1)
                    target = r + gnd * Q_next
                    target = target.reshape(iqn_n, batch_size, 1).transpose(0, 1)
                    targets_list.append(target)

            Q_online_branches, tau_online = self.online_net(screens, floats, iqn_n)

            outputs_list = []
            for d in range(len(ACTION_DIMS)):
                Q_o = Q_online_branches[d]
                a_d = actions[:, d].unsqueeze(-1).repeat(iqn_n, 1)
                Q_a = Q_o.gather(1, a_d)
                Q_a = Q_a.reshape(iqn_n, batch_size, 1).transpose(0, 1)
                outputs_list.append(Q_a)

            loss = iqn_loss_branching(
                targets_list, outputs_list, tau_online, iqn_n, batch_size, self.cfg.iqn_kappa
            )
            total_loss = loss.mean()

        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.cfg.max_grad_norm)
        torch.nn.utils.clip_grad_value_(self.online_net.parameters(), self.cfg.max_grad_value)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return total_loss.item()

    def train(self):
        """Main training loop."""
        obs = self.env.reset()
        total_steps = 0
        updates = 0
        last_log_time = time.time()
        losses = deque(maxlen=100)

        print(f"Starting IQN training for {self.cfg.total_timesteps} timesteps...", flush=True)
        print(f"  Run: {self.run_name}")
        print(f"  Environments: {self.n_envs}")
        print(f"  Device: {self.device}")
        print(f"  AMP: {self.use_amp}")
        print(f"  N-step: {self.cfg.n_step_returns}")
        print(f"  Gradient steps/env step: {self.cfg.gradient_steps_per_env_step}", flush=True)

        while total_steps < self.cfg.total_timesteps:
            epsilon = self._get_schedule_value(self.cfg.epsilon_schedule, total_steps)
            current_gamma = self._update_gamma(total_steps)
            actions = self._select_actions(obs, epsilon)

            next_obs, rewards, dones, infos = self.env.step(actions)

            # Store transitions via n-step buffers
            for i in range(self.n_envs):
                self.n_step_buffers[i].add(
                    obs["screen"][i], obs["float"][i], actions[i],
                    rewards[i], dones[i],
                )
                self._ep_reward_accum[i] += rewards[i]
                self._ep_len_accum[i] += 1

                # Accumulate reward components
                info = infos[i]
                for k, v in info.get("reward_components", {}).items():
                    self._ep_component_accum[i][k] = self._ep_component_accum[i].get(k, 0.0) + v

                # Add completed n-step transitions to replay buffer
                if self.n_step_buffers[i].is_ready():
                    s, f, a, r, ns, nf, gnd = self.n_step_buffers[i].pop(
                        next_obs["screen"][i], next_obs["float"][i], dones[i]
                    )
                    self.buffer.add(s, f, a, r, ns, nf, gnd)

                # On episode end, flush remaining partial n-step transitions
                if dones[i]:
                    for transition in self.n_step_buffers[i].flush():
                        self.buffer.add(*transition)

                    self.episode_rewards.append(self._ep_reward_accum[i])
                    self.episode_lengths.append(self._ep_len_accum[i])

                    # Log reward components
                    try:
                        import wandb
                        if wandb.run is not None:
                            comp_log = {f"reward/{k}": v for k, v in self._ep_component_accum[i].items()}
                            if "time_penalty_value" in info:
                                comp_log["reward/time_penalty_value"] = info["time_penalty_value"]
                            wandb.log(comp_log, commit=False)
                    except ImportError:
                        pass

                    self._ep_reward_accum[i] = 0.0
                    self._ep_len_accum[i] = 0
                    self._ep_component_accum[i] = {}

                    # Check for best race time
                    lap = info.get("lap", 0)
                    if lap >= 5:
                        # Use BCD-decoded race_time from fzero_env
                        race_time = info.get("race_time", None)
                        if race_time is None:
                            continue
                        if race_time < self._best_race_time:
                            self._best_race_time = race_time
                            save_path = str(Path(self.dirs["best"]) / "best_model.pt")
                            torch.save(self.online_net.state_dict(), save_path)
                            print(f"New best race time: {race_time:.2f}s -> saved")
                            try:
                                import wandb
                                if wandb.run is not None:
                                    wandb.log({"best/race_time": race_time}, commit=False)
                            except ImportError:
                                pass

            obs = next_obs
            total_steps += self.n_envs

            # Train (multiple gradient steps per env.step)
            if self.buffer.size >= self.cfg.learning_starts:
                current_lr = self._update_lr(total_steps)
                for _ in range(self.cfg.gradient_steps_per_env_step):
                    loss = self._train_step(current_gamma)
                    losses.append(loss)
                    updates += 1

                    if updates % self.cfg.target_update_interval == 0:
                        tau = self.cfg.target_update_tau
                        for tp, op in zip(self.target_net.parameters(), self.online_net.parameters()):
                            tp.data.copy_(tau * op.data + (1.0 - tau) * tp.data)

            # Log periodically
            log_every = self.n_envs * self.config.log_interval
            if total_steps % log_every == 0 and len(self.episode_rewards) > 0:
                now = time.time()
                fps = log_every / max(now - last_log_time, 1e-6)
                last_log_time = now

                mean_rew = np.mean(self.episode_rewards)
                mean_len = np.mean(self.episode_lengths)
                mean_loss = np.mean(losses) if losses else 0
                current_lr = self.optimizer.param_groups[0]["lr"]

                print(f"steps={total_steps:>10d} | eps={epsilon:.3f} | "
                      f"lr={current_lr:.1e} | gamma={current_gamma:.4f} | "
                      f"rew={mean_rew:.1f} | len={mean_len:.0f} | "
                      f"loss={mean_loss:.4f} | fps={fps:.0f}")

                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.log({
                            "rollout/ep_rew_mean": mean_rew,
                            "rollout/ep_len_mean": mean_len,
                            "train/loss": mean_loss,
                            "train/epsilon": epsilon,
                            "train/learning_rate": current_lr,
                            "train/gamma": current_gamma,
                            "time/fps": fps,
                            "time/total_timesteps": total_steps,
                        }, commit=True)
                except ImportError:
                    pass

            # Checkpoint
            if total_steps % self.config.save_freq == 0:
                ckpt_path = Path(self.dirs["checkpoints"]) / f"iqn_{total_steps}.pt"
                torch.save(self.online_net.state_dict(), str(ckpt_path))

        # Final save
        final_path = Path(self.dirs["final"]) / "fzero_iqn_final.pt"
        torch.save(self.online_net.state_dict(), str(final_path))
        print(f"Training complete. Final model saved to {final_path}")
