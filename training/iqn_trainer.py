"""
IQN training loop for F-Zero RL.

Self-contained trainer that handles:
  - Replay buffer (uniform sampling)
  - IQN quantile regression loss (Huber pinball)
  - Target network sync
  - Epsilon-greedy exploration with linear decay
  - Integration with VecEnv and W&B logging

Adapted from Linesight's IQN implementation for branching MultiDiscrete.
"""
import copy
import math
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from training.config import TrainingConfig, IQNConfig, NetworkConfig
from network.iqn import IQNNetwork
from env.actions import ACTION_DIMS


class ReplayBuffer:
    """Simple uniform replay buffer for Dict observations."""

    def __init__(self, capacity: int, screen_shape: tuple, float_dim: int, n_action_dims: int):
        self.capacity = capacity
        self.pos = 0
        self.size = 0

        self.screens = np.zeros((capacity, *screen_shape), dtype=np.float32)
        self.floats = np.zeros((capacity, float_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, n_action_dims), dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_screens = np.zeros((capacity, *screen_shape), dtype=np.float32)
        self.next_floats = np.zeros((capacity, float_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def add(self, screen, floats, action, reward, next_screen, next_floats, done):
        self.screens[self.pos] = screen
        self.floats[self.pos] = floats
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_screens[self.pos] = next_screen
        self.next_floats[self.pos] = next_floats
        self.dones[self.pos] = float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.screens[idxs]),
            torch.from_numpy(self.floats[idxs]),
            torch.from_numpy(self.actions[idxs]),
            torch.from_numpy(self.rewards[idxs]),
            torch.from_numpy(self.next_screens[idxs]),
            torch.from_numpy(self.next_floats[idxs]),
            torch.from_numpy(self.dones[idxs]),
        )


def iqn_loss_branching(targets_list, outputs_list, tau_outputs, iqn_n, batch_size, kappa):
    """Compute IQN quantile regression loss for branching architecture.

    Args:
        targets_list: list of 5 tensors, each (batch_size, iqn_n, 1) — target Q per dimension
        outputs_list: list of 5 tensors, each (batch_size, iqn_n, 1) — predicted Q per dimension
        tau_outputs: (batch_size * iqn_n, 1) — quantile fractions
        iqn_n: number of quantiles
        batch_size: batch size
        kappa: Huber loss threshold

    Returns:
        loss: scalar tensor (sum across branches)
    """
    tau = tau_outputs.reshape(iqn_n, batch_size, 1).transpose(0, 1)  # (B, N, 1)
    tau_expanded = tau[:, None, :, :].expand(-1, iqn_n, -1, -1)  # (B, N, N, 1)

    total_loss = torch.zeros(batch_size, device=targets_list[0].device)
    for targets, outputs in zip(targets_list, outputs_list):
        # targets: (B, N, 1), outputs: (B, N, 1)
        td_error = targets[:, :, None, :] - outputs[:, None, :, :]  # (B, N, N, 1)

        # Huber loss
        huber = torch.where(
            td_error.abs() < kappa,
            0.5 / kappa * td_error ** 2,
            td_error.abs() - 0.5 * kappa,
        )
        # Pinball loss with asymmetric weighting
        pinball = torch.where(td_error < 0, 1 - tau_expanded, tau_expanded) * huber
        total_loss = total_loss + pinball.sum(dim=2).mean(dim=1)[:, 0]

    return total_loss


class IQNTrainer:
    """Complete IQN training loop with VecEnv integration."""

    def __init__(self, env, config: TrainingConfig, run_name: str, dirs: dict):
        self.env = env
        self.config = config
        self.cfg = config.iqn
        self.dirs = dirs
        self.run_name = run_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Get observation dimensions from env
        screen_shape = env.observation_space["screen"].shape
        float_dim = env.observation_space["float"].shape[0]
        self.n_envs = env.num_envs

        # Networks
        self.online_net = IQNNetwork(
            float_input_dim=float_dim,
            net_cfg=config.network,
            iqn_cfg=self.cfg,
        ).to(self.device)
        self.target_net = copy.deepcopy(self.online_net).to(self.device)
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(
            self.online_net.parameters(), lr=self.cfg.learning_rate
        )

        # Replay buffer (stores single transitions, not batched)
        self.buffer = ReplayBuffer(
            capacity=self.cfg.buffer_size,
            screen_shape=screen_shape,
            float_dim=float_dim,
            n_action_dims=len(ACTION_DIMS),
        )

        # Logging
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self._ep_reward_accum = np.zeros(self.n_envs)
        self._ep_len_accum = np.zeros(self.n_envs, dtype=np.int64)

    def _get_epsilon(self, step: int) -> float:
        """Linear epsilon decay."""
        frac = min(1.0, step / (self.cfg.exploration_fraction * self.cfg.total_timesteps))
        return self.cfg.exploration_initial_eps + frac * (
            self.cfg.exploration_final_eps - self.cfg.exploration_initial_eps
        )

    @torch.no_grad()
    def _select_actions(self, obs: dict, epsilon: float) -> np.ndarray:
        """Select actions for all envs using epsilon-greedy over branching heads."""
        actions = np.zeros((self.n_envs, len(ACTION_DIMS)), dtype=np.int64)

        # Greedy actions from network
        screen = torch.from_numpy(obs["screen"]).to(self.device)
        floats = torch.from_numpy(obs["float"]).to(self.device)
        Q_branches, _ = self.online_net(screen, floats, self.cfg.iqn_k)

        for d, Q in enumerate(Q_branches):
            # Average across quantiles: (iqn_k * n_envs, dim_d) → (n_envs, dim_d)
            Q_mean = Q.reshape(self.cfg.iqn_k, self.n_envs, -1).mean(dim=0)
            greedy = Q_mean.argmax(dim=1).cpu().numpy()
            actions[:, d] = greedy

        # Epsilon-greedy: random actions for some envs
        random_mask = np.random.random(self.n_envs) < epsilon
        for i in range(self.n_envs):
            if random_mask[i]:
                for d, dim_size in enumerate(ACTION_DIMS):
                    actions[i, d] = np.random.randint(dim_size)

        return actions

    def _train_step(self):
        """Sample batch and perform one gradient step."""
        screens, floats, actions, rewards, next_screens, next_floats, dones = \
            self.buffer.sample(self.cfg.batch_size)

        screens = screens.to(self.device)
        floats = floats.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_screens = next_screens.to(self.device)
        next_floats = next_floats.to(self.device)
        dones = dones.to(self.device)

        batch_size = self.cfg.batch_size
        iqn_n = self.cfg.iqn_n

        with torch.no_grad():
            # Target Q values for next state
            Q_target_branches, tau_target = self.target_net(
                next_screens, next_floats, iqn_n
            )

            if self.cfg.use_ddqn:
                # Use online net to select actions (DDQN)
                Q_online_branches, _ = self.online_net(
                    next_screens, next_floats, iqn_n
                )

            targets_list = []
            for d in range(len(ACTION_DIMS)):
                Q_t = Q_target_branches[d]  # (iqn_n * B, dim_d)

                if self.cfg.use_ddqn:
                    Q_o = Q_online_branches[d]  # (iqn_n * B, dim_d)
                    # Select action from online net (mean over quantiles)
                    best_a = Q_o.reshape(iqn_n, batch_size, -1).mean(0).argmax(1, keepdim=True)
                    best_a = best_a.repeat(iqn_n, 1)  # (iqn_n * B, 1)
                    Q_next = Q_t.gather(1, best_a)  # (iqn_n * B, 1)
                else:
                    Q_next = Q_t.max(dim=1, keepdim=True)[0]  # (iqn_n * B, 1)

                # Expand rewards and dones
                r = rewards.unsqueeze(-1).repeat(iqn_n, 1)  # (iqn_n * B, 1)
                d_mask = dones.unsqueeze(-1).repeat(iqn_n, 1)  # (iqn_n * B, 1)

                # Bellman target (reward is shared across branches, divided by n_branches)
                target = r / len(ACTION_DIMS) + self.cfg.gamma * (1 - d_mask) * Q_next
                target = target.reshape(iqn_n, batch_size, 1).transpose(0, 1)  # (B, N, 1)
                targets_list.append(target)

        # Online Q values for current state
        Q_online_branches, tau_online = self.online_net(
            screens, floats, iqn_n
        )

        outputs_list = []
        for d in range(len(ACTION_DIMS)):
            Q_o = Q_online_branches[d]  # (iqn_n * B, dim_d)
            a_d = actions[:, d].unsqueeze(-1).repeat(iqn_n, 1)  # (iqn_n * B, 1)
            Q_a = Q_o.gather(1, a_d)  # (iqn_n * B, 1)
            Q_a = Q_a.reshape(iqn_n, batch_size, 1).transpose(0, 1)  # (B, N, 1)
            outputs_list.append(Q_a)

        loss = iqn_loss_branching(
            targets_list, outputs_list, tau_online, iqn_n, batch_size, self.cfg.iqn_kappa
        )
        total_loss = loss.mean()

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.cfg.max_grad_norm)
        self.optimizer.step()

        return total_loss.item()

    def train(self, callbacks=None):
        """Main training loop."""
        obs = self.env.reset()
        total_steps = 0
        updates = 0
        last_log_time = time.time()
        losses = deque(maxlen=100)

        # Import callbacks
        from training.callbacks import RewardLoggingCallback, BestLapCallback

        print(f"Starting IQN training for {self.cfg.total_timesteps} timesteps...")
        print(f"  Run: {self.run_name}")
        print(f"  Environments: {self.n_envs}")
        print(f"  Device: {self.device}")

        while total_steps < self.cfg.total_timesteps:
            epsilon = self._get_epsilon(total_steps)
            actions = self._select_actions(obs, epsilon)

            next_obs, rewards, dones, infos = self.env.step(actions)

            # Store transitions
            for i in range(self.n_envs):
                self.buffer.add(
                    obs["screen"][i], obs["float"][i], actions[i],
                    rewards[i], next_obs["screen"][i], next_obs["float"][i],
                    dones[i],
                )
                self._ep_reward_accum[i] += rewards[i]
                self._ep_len_accum[i] += 1

                if dones[i]:
                    self.episode_rewards.append(self._ep_reward_accum[i])
                    self.episode_lengths.append(self._ep_len_accum[i])
                    self._ep_reward_accum[i] = 0.0
                    self._ep_len_accum[i] = 0

                    # Check for best race time
                    info = infos[i] if isinstance(infos, list) else infos
                    lap = info.get("lap", 0) if isinstance(info, dict) else 0
                    if isinstance(info, dict) and lap >= 5:
                        t_min = info.get("race_timer_min", 0)
                        t_sec = info.get("race_timer_sec", 0)
                        t_csec = info.get("race_timer_csec", 0)
                        race_time = t_min * 60 + t_sec + t_csec / 100
                        if not hasattr(self, '_best_race_time') or race_time < self._best_race_time:
                            self._best_race_time = race_time
                            save_path = str(Path(self.dirs["best"]) / "best_model.pt")
                            torch.save(self.online_net.state_dict(), save_path)
                            print(f"New best race time: {race_time:.2f}s -> saved")

            obs = next_obs
            total_steps += self.n_envs

            # Train on batch
            if self.buffer.size >= self.cfg.learning_starts and total_steps % 4 == 0:
                loss = self._train_step()
                losses.append(loss)
                updates += 1

                # Sync target network
                if updates % self.cfg.target_update_interval == 0:
                    self.target_net.load_state_dict(self.online_net.state_dict())

            # Log periodically
            if total_steps % (self.n_envs * 10) == 0 and len(self.episode_rewards) > 0:
                now = time.time()
                fps = (self.n_envs * 10) / max(now - last_log_time, 1e-6)
                last_log_time = now

                mean_rew = np.mean(self.episode_rewards)
                mean_len = np.mean(self.episode_lengths)
                mean_loss = np.mean(losses) if losses else 0

                print(f"steps={total_steps:>10d} | eps={epsilon:.3f} | "
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
                            "time/fps": fps,
                            "time/total_timesteps": total_steps,
                        }, commit=True)
                except ImportError:
                    pass

            # Save checkpoint periodically
            if total_steps % self.config.save_freq == 0:
                ckpt_path = Path(self.dirs["checkpoints"]) / f"iqn_{total_steps}.pt"
                torch.save(self.online_net.state_dict(), str(ckpt_path))

        # Save final model
        final_path = Path(self.dirs["final"]) / "fzero_iqn_final.pt"
        torch.save(self.online_net.state_dict(), str(final_path))
        print(f"Training complete. Final model saved to {final_path}")
