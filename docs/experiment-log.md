# F-Zero RL Training Experiment Log

**Project:** Training an RL agent to race F-Zero (SNES, 1991) at competitive speeds on Mute City I.
**Goal:** Approach the world record of 1'57"96 (117.96s) for a 5-lap race.
**Algorithm:** PPO (Stable-Baselines3) with custom dual-input CNN+MLP architecture.

---

## Stage 1: Initial Training (Blue Falcon, Basic Setup)

**Run:** `ppo-80env-bs1024-20260329-071553` and successors
**Car:** Blue Falcon
**Duration:** ~5M steps

### Configuration
- Action space: Discrete(18) — accelerate always held, no brake, no accel toggle
- Screen: 84x84 grayscale, cropped top/bottom 32px
- Frameskip: 4 (15 Hz)
- Reward: progress along track centerline + EMA-adaptive time penalty
- gamma: 0.99
- No track preview, no boost control

### Key Results
- Agent learned to complete 5-lap races
- Best race time: reported as 138s (later found to be ~134s due to BCD timer bug)
- Plateaued — EMA penalty self-cancels speed optimization gradient

### Bugs Found
- **A/Y button swap:** Code mapped A=Brake, Y=Boost, but F-Zero manual says A=Boost(Super Jet), Y=Brake. Agent was braking when it meant to boost.
- **EMA penalty self-cancellation:** Total penalty per race ≈ speed × steps ≈ constant regardless of speed. No gradient for speed optimization.

---

## Stage 2: Architecture Overhaul (MultiDiscrete, Fixed Penalty, Track Preview)

**Run:** `ppo-80env-bs2048-20260329-234655` and `ppo-80env-bs2048-20260330-014729`
**Car:** Blue Falcon
**Duration:** ~5M steps each

### Configuration Changes
- Action space: MultiDiscrete([3,3,2,2,2]) — steer, shoulder, accel, brake, boost all independent
- Screen: 96x84, no crop (full SNES frame)
- Frameskip: 3 (20 Hz, enables blast turning)
- Reward: progress + fixed time penalty (0.20/step) + quadratic time bonus at race completion
- gamma: 0.999
- Track preview: 10 upcoming checkpoints in float observations
- Network: dense_hidden_dim increased from 512 to 1024 (matches Linesight)
- Correct button mapping: A=Boost, B=Accel, Y=Brake
- VecNormalize for reward normalization

### Key Results
- Agent reached 136s (BCD-decoded) best race time
- Learned to boost (43.6% of steps)
- Learned basic blast turning (3% of steps)
- Shoulder lean only used 20% of corners (WR drivers use 100%)

### Bugs Found During This Stage
1. **Timer BCD encoding:** race_timer_sec and race_timer_csec use Binary Coded Decimal, not binary. 0x59 = 59 (not 89). All previously reported race times were wrong.
2. **Lap counter (0x00FC):** Only updates at race end (0→5), never shows 1-4 during the race. Agent had no lap awareness. boost_available feature was always 0.
3. **checkpoint_facing (0x00C5):** Is a facing ANGLE (0-255), not a checkpoint INDEX. Track preview was feeding wrong checkpoints.
4. **rank (0x00D2):** Is a display format flag (0x0F = one digit), not actual race position.
5. **VecNormalize clip_reward=10.0:** Clipped the time bonus to the same value (10.0 normalized) for all race times, destroying the speed differentiation signal.
6. **Partial time bonus bugs:** Estimated lap from cumulative distance could exceed 5, causing projected race times near 0 and bonus spikes to 5000.

### Fixes Applied
- BCD decoding via `_bcd_to_int()` for timer fields
- Cumulative track distance for lap estimation (`_cumulative_dist`)
- Nearest checkpoint by position for track preview (replaces broken checkpoint_facing)
- boost_available derived from estimated_lap instead of broken RAM lap
- Time bonus only on race completion (ram_lap >= 5), no partial bonus
- clip_reward raised to 30.0 to preserve bonus differentiation
- time_bonus_scale = 5000 (quadratic: `((180-t)/180)^2 * 5000`)

---

## Stage 3: Fire Stingray + Fine-tuning (Current)

**Run:** `ppo-80env-bs2048-20260330-0527xx` onward
**Car:** Fire Stingray (fastest car: 478 km/h top speed, best grip, best boost)
**Starting checkpoint:** Best model from Stage 2 (041909 run)

### Configuration
- LR: 1.5e-4 → 1e-5 linear decay (fine-tuning from pre-trained model)
- ent_coef: 0.01
- gamma: 0.999
- time_penalty: 0.20 fixed
- time_bonus_scale: 5000, clip_reward: 30.0
- VecNormalize: norm_reward=True, norm_obs=False
- No PBRS (removed — contributed <4% of reward signal)
- Logging: `reward/norm_episode_return` shows what PPO actually trains on

### Architecture Summary
```
Screen (4, 84, 96) → CNN [4→16→32→64→32] → Flatten → 1792
Float (59,) → MLP [59→128→128] → 128
Concat → 1920 → Linear(1920, 1024) → LeakyReLU → 1024-dim features
→ SB3 policy head (MultiDiscrete [3,3,2,2,2])
→ SB3 value head (scalar)
```

### Reward Structure
```
per_step_reward = progress * 0.01 - 0.20 (fixed penalty)
terminal_bonus = ((180 - race_time) / 180)^2 * 5000  (only on race completion)
```

### What We Learned
- **EMA/adaptive penalty fails** — self-cancels at equilibrium speed
- **Terminal bonus is heavily discounted** by gamma^3000 ≈ 0.05, but still provides directional signal through the value function
- **VecNormalize clip_reward must accommodate the bonus** — too low clips all bonuses to the same value
- **Per-step progress is the primary driver** of speed optimization, not the terminal bonus
- **PBRS is negligible** (<4% of reward) for F-Zero's checkpoint spacing
- **RAM addresses require careful verification** — SnesLab descriptions can be misleading (lap counter, checkpoint_facing, rank)
- **BCD encoding** is common in SNES games — always verify timer formats experimentally

### Comparison to Linesight (Trackmania AI)
| Aspect | Linesight | Ours |
|--------|-----------|------|
| Algorithm | IQN (off-policy) | PPO (on-policy) |
| Action space | Continuous (analog) | MultiDiscrete (digital SNES buttons) |
| Horizon | 7-second mini-race, gamma=1.0 | Full race, gamma=0.999 |
| Reward | progress - fixed penalty (per step) | progress - fixed penalty + terminal bonus |
| Speed signal | Directly from per-step reward in bounded window | Per-step + discounted terminal bonus |
| Data efficiency | High (replay ratio 32x) | Low (each sample used 4x) |
| Wall-clock speed | ~20 fps (heavy 3D game) | ~900 fps (lightweight SNES emulation) |

---

## Key Metrics to Watch

- `best/race_time` — BCD-decoded race time in seconds (lower is better)
- `reward/norm_episode_return` — what PPO actually trains on (higher is better)
- `reward/time_bonus` — raw time bonus per episode (higher = faster race)
- `train/approx_kl` — should be 0.01-0.02 (policy update stability)
- `train/entropy_loss` — exploration level (~2.5 is good for our action space)
- `train/learning_rate` — should decay from 1.5e-4 to 1e-5

## World Record Reference

- **WR:** 1'57"96 (117.96s) — Fire Stingray, Mute City I, 5 laps
- **Key WR techniques:** blast turning (rapid accel toggle), shoulder lean on every corner, optimal boost timing, recharge strip usage
