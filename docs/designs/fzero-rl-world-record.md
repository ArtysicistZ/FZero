# Design Document: F-Zero SNES RL Agent — Beating the World Record

**Author:** Zhou Yincheng
**Date:** 2026-03-28
**Status:** Draft

---

## 1. Problem Statement

F-Zero (SNES, 1991) was labeled an **"unsolved challenge"** for reinforcement learning in the 2016 paper *"Playing SNES in the Retro Learning Environment"* (arxiv.org/abs/1611.02205). The reason: the game provides zero intermediate score during racing — rewards only come at race completion, creating an extreme delayed reward problem that standard RL algorithms (DQN, A3C) cannot solve without modification.

This project solves F-Zero by applying **reward shaping** using direct RAM state reading — the same technique that Linesight used to beat 10/12 Trackmania world records. We target **Mute City I** (the most iconic track, WR: 1'57"96 for 5 laps by "Legend") and aim to train an RL agent that matches or beats the human world record.

The project serves dual purposes: (1) a hands-on learning vehicle for classical RL (the author has GRPO/LLM experience but no traditional RL background), and (2) a resume-worthy project with the narrative *"solved what researchers called an unsolved RL challenge on an iconic 1991 game."*

## 2. Goals

- **Beat the Mute City I world record (1'57"96, 5 laps)** — the marquee goal
- **Build a complete, reproducible RL training pipeline** — from environment integration through training to evaluation, with clean code
- **Solve the "unsolved" delayed reward problem** — prove the 2016 paper wrong via reward shaping using RAM-based game state
- **Produce visual debugging tools** — gameplay videos with debug overlays, training curves with per-component reward breakdown, track position heatmaps
- **Compare multiple RL algorithms** — PPO vs DQN (vs optional IQN stretch goal) on the same environment, demonstrating understanding of algorithmic trade-offs

## 3. Proposed Solution

### 3.1 Overview

We integrate F-Zero into **Stable-Retro** (Farama Foundation's maintained fork of gym-retro) as a custom game, mapping the well-documented RAM addresses (from SnesLab's F-Zero RAM Map) into a Gymnasium environment. The observation space combines **game screenshots** (4-frame stack, 84x84 grayscale, processed by a CNN) with **RAM-extracted features** (speed, energy, lap, checkpoint progress), following the same dual-input architecture that Linesight used to beat Trackmania world records.

The reward function is **dense and shaped** — instead of waiting for the game's end-of-race score, we compute per-frame rewards from checkpoint progress, speed, wall collision penalties, and time penalties. This transforms the "unsolvable" sparse-reward problem into a standard dense-reward RL task.

Training uses **Stable-Baselines3** (PPO primary, DQN secondary) with 8 parallel SNES emulator instances via SubprocVecEnv. The SNES is a 1991 console — emulation is extremely lightweight, and 8 instances consume under 1GB total RAM. The policy network is ~2-3M parameters — modeled after Linesight's proven architecture with a 4-layer CNN, rich float inputs (~50 dims including track preview and action history), and a dueling output head. This comfortably fits on the laptop's RTX 4060. **No GPU cluster is needed.**

## 4. Technical Design

### 4.1 Tech Stack

| Criteria | Chosen: Stable-Retro + SB3 + W&B | Alternative: CleanRL + TensorBoard | Alternative: Custom Wrapper + TorchRL |
|----------|----------------------------------|-------------------------------------|---------------------------------------|
| Emulator integration | Stable-Retro: built-in SNES core (snes9x), RAM access, save states, Gymnasium API | Same emulator, must wire Gymnasium manually | RetroArch/libretro with custom Python bindings — full rebuild |
| RL algorithms | SB3: PPO + DQN built-in, swap in 1 line | Single-file PPO/DQN, transparent but less tooling | TorchRL: powerful but steep learning curve |
| Parallel envs | SB3 SubprocVecEnv — battle-tested | Must implement yourself | TorchRL ParallelEnv, less documented for retro |
| Video recording | SB3 VecVideoRecorder — 1 line | Manual (cv2/ffmpeg) | Manual |
| Logging & viz | W&B: cloud dashboard, video browser, run comparison. Free tier. Great for portfolio sharing. | TensorBoard: local only, clunkier UI | W&B possible but more wiring |
| Learning value | Abstracts details (study CleanRL source alongside for understanding) | Maximum transparency — see every PPO line | Overkill abstraction |
| Community | Largest RL community, most tutorials | Great docs, smaller community | Meta-backed but less beginner content |

**Decision:** Stable-Retro + SB3 + W&B minimizes infrastructure time and maximizes time on the actual interesting work: reward engineering, observation design, algorithm comparison. SB3 provides PPO and DQN with identical APIs, so algorithm comparison is trivial. W&B is chosen over TensorBoard because its cloud dashboard is superior for browsing gameplay videos, comparing runs side-by-side, and sharing results as a portfolio piece.

CleanRL is rejected *as the training framework* but embraced *as a learning companion* — its single-file `ppo_atari.py` (~340 lines) is the best reference for understanding what SB3 does internally. Reading "The 37 Implementation Details of PPO" blog post alongside CleanRL's code is the recommended learning path.

For the IQN stretch goal, SB3 doesn't include IQN, so we'd use **sb3-contrib** (which has QR-DQN, a distributional RL variant in the same family) or implement IQN manually.

### 4.2 File System Structure

```
fzero-rl/
├── env/                              # F-Zero Gymnasium environment
│   ├── __init__.py                   # Exports: make_fzero_env()
│   ├── integration/                  # Stable-Retro game integration files
│   │   ├── data.json                 # RAM address → variable mappings
│   │   ├── scenario.json             # Base reward/done conditions for Stable-Retro
│   │   ├── metadata.json             # Default save state reference
│   │   └── MuteCity1.state           # Save state: race start (right after "GO")
│   ├── fzero_env.py                  # Custom Gymnasium wrapper (~200 lines)
│   │                                 #   - Wraps Stable-Retro env
│   │                                 #   - Constructs dual observation (screenshot + RAM)
│   │                                 #   - Computes shaped reward
│   │                                 #   - Handles episode termination
│   ├── rewards.py                    # Reward components (~80 lines)
│   │                                 #   - progress_reward(), speed_reward()
│   │                                 #   - wall_penalty(), time_penalty()
│   │                                 #   - lap_bonus(), finish_bonus()
│   ├── observations.py               # Observation construction (~100 lines)
│   │                                 #   - frame_to_observation(): crop, grayscale, resize, stack
│   │                                 #   - ram_to_features(): normalize RAM values
│   └── actions.py                    # Action mapping table (~50 lines)
│                                     #   - 18 discrete actions → button combos
│
├── network/                          # Custom neural network architecture
│   ├── __init__.py                   # Exports: FZeroFeatureExtractor
│   └── dual_input.py                 # CNN + RAM dual-input feature extractor (~150 lines)
│                                     #   - CNN branch: 4 conv layers [4→16→32→64→32], LeakyReLU
│                                     #   - Float branch: 2-layer MLP [~50→128→128]
│                                     #   - Fusion: concat → dense → dueling heads (A + V)
│
├── training/                         # Training orchestration
│   ├── __init__.py                   # Exports: train()
│   ├── train.py                      # Main CLI entrypoint (~150 lines)
│   │                                 #   - Parses args (algo, n_envs, timesteps)
│   │                                 #   - Creates env, model, callbacks
│   │                                 #   - Runs training loop
│   ├── config.py                     # Hyperparameter defaults (~80 lines)
│   │                                 #   - PPO: lr, n_steps, batch_size, n_epochs, clip_range
│   │                                 #   - DQN: lr, buffer_size, exploration, target_update
│   │                                 #   - Env: n_envs, frameskip, frame_stack, max_episode_steps
│   └── callbacks.py                  # SB3 callbacks (~120 lines)
│                                     #   - WandbCallback: log reward components
│                                     #   - VideoCallback: periodic gameplay recording
│                                     #   - BestModelCallback: save on best lap time
│
├── evaluation/                       # Evaluation & visualization
│   ├── __init__.py                   # Exports: evaluate(), render_overlay(), plot_heatmap()
│   ├── evaluate.py                   # Run trained model, measure lap times (~80 lines)
│   ├── overlay.py                    # Debug info overlay on game frames (~100 lines)
│   │                                 #   - Speed, checkpoint, energy, reward, action text
│   │                                 #   - Rendered via cv2.putText on captured frames
│   ├── heatmap.py                    # Track position heatmap (~60 lines)
│   │                                 #   - Scatter plot (X, Y) colored by speed or reward
│   │                                 #   - Compare early vs late training
│   └── compare.py                    # Algorithm comparison charts (~80 lines)
│                                     #   - PPO vs DQN learning curves
│                                     #   - Best lap time progression
│
├── tests/                            # Unit & integration tests
│   ├── test_env.py                   # Env stepping, RAM reading, obs shape
│   ├── test_rewards.py               # Reward component edge cases
│   └── test_observations.py          # Frame processing, normalization bounds
│
├── models/                           # Saved checkpoints (gitignored)
├── videos/                           # Recorded gameplay (gitignored)
├── logs/                             # W&B / TensorBoard logs (gitignored)
├── roms/                             # ROM files (gitignored, user-provided)
│
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Ignore models/, videos/, logs/, roms/
└── README.md                         # Project overview, setup, results
```

### 4.3 Encapsulation Design

```
┌──────────────────────────────────────────────────────────┐
│                   train.py (CLI entrypoint)               │
│         Parses args, wires modules, runs training         │
└─────────┬──────────────────┬──────────────────┬──────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────┐ ┌────────────────┐ ┌────────────────────┐
│   env module    │ │ network module │ │ evaluation module   │
│                 │ │                │ │                     │
│ Public API:     │ │ Public API:    │ │ Public API:         │
│ • make_fzero_   │ │ • FZeroCNN    │ │ • evaluate(model)   │
│   env(n_envs,   │ │   (policy     │ │ • render_overlay()  │
│   render_mode)  │ │   network     │ │ • plot_heatmap()    │
│                 │ │   class for   │ │ • compare_algos()   │
│ Internal:       │ │   SB3)       │ │                     │
│ • rewards.py    │ │               │ │ Internal:           │
│ • observations  │ │ Internal:     │ │ • overlay.py        │
│ • actions.py    │ │ • CNN layers  │ │ • heatmap.py        │
│ • integration/  │ │ • RAM MLP     │ │ • compare.py        │
│   (data.json,   │ │ • Fusion      │ │                     │
│    scenario,    │ │   layer       │ │ Dependencies:       │
│    save states) │ │               │ │ • env module        │
│                 │ │ Dependencies: │ │ • opencv-python     │
│ Dependencies:   │ │ • torch       │ │ • matplotlib        │
│ • stable-retro  │ │ • sb3         │ └────────────────────┘
│ • gymnasium     │ └───────────────┘
│ • numpy, cv2    │
└─────────────────┘

Dependency rules:
  • env depends on nothing (except stable-retro/gymnasium)
  • network depends on torch/sb3 only
  • training depends on env + network
  • evaluation depends on env only
  • NO circular dependencies
  • integration/ files are data-only (JSON + binary save states)
```

**Module: `env`**
- Public API: `make_fzero_env(n_envs, render_mode) -> VecEnv`
- Internal: reward computation, frame processing, action table, Stable-Retro integration files
- Dependencies: `stable-retro`, `gymnasium`, `numpy`, `opencv-python`
- NOT exposed: raw RAM addresses, reward scaling constants, frame cropping coordinates

**Module: `network`**
- Public API: `FZeroFeatureExtractor` (a custom SB3-compatible `BaseFeaturesExtractor` subclass)
- Internal: 4-layer CNN (image head), 2-layer MLP (float head), fusion concatenation, dueling output heads (Advantage + Value)
- Dependencies: `torch`, `stable-baselines3`
- NOT exposed: layer sizes, activation functions (configurable via `config.py`)

**Module: `training`**
- Public API: `train(algorithm, config) -> trained_model_path`
- Internal: SB3 model construction, callback setup, W&B initialization
- Dependencies: `env`, `network`, `stable-baselines3`, `wandb`

**Module: `evaluation`**
- Public API: `evaluate(model_path, n_episodes) -> metrics`, `render_overlay(model_path) -> video_path`, `plot_heatmap(trajectory_data) -> figure`, `compare_algos(run_ids) -> figure`
- Internal: OpenCV rendering, matplotlib plotting
- Dependencies: `env`, `opencv-python`, `matplotlib`

### 4.4 Scalability & Parallelism

**Parallel environments (main scaling axis):**

```
┌───────────────────────┐
│   PPO/DQN Agent       │  GPU: RTX 4060 (8GB VRAM)
│   ~2-3M param model   │  Uses ~200MB VRAM
│   (1 process)         │
└───────────┬───────────┘
            │ batch of 8 actions ↓ / batch of 8 observations ↑
            ▼
┌──────────────────────────────────────────────────┐
│              SubprocVecEnv (8 workers)            │
│                                                  │
│  ┌───────┐ ┌───────┐ ┌───────┐     ┌───────┐    │
│  │SNES #1│ │SNES #2│ │SNES #3│ ... │SNES #8│    │
│  │~100MB │ │~100MB │ │~100MB │     │~100MB │    │
│  │1 core │ │1 core │ │1 core │     │1 core │    │
│  └───────┘ └───────┘ └───────┘     └───────┘    │
│  Each: independent process, own emulator state   │
│  Total: ~800MB RAM, 8 CPU cores                  │
└──────────────────────────────────────────────────┘
```

**Resource budget (laptop: 32GB RAM, RTX 4060, ~8 perf cores):**

| Resource | Available | Used by 8 envs | Headroom |
|----------|-----------|----------------|----------|
| RAM | 32 GB | ~1 GB (envs) + ~2 GB (replay buffer, model, OS) | ~29 GB free |
| CPU cores | ~8 perf | 8 (one per env) | Tight — can go to 12 with efficiency cores |
| GPU VRAM | 8 GB | ~200 MB | ~7.8 GB free |
| GPU compute | RTX 4060 | <10% utilization (model is small) | Largely underutilized |

**Throughput estimate:**
- SNES emulation runs at thousands of FPS unthrottled (1991 console on 2026 hardware)
- With frameskip=4, 8 parallel envs: estimated **2,000-5,000 env steps/sec**
- 10M timesteps ≈ **30-80 minutes** of wall-clock training
- **The laptop is more than sufficient. No GPU cluster needed.**

**If scaling to GPU cluster later (for hyperparameter sweeps):**
- More CPU cores → more parallel envs → faster per-experiment
- Main benefit: run 20+ experiments simultaneously (different reward weights, learning rates)
- GPU remains irrelevant — the bottleneck is always CPU (emulator instances)

**Async patterns:**
- Environment stepping is synchronous within SubprocVecEnv (SB3 handles the multiprocessing)
- W&B logging is async (non-blocking uploads)
- Video recording uses a separate evaluation env (doesn't slow training)

**Caching:**
- Track checkpoint geometry: precomputed once at env init, stored as numpy array
- Frame processing pipeline (crop coordinates, resize target): constants, no caching needed
- No model caching needed (loaded once per session)

### 4.5 Detailed Design

#### Observation Space (Dual-Input: Screenshot + Float Features)

This design is modeled after **Linesight** (the Trackmania AI that beat 10/12 world records), which uses a dual-input architecture: a CNN for screenshots and an MLP for rich game-state float features. Linesight feeds 160x120 grayscale screenshots + 164 float dimensions (including 40 upcoming zone centers, 5 previous actions, speed, and contact materials). We adapt this to F-Zero's specifics.

**Visual branch — what the agent sees:**

F-Zero's Mode 7 pseudo-3D view shows the road ahead, walls, turns, and other racers. We crop the HUD (dashboard at bottom, score at top) and feed the road view to a CNN:

```
Raw frame (256x224, SNES native resolution)
       |
  Crop to road view (remove top ~32px sky, bottom ~32px HUD)
       |
  Convert to grayscale (1 channel)
       |
  Resize to 84x84
       |
  Stack 4 consecutive frames (motion perception)
       |
  Result: (4, 84, 84) float32 tensor, values [0, 1]
```

The 4-frame stack is critical: a single frame contains no velocity information. Stacking frames lets the CNN perceive motion — how fast the road is approaching, how the horizon is shifting during turns. This is the standard approach from DeepMind's Atari work.

**Float branch — rich game state (~50 dimensions):**

A key lesson from Linesight: 5 float dimensions is far too thin. Linesight feeds 164 dimensions. The most critical inputs beyond basic state are (1) **upcoming track geometry** — Linesight feeds 40 upcoming zone centers (120 dims), giving the agent a map of the road ahead from game data, and (2) **action history** — the last N actions (20 dims), so the agent understands its own momentum and can commit to multi-step maneuvers.

For F-Zero, we construct ~50 float dimensions:

```python
float_inputs = np.concatenate([
    # === Core state (5 dims) ===
    [speed / MAX_SPEED],              # derived from position delta
    [energy / MAX_ENERGY],            # $7E00C9 — health bar
    [lap / 5.0],                      # $7E00FC — current lap progress
    [float(boost_available)],         # 1.0 if lap >= 2, else 0.0
    [checkpoint_idx / total_cp],      # track progress [0, 1]

    # === Track preview: next 10 checkpoints relative to player (30 dims) ===
    # For each upcoming checkpoint: (delta_x, delta_y, distance), normalized
    # This is the F-Zero equivalent of Linesight's 40 zone centers.
    # Provides a dense map of what's coming — turns, straights, etc.
    next_checkpoints_relative.flatten(),  # 10 x 3 = 30

    # === Action history: last 3 actions, one-hot encoded (15 dims) ===
    # Helps the agent understand momentum from its recent steering decisions.
    # Linesight uses 5 previous actions x 4 bools = 20 dims.
    prev_action_onehot_1,             # 5 dims (simplified encoding)
    prev_action_onehot_2,             # 5 dims
    prev_action_onehot_3,             # 5 dims
], dtype=np.float32)
# Total: ~50 dimensions
```

**Why ~50 dims instead of Linesight's 164?** F-Zero is a simpler game: no gear/RPM, no surface material types, fewer track features. But the original 5-dim proposal was catastrophically insufficient — it lacked the track preview and action history that are essential for racing.

**Why dual-input matters:** The screenshot tells the agent *where to steer* (visual road geometry, walls, curvature). The float features tell the agent *what's coming numerically* (precise track geometry ahead), *what it just did* (action history for momentum), and *how to manage resources* (energy, boost). The CNN and float branch provide complementary information — neither alone is sufficient.

**Float input normalization:** Following Linesight, float inputs are normalized using running mean and standard deviation computed from training data. This prevents features with large magnitudes (e.g., raw pixel positions) from dominating smaller ones (e.g., booleans). The normalization statistics are stored as model parameters.

#### Neural Network Architecture

This architecture is adapted from **Linesight's IQN_Network**, which uses: 4 conv layers [1->16->32->64->32] with LeakyReLU, a 2-layer float MLP [164->256->256], concatenation into a 5888-dim fusion vector, and dueling output heads [5888->512->N_actions / 5888->512->1]. Linesight's model is ~5-8M parameters and beat world records.

We scale down proportionally for F-Zero's simpler visuals (Mode 7 pseudo-3D vs Trackmania's full 3D) while preserving the architectural principles:

```
 Screenshot (4x84x84)                        Float inputs (~50 dims)
       |                                              |
  Conv2d(4, 16, 4x4, stride=2) + LeakyReLU         Linear(50, 128) + LeakyReLU
       |   output: 16x41x41                           |
  Conv2d(16, 32, 4x4, stride=2) + LeakyReLU        Linear(128, 128) + LeakyReLU
       |   output: 32x19x19                           |
  Conv2d(32, 64, 3x3, stride=2) + LeakyReLU         128
       |   output: 64x9x9                             |
  Conv2d(64, 32, 3x3, stride=1) + LeakyReLU          |
       |   output: 32x7x7                             |
  Flatten -> 1568                                     |
       |                                              |
       +---------------- Concat ---------------------+
                            |
                          1696
                            |
                   Linear(1696, 512) + LeakyReLU
                            |
                     +------+------+
                     |             |
               Advantage       Value           <-- Dueling architecture
               head            head
               Lin(512,256)    Lin(512,256)
               LeakyReLU      LeakyReLU
               Lin(256,18)    Lin(256,1)
                     |             |
                     +-- Q = V + A - mean(A) --+

Total parameters: ~2-3M
Memory footprint: ~10 MB
Inference time: <1ms on RTX 4060
```

**Key architectural decisions (learned from Linesight):**

1. **4 conv layers, not 3:** The Nature DQN (2015) used 3 layers with channels [32, 64, 64]. Linesight uses 4 layers [16, 32, 64, 32] — the extra layer provides more spatial feature extraction, and the final 32-channel layer compresses before flattening, reducing the fusion vector size. We follow this pattern.

2. **LeakyReLU, not ReLU:** Prevents dead neurons during training. Linesight uses LeakyReLU throughout. Small change, meaningful improvement in training stability.

3. **2-layer float MLP, not 1-layer:** With ~50 float inputs, a single linear layer is too shallow to extract useful features. Two layers (50->128->128) allow the network to learn nonlinear combinations of track geometry, speed, and action history.

4. **Dueling architecture:** Separates state value V(s) from action advantage A(s,a). In racing, many states have similar value regardless of action (e.g., on a straight, any forward action is fine). Dueling lets the network learn V(s) efficiently without requiring every action to be evaluated. From Wang et al., 2016 — also used by Linesight.

5. **Orthogonal weight initialization:** Following Linesight, all linear and conv layers are initialized with orthogonal initialization scaled by the LeakyReLU gain. This provides better gradient flow in deep networks compared to default Xavier/Kaiming init.

6. **~2-3M params (not 600K, not 8M):** The original 600K was too small — a bottleneck that would cap agent performance. Linesight's ~5-8M is tuned for Trackmania's richer 3D visuals. F-Zero's Mode 7 is simpler, so ~2-3M is the right scale. This is still tiny for an RTX 4060.

This is implemented as a custom SB3 feature extractor by subclassing `BaseFeaturesExtractor`. The dueling heads are implemented in a custom policy class. SB3's PPO and DQN both accept custom feature extractors and policies, so the same architecture works for both algorithms.

#### Action Space

F-Zero's meaningful racing inputs form 18 discrete action combinations. Following Linesight's approach (which defines 12 button combinations including forward, steer, brake, and combined inputs), we enumerate all useful F-Zero input combos:

| Steer (3) | Shoulder (3) | Boost (2) | Description |
|-----------|-------------|-----------|-------------|
| None | None | No | Go straight |
| Left | None | No | Gentle left turn |
| Right | None | No | Gentle right turn |
| Left | L-shoulder | No | Sharp left turn (costs speed) |
| Right | R-shoulder | No | Sharp right turn (costs speed) |
| None | None | Yes | Straight + boost (costs energy) |
| Left | L-shoulder | Yes | Sharp left + boost |
| ... | ... | ... | ... (18 total) |

Accelerate is always held (Linesight also always holds accelerate as a baseline). Brake is included as a separate set of 6 actions (steer x3 + brake, no shoulder) for rare situations where the agent needs to slow down for tight turns, bringing the total consideration to potentially more — but we start with 18 and can expand if needed.

The action space is `Discrete(18)` — compatible with both PPO and DQN. Linesight uses `Discrete(12)` for Trackmania.

#### Reward Function

```python
def compute_reward(prev_state, curr_state, info):
    """
    Dense reward computed every frame from RAM state.
    Each component is logged separately for debugging.
    """
    rewards = {}

    # 1. TRACK PROGRESS (primary signal, ~80% of reward magnitude)
    #    Delta distance along the checkpoint path.
    #    Positive when advancing, negative when going backward.
    progress = compute_track_distance(curr_state) - compute_track_distance(prev_state)
    rewards["progress"] = progress * 10.0

    # 2. SPEED BONUS (encourage maintaining high speed)
    #    Small continuous reward proportional to velocity.
    rewards["speed"] = (curr_state.speed / MAX_SPEED) * 0.1

    # 3. WALL COLLISION PENALTY (energy drops when hitting walls)
    #    Only penalize if energy decreased AND agent is not boosting.
    energy_delta = curr_state.energy - prev_state.energy
    if energy_delta < 0 and not curr_state.boosting:
        rewards["wall"] = energy_delta * 0.5  # negative value
    else:
        rewards["wall"] = 0.0

    # 4. TIME PENALTY (constant per-step cost to encourage finishing fast)
    rewards["time"] = -0.01

    # 5. LAP COMPLETION BONUS
    rewards["lap"] = 100.0 if curr_state.lap > prev_state.lap else 0.0

    # 6. RACE FINISH BONUS (inversely scaled by total time)
    if curr_state.finished:
        rewards["finish"] = max(0, 300.0 - curr_state.race_time_seconds)
    else:
        rewards["finish"] = 0.0

    # Log each component separately to W&B for debugging
    info["reward_components"] = rewards

    total = sum(rewards.values())
    return np.clip(total, -10.0, 200.0)  # prevent reward explosion
```

**How to debug with this:** If training curves show high progress reward but the agent drives slowly, the speed bonus weight is too low. If the agent avoids walls perfectly but takes wide slow turns, the time penalty is too weak. Each component can be tuned independently by observing its contribution in W&B.

#### Episode Structure

```
1. Load MuteCity1.state → race begins right after "GO" countdown
2. Agent receives observation every 4 frames (frameskip=4, ~15 decisions/sec)
3. Each step:
   a. Agent selects action from 18 discrete choices
   b. Emulator advances 4 frames with that button combo held
   c. Wrapper reads RAM: position, energy, lap, checkpoints
   d. Wrapper captures screenshot, processes into 84×84 grayscale
   e. Reward computed from state delta
   f. Observation = (4-frame-stack, RAM features)
4. Episode terminates when:
   a. SUCCESS: Agent finishes 5 laps → large finish bonus
   b. DEATH: Energy reaches 0 → episode ends, no bonus
   c. TIMEOUT: 5 minutes elapsed → episode truncated
   d. STUCK: No checkpoint progress for 500 steps → early termination + small penalty
5. Metrics recorded: total reward, all lap times, best lap, energy remaining, termination reason
6. Reset: reload save state, clear frame stack
```

#### Visualization Pipeline (4 Layers)

**Layer 1: Training curves (W&B, always on)**
- Total episode reward, episode length
- Individual reward components: progress, speed, wall, time, lap, finish
- Best lap time achieved (tracked as W&B summary metric)
- Lap completion rate (% of episodes where agent finishes all 5 laps)

**Layer 2: Periodic gameplay videos (SB3 VecVideoRecorder)**
- Every 50K training steps: record one full evaluation episode as MP4
- Uploaded to W&B media panel for easy browsing
- Shows raw game footage — visual confirmation of agent behavior

**Layer 3: Debug overlay (custom, OpenCV)**
- Rendered on top of game frames during evaluation runs:
```
┌─────────────────────────────────┐
│      [F-Zero game frame]        │
│                                 │
│  SPD: 478 km/h   NRG: ████ 80% │
│  CP:  12/23      LAP: 2/5      │
│  RWD: +0.34      ACT: → + Accel│
│  LAP TIME: 0:23.41             │
│  BEST LAP: 0:22.87             │
└─────────────────────────────────┘
```
- ~100 lines of code using `cv2.putText()` over captured frames
- Saved as MP4 for review

**Layer 4: Track position heatmap (matplotlib)**
- Scatter plot of agent (X, Y) positions across many episodes
- Color by speed: blue=slow, red=fast → reveals braking points
- Color by reward: green=positive, red=negative → reveals problem areas
- Overlay multiple training stages to show learning progression
- Uses RAM-read position data, plotted with matplotlib

### 4.6 Error Handling

| Scenario | Handling |
|----------|----------|
| ROM not found at startup | Clear error: "Place F-Zero (USA).sfc in roms/ directory" with instructions |
| Emulator subprocess crash | SubprocVecEnv auto-restarts; log crash count to W&B; if >5 crashes in 1K steps, halt training |
| RAM returns unexpected values | Assertion checks on first env.reset(): verify position in valid range, energy > 0, lap == 1 |
| NaN in observations | `np.nan_to_num()` + assertion + W&B alert. NaNs propagate silently and ruin training |
| Reward explosion | Clip total reward to [-10, 200] per step; log clipping events to W&B |
| Agent stuck (no progress) | Terminate episode after 500 steps of no checkpoint progress; small penalty (-5.0) |
| Frame processing failure | Fallback to black frame + log warning; don't crash the training run |
| W&B connection lost | Automatic fallback to local TensorBoard logging (SB3 always logs locally) |
| GPU out of memory | Very unlikely — model is ~10MB. If it happens, reduce batch_size in config |

## 5. Alternatives Considered

### 5.1 Pure RAM Observations (No Screenshots)

- **Description:** Use only RAM-extracted features (~50 dimensions: position, speed, checkpoint deltas, energy, track preview, action history) with an MLP policy. No CNN, no frame processing.
- **Pros:** Fastest training (MLP is ~200K params vs dual-input's ~2-3M). Simplest to implement. No frame capture overhead.
- **Cons:** Missing the critical visual look-ahead information. Checkpoint waypoints provide a coarse track preview but don't capture wall positions, track width, or curvature between checkpoints. This is why Linesight uses screenshots — the CNN learns to read the road geometry that sparse waypoints miss. An agent without visual input would need to memorize the entire track through trial-and-error rather than seeing it. Could be used as a quick baseline in early milestones before the full CNN pipeline is ready.

### 5.2 CleanRL Instead of SB3

- **Description:** Use CleanRL's single-file PPO/DQN implementations. Fork and modify directly.
- **Pros:** Maximum transparency — every line of PPO visible. Best for deep learning of the algorithm. No dependency on SB3's abstractions.
- **Cons:** Less tooling: no built-in VecVideoRecorder, no callback system, no model saving/loading utilities. Must reimplement infrastructure that SB3 provides. For a project where the interesting work is reward engineering (not PPO implementation), this is wasted effort. **Mitigated by reading CleanRL's source as a companion** without using it as the framework.

### 5.3 BizHawk Emulator + Lua Scripts

- **Description:** Use BizHawk (the TAS community's SNES emulator) with Lua scripting for memory reading, bridged to Python via sockets.
- **Pros:** Best SNES debugging tools. RAM Watch, RAM Search built-in. Widely used by F-Zero speedrunners for TAS.
- **Cons:** Windows-only, heavy, designed for interactive use. Socket bridge between Lua and Python adds latency. Parallelization is extremely painful (multiple BizHawk windows). Not designed for headless batch execution.

## 7. Performance Considerations

**Training throughput bottleneck analysis:**

| Component | Time per step | Bottleneck? |
|-----------|--------------|-------------|
| SNES emulation (4 frames) | ~0.1ms | No — 1991 hardware on 2026 CPU |
| Frame capture + processing | ~0.2ms | No — resize + grayscale is fast |
| RAM reading | ~0.01ms | No — direct memory access |
| Reward computation | ~0.01ms | No — simple arithmetic |
| Neural network forward pass | ~1ms (batched, GPU) | No — 2-3M params is small |
| PPO update (backprop) | ~5ms per batch | No — moderate model |
| **Subprocess communication** | ~0.3ms | **Yes — IPC is the likely bottleneck** |

The interprocess communication between SubprocVecEnv workers and the main process is typically the bottleneck for lightweight emulators. Mitigation: use `n_steps=256` or higher in PPO config to amortize communication overhead over longer rollouts.

**Estimated wall-clock training times:**

| Timesteps | Estimated time | Expected progress |
|-----------|---------------|-------------------|
| 1M | ~30 min | Agent learns to move forward |
| 5M | ~2.5 hours | Agent completes laps sometimes |
| 10M | ~5 hours | Agent completes laps consistently |
| 50M | ~25 hours | Agent achieves competitive times |
| 100M+ | ~50+ hours | Pushing toward world record |

These estimates assume 8 parallel environments at ~3,000 steps/sec aggregate throughput. Actual speed depends on CPU core count and emulation overhead.

**Memory budget:**

| Component | Memory |
|-----------|--------|
| 8 SNES emulator processes | ~800 MB |
| Frame stack buffers (4×84×84 × 8 envs) | ~9 MB |
| PPO rollout buffer (256 steps × 8 envs) | ~50 MB |
| DQN replay buffer (100K transitions) | ~3 GB (largest single allocation) |
| Policy network (GPU) | ~10 MB |
| OS + Python overhead | ~2 GB |
| **Total** | **~7 GB** (well within 32 GB) |

Note: DQN's replay buffer is the largest memory consumer. If memory becomes tight, reduce `buffer_size` from 100K to 50K.

## 8. Milestones

| Milestone | Scope | Dependencies | Verification Criteria |
|-----------|-------|-------------|----------------------|
| **M1: Environment Integration** | Get F-Zero ROM running in Stable-Retro. Map all RAM addresses into `data.json` (position, energy, lap, checkpoints, speed). Create save state at Mute City I race start. Build basic Gymnasium wrapper that reads RAM and renders frames. | ROM file, Stable-Retro installed | `env.reset()` and `env.step(action)` work. Print RAM values and verify they change correctly. Render a frame and see the game. |
| **M2: Training Pipeline + Visualization** | Implement full observation space (screenshot CNN + RAM features). Implement reward function with all 6 components. Wire PPO training with SB3. Set up W&B logging with per-component reward tracking. Add periodic video recording. | M1 | Agent's total reward increases over 1M steps. All 6 reward components visible in W&B. Gameplay videos show the agent moving (even if poorly). No NaN or crash errors. |
| **M3: Reward Engineering + Debug Tools** | Iterate on reward function weights using debug overlay and track heatmap. Tune reward balance until agent completes laps consistently. Implement stuck detection and early termination. | M2 | Agent finishes 5 laps in >50% of episodes. Debug overlay video shows coherent racing behavior. Heatmap shows agent following the track layout. |
| **M4: Algorithm Comparison + Tuning** | Add DQN as second algorithm. Hyperparameter sweep (learning rate, n_steps, batch_size, clip_range for PPO; buffer_size, exploration_fraction for DQN). Scale to max parallel envs. Generate comparison report. | M3 | Side-by-side PPO vs DQN learning curves in W&B. Best lap time within 50% of world record (~3 minutes). Clear recommendation of which algorithm works better. |
| **M5: Record Chasing + Stretch Goals** | Fine-tune best algorithm with optimized reward function. Advanced techniques: racing line optimization reward, curriculum learning (start slower). Try QR-DQN/IQN from sb3-contrib. Extended training runs (50M+ steps). Polish README with results. | M4 | Best 5-lap time approaches or beats 1'57"96 (Mute City I WR). Clean README with training curves, comparison charts, and gameplay video links. Reproducible training with documented hyperparameters. |

## 9. Open Questions

- [ ] What is the exact RAM address for car speed in F-Zero? (Not explicitly in SnesLab map — may need to derive from position deltas or discover via RAM search in M1)
- [ ] Does the Stable-Retro snes9x core emulate F-Zero's Mode 7 rendering correctly? (Test in M1 — some emulator cores have Mode 7 quirks)
- [ ] Is checkpoint index directly readable from RAM, or must we compute track progress from X/Y position vs. waypoint list? (Explore during M1 integration)
- [ ] What frameskip value optimizes the precision-vs-speed tradeoff? (frameskip=2 gives more precise control but slower training; frameskip=4 is standard but may miss tight turns; test in M2)
- [ ] How does the F-Zero boost mechanic interact with reward shaping? (Boosting costs energy but gains speed — the agent must learn this tradeoff; may need reward tuning in M3)
- [ ] Can we extract the car's heading angle from RAM? (Would improve the RAM observation branch; search in M1)
- [ ] Does the snes9x emulator support running faster than real-time in Stable-Retro? (Critical for training throughput; verify in M1)

---

*Generated with /design*
