# Research Report: Reward Design for Speed Optimization in Racing RL

**Mode:** Deep
**Date:** 2026-03-30
**Query:** How to create a strong, detectable speed improvement signal for PPO in F-Zero racing

---

## Executive Summary

Our terminal time bonus (79 points at race end) is discounted to 3.15 by gamma=0.999 over 3223 steps — **undetectable by PPO**. This is a fundamental problem with terminal-only rewards in long-horizon discounted RL.

Three viable solutions exist, ordered by practicality:

1. **Per-step speed^2 reward** — eliminates discounting entirely, one-line code change
2. **Undiscounted short horizon (gamma=1.0 + n_steps=140)** — Linesight's actual approach, adapted for PPO
3. **RUDDER reward redistribution** — theoretically elegant but requires deep SB3 surgery

The recommended approach is **Option 1 (per-step speed^2)**, optionally combined with **Option 2 (gamma=1.0)**.

---

## Key Findings

### Finding 1: Linesight's Mini-Race Is NOT About Discounting — It's About Horizon Bounding

From complete source code analysis of [Linesight buffer_utilities.py](https://github.com/Linesight-RL/linesight/blob/main/trackmania_rl/buffer_utilities.py):

Linesight uses `temporal_mini_race_duration_actions = 140` (7 seconds at 20 Hz). At collate time, each transition is randomly assigned a position within a 7-second window. `state_float[0]` is **overwritten** with the mini-race timer. When the timer reaches 140, gamma is forced to 0 (terminal).

**The Q-value is defined as: "total undiscounted reward over the next 7 seconds."** Combined with gamma=1.0 (after schedule), this means:
- Q-value range: [-8.4, +0.3] (tiny, easy to learn)
- No exponential discounting of any reward
- Every step within the window contributes equally

This is possible because Linesight uses **off-policy IQN with a replay buffer**. The random re-contextualization (assigning different mini-race timers to the same transition) is a collate-time operation that is impossible with PPO's on-policy rollout structure.

### Finding 2: Terminal Rewards Are Fundamentally Broken With Discounted PPO

Our experiment confirmed: `gamma^3223 = 0.0398`. A bonus of 79 at race end is worth **3.15** at race start. This is 1.5% of the raw total episode reward (206). After VecNormalize, it's noise.

This is NOT a tuning problem — it's architectural. No amount of scaling the terminal bonus fixes it because VecNormalize adjusts its std proportionally.

### Finding 3: Per-Step Speed^2 Reward Is Used In Practice

From [Deep RL Racing Game (Lopes 2016)](https://lopespm.com/machine_learning/2016/10/06/deep-reinforcement-learning-racing-game.html): "rewards logarithmically based on speed."

From [Stanford Racing RL (Aldape 2018)](https://web.stanford.edu/class/aa228/reports/2018/final150.pdf): reward proportional to squared speed.

From [AWS DeepRacer reward design (Nature 2025)](https://www.nature.com/articles/s41598-025-27702-6): multi-component reward with speed as a primary dense signal.

The common pattern: **speed appears directly in the per-step reward, not as a terminal bonus.** This is dense, undiscounted, and immediately detectable by any RL algorithm.

### Finding 4: RUDDER Exists But Is Impractical For SB3

[RUDDER (Arjona-Medina et al., NeurIPS 2019)](https://ml-jku.github.io/rudder/) decomposes the episode return into per-step contributions using a trained LSTM. This converts sparse terminal rewards into dense per-step signals.

However, RUDDER requires: (a) training a separate sequence model, (b) retroactively modifying rewards after episode completion, (c) recomputing advantages. SB3's PPO consumes and discards rewards immediately during `collect_rollouts()` — there is no "retroactive" hook. Implementation requires forking SB3.

### Finding 5: gamma=1.0 Works With PPO If VecNormalize Is Handled Correctly

From [CleanRL Issue #203](https://github.com/vwxyzjn/cleanrl/issues/203): VecNormalize with gamma=1.0 causes `ret = ret + reward` to grow without bound, breaking normalization statistics. The fix: use a smaller gamma for VecNormalize only (e.g., 0.99) while using gamma=1.0 for PPO's GAE computation.

This means **gamma=1.0 in PPO IS viable** with the VecNormalize gamma decoupled. The terminal bonus would then be fully valued (no discounting), but VecNormalize would still normalize returns at a reasonable scale.

### Finding 6: What Linesight's Reward Actually Looks Like Per Step

From verified source (config.py):
```
per_step = -0.06 + delta_meters * 0.01
```

At 200 km/h: net = -0.032/step (negative!)
At 432 km/h (break-even): net = 0.00/step

The reward is **always negative at normal speeds.** The agent is incentivized purely by "less negative = better." Over a 140-step mini-race with gamma=1.0, the Q-value range is [-8.4, 0] — a bounded, easy-to-learn range where faster driving is unambiguously better (closer to 0).

---

## Analysis & Connections

### Why Our Design Failed

We tried to create speed incentive via a terminal bonus. But with discounted returns over 3000+ steps, the terminal reward is invisible. Linesight avoids this entirely by:

1. Using undiscounted returns (gamma=1.0)
2. Bounding the horizon (7-second mini-races)
3. Having the **per-step** reward directly encode speed information (progress - fixed_penalty)

Their speed signal is in EVERY step, not just the last one. The fixed penalty ensures that every step of slow driving is immediately penalized, and every step of fast driving is immediately rewarded (less negatively).

### The Per-Step Speed^2 Solution

Instead of a terminal bonus, we should add speed information **directly to every step's reward**. The simplest approach: `delta^2 * scale` where `delta` is the per-step track progress (already computed in rewards.py).

This creates superlinear speed incentive:
- At 20 u/step: speed^2 reward = 400 * scale
- At 30 u/step: speed^2 reward = 900 * scale (2.25x more)
- At 32.3 u/step (WR): speed^2 reward = 1043 * scale (2.6x more)

The agent sees this on EVERY step. No discounting. No VecNormalize cancellation (the signal is in the per-step variance, not a terminal spike). Dense, immediate, superlinear.

### Can We Also Use gamma=1.0?

Yes, with a decoupled VecNormalize gamma. gamma=1.0 means the per-step speed^2 reward at step 0 and step 3000 contribute equally to the return. Combined with n_steps=512 in PPO, the value function bootstraps at 512 steps. The bootstrap is undiscounted, so the value function needs to predict "total future reward" — harder to learn than a bounded mini-race, but the per-step speed^2 signal makes this manageable because every step provides immediate feedback.

---

## Practical Implications

### Recommended Implementation

**Replace terminal time bonus with per-step speed^2 reward:**

```python
# In rewards.py compute(), after computing delta:
speed_fwd = max(0.0, delta)
components["speed_bonus"] = (speed_fwd ** 2) * self.cfg.speed_bonus_scale
```

Config:
```python
speed_bonus_scale: float = 0.0001  # tune: makes WR-speed steps worth ~0.1 extra
```

Calibration:
- At 23.5 u/step (current): speed^2 * 0.0001 = 0.055/step
- At 32.3 u/step (WR): speed^2 * 0.0001 = 0.104/step
- Difference: +0.049/step — 2x the per-step time penalty signal

**Keep fixed time penalty at 0.20** — this provides the base "keep driving" pressure.

**Keep PBRS** — this provides cornering guidance.

**Remove or reduce terminal time bonus** — it's invisible to PPO anyway.

**Consider gamma=1.0** with VecNormalize gamma decoupled to 0.99.

---

## Open Questions

- [ ] What is the optimal speed_bonus_scale? Need to calibrate against progress_scale and time_penalty
- [ ] Does gamma=1.0 cause PPO training instability even with decoupled VecNormalize?
- [ ] Should speed^2 bonus replace terminal bonus entirely, or supplement it?
- [ ] Does the quadratic speed reward interfere with PBRS (both reward fast driving)?

---

## References

### Academic Sources
1. [Arjona-Medina et al. "RUDDER: Return Decomposition for Delayed Rewards." NeurIPS 2019.](https://ml-jku.github.io/rudder/)
2. [Huang et al. "The 37 Implementation Details of PPO." ICLR Blog Track 2022.](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
3. [Ng et al. "Policy Invariance Under Reward Transformations." ICML 1999.](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf)
4. ["Reward Design and Hyperparameter Tuning for Generalizable Deep RL Agents in Autonomous Racing." Nature Scientific Reports 2025.](https://www.nature.com/articles/s41598-025-27702-6)

### Industry & Practical Sources
5. [Linesight Trackmania AI — Complete Source Code.](https://github.com/Linesight-RL/linesight)
6. [CleanRL Issue #203 — PPO reward normalization gamma mismatch.](https://github.com/vwxyzjn/cleanrl/issues/203)
7. [Lopes. "Deep Reinforcement Learning: Playing a Racing Game." 2016.](https://lopespm.com/machine_learning/2016/10/06/deep-reinforcement-learning-racing-game.html)

### Codebase Sources
8. Linesight `buffer_utilities.py` — mini-race collate function (verified from raw GitHub)
9. Linesight `buffer_management.py` — reward computation (verified from raw GitHub)
10. Linesight `config.py` — all hyperparameters (verified from raw GitHub)
