# Research Report: RL Agent for Game World Record

**Mode:** Deep
**Date:** 2026-03-28
**Query:** Finding the best lightweight, locally-runnable game to train an RL agent toward world-record performance, as a learning project for someone with GRPO/LLM background but new to traditional RL.

---

## Executive Summary

Training an RL agent to beat or approach a game world record is a proven, resume-worthy project with several successful precedents. The most famous recent example is **Linesight** (Trackmania Nations Forever), which beat 10 out of 12 official campaign world records using IQN on a single consumer GPU. Other notable projects include the **QWOP RL agent** (40 hours of training to set a new world record), **Pokemon Red RL** (beat the entire game with a <10M param policy), and **Yosh's Trackmania AI** (beat the legendary A01 record of 23.79s with 23.64s).

The best game candidates fall into three tiers: (1) **well-trodden but proven** (Trackmania, Atari), (2) **partially explored with room to push further** (Geometry Dash, Game Boy games via PyBoy, SuperTuxKart), and (3) **largely unexplored and novel** (Getting Over It, Celeste, Jump King, niche retro games via stable-retro). For a resume project, the sweet spot is either replicating Linesight on Trackmania and pushing for new records on untouched tracks, or tackling a game with minimal prior RL work where even partial success is novel.

Your GRPO background transfers well: PPO is essentially the same trust-region optimization idea applied to game actions instead of tokens. The bottleneck in game RL is CPU cores (running parallel environments), not GPU VRAM. A standard desktop with 8+ CPU cores and a mid-range GPU is sufficient.

---

## Key Findings

### 1. The Trackmania Ecosystem is the Gold Standard

The video you watched is by **Yosh (@yoshtm)**, who spent 3+ years training AI on Trackmania Nations Forever. His agent achieved **23.64s on A01** (the game's most iconic track), beating both the human world record (23.79s) and even tool-assisted speedruns. He used a progression of algorithms (NEAT -> DQN -> SAC) and invented "training wheels" rewards for specific driving techniques.

The open-source **[Linesight](https://github.com/Linesight-RL/linesight)** project is the most mature Trackmania RL codebase:
- **Game:** Trackmania Nations Forever (completely free)
- **Algorithm:** IQN (Implicit Quantile Networks)
- **Interface:** TMInterface 2.1.0 (allows programmatic control + game state reading)
- **Training:** ~80 hours at 9x game speed on a single GPU
- **Results:** Beat 10/12 official campaign world records (May 2024)
- **Hardware:** 20GB RAM, NVIDIA GPU with CUDA, Windows or Linux
- **Docs:** [linesight-rl.github.io](https://linesight-rl.github.io/linesight/build/html/)

There's also **[TMRL](https://github.com/trackmania-rl/tmrl)** for Trackmania 2020 (pip-installable, uses SAC/REDQ), but it hasn't matched Linesight's results yet.

**Verdict:** Proven and accessible, but heavily explored. To stand out, you'd need to beat records on tracks where no AI has been trained yet.

### 2. QWOP - The "40 Hours to World Record" Story

[Wesley Liao's QWOP-RL project](https://github.com/Wesleyliao/QWOP-RL) is perhaps the most impressive "small project, big result" RL achievement:
- **Training time:** ~40 hours total (25h self-play + 15h imitation from a top human player)
- **Algorithm:** ACER -> DDQN with prioritized replay
- **Result:** Set a new QWOP world record, top-10 speedrun time
- **Game:** QWOP (browser game, simple state space)
- [Detailed blog post on Medium](https://medium.com/data-science/achieving-human-level-performance-in-qwop-using-reinforcement-learning-and-imitation-learning-81b0a9bbac96)

**Verdict:** Already done, but the approach (combining RL with imitation learning from human experts) is a great template.

### 3. Retro/Emulated Games via PyBoy and Stable-Retro

**[PyBoy](https://github.com/Baekalfen/PyBoy)** (Game Boy emulator in Python):
- Runs headlessly, supports frame skipping for 3x+ speed
- Direct memory access for extracting game state (HP, position, score)
- Multiple instances can run in parallel
- Notable projects:
  - **[PokemonRedExperiments](https://github.com/PWhiddy/PokemonRedExperiments):** Beat Pokemon Red with <10M param policy using SB3
  - **[PyBoy-RL](https://github.com/lixado/PyBoy-RL):** Trained Super Mario Land (26h on 4-core CPU) and Kirby's Dream Land (20h on GPU)

**[Stable-Retro](https://stable-retro.farama.org/)** (Farama Foundation):
- Wraps SNES, Genesis, NES, Atari, N64, and more into Gymnasium environments
- ~1000 game integrations with memory variable mappings
- You supply the ROM, it provides reward functions and state access
- Supports: Sega Master System/Genesis/CD/32X, NES, SNES, N64, DS, Atari 2600, Arcade

**Verdict:** Massive catalog of games, many with NO prior RL work. Pick a Game Boy or SNES game with a speedrun community and you're likely doing something novel.

### 4. Geometry Dash - Partially Explored

Multiple GitHub projects exist ([geodashml](https://github.com/hakanonal/geodashml), [OpenGD-RL](https://github.com/bdang10/OpenGD-RL), [geometry-dash-ai](https://github.com/ThePickleGawd/geometry-dash-ai)), using DQN, PPO, and genetic algorithms. However:
- Most projects are incomplete or train on only the simplest levels
- No project has approached world-record performance on hard levels
- The game has a simple binary action space (jump / don't jump) which is ideal for RL
- Challenge: The actual game requires purchase + memory hooking; alternatives use open-source clones

**Verdict:** Room to push much further. Beating hard custom levels would be novel and visually impressive.

### 5. Getting Over It - Barely Explored

Only [one GitHub repo](https://github.com/nuxlear/GettingOverIt) attempts RL on Getting Over It:
- Continuous action space (mouse angle + force), which is harder for RL
- Physics-based, with catastrophic failure states (falling back to start)
- The speedrun world record is well-defined (~1 minute)
- Very few serious RL attempts

**Verdict:** High novelty, high difficulty. If you succeed even partially, it would be very impressive and likely go viral (the game is culturally iconic). But the continuous control + sparse rewards make this a hard first project.

### 6. Celeste - Partially Explored but Unsolved

Several attempts exist ([celesteRL](https://github.com/boesingerl/celesteRL), [Celeste-NEAT](https://github.com/hdrien0/Celeste-NEAT)):
- celesteRL uses rllib + Gymnasium with an Everest mod but **hasn't been able to finish even level 1**
- NEAT approach can clear simple rooms in ~10 generations but doesn't scale
- The game has a rich speedrun community with very precise records

**Verdict:** High novelty potential. Nobody has successfully trained RL to speedrun Celeste. But the complexity (dashes, wall jumps, momentum) makes it challenging.

### 7. SuperTuxKart - Open Source Racing Alternative

[SuperTuxKart](https://github.com/topics/supertuxkart) is fully open-source with a Python API ([pystk2-gymnasium](https://gymnasium.farama.org/environments/third_party_environments/)):
- Free, open source, pip-installable gym wrapper
- Multiple projects exist but none have achieved world-record times
- 3D racing game similar in spirit to Mario Kart
- [Decision Transformer approach](https://github.com/vibalcam/deep-rl-supertux-race) exists but more can be done

**Verdict:** Good alternative to Trackmania if you want a fully open-source stack. Less competitive scene though.

### 8. Jump King - Partially Explored

[Code-Bullet's Jump-King](https://github.com/Code-Bullet/Jump-King) went viral on YouTube. A [few academic projects](https://github.com/DemetriCassidy/Jump-King-AI-Project) exist using actor-critic methods but only partially play the game. The real game requires purchase + memory reading.

**Verdict:** Moderate novelty. A full completion would be impressive but the game's "one mistake loses everything" design is brutal for RL.

---

## Analysis & Connections

### The "Novelty vs. Feasibility" Tradeoff

| Game | Novelty | Feasibility | Impressiveness | Recommended? |
|------|---------|-------------|----------------|--------------|
| **Trackmania (Linesight)** | Low (well-explored) | Very High (docs, code, free game) | High | Yes, as learning vehicle |
| **Retro game via PyBoy/Stable-Retro** | High (pick unexplored game) | High (emulator handles parallelism) | Medium-High | **Best for novelty** |
| **Getting Over It** | Very High | Low (continuous control, sparse reward) | Very High if successful | Risky but viral potential |
| **Celeste** | High (unsolved by RL) | Medium (mod needed, complex mechanics) | Very High | Good stretch goal |
| **Geometry Dash** | Medium (some work exists) | High (simple action space) | Medium-High | Good middle ground |
| **SuperTuxKart** | Medium | High (open source, gym wrapper) | Medium | Safe choice |
| **QWOP** | Low (already done) | High | Medium | Good for learning only |

### What Makes an RL Game Project Resume-Worthy

1. **Clear metric:** A world record time/score that your agent approaches or beats
2. **Visual demo:** A video of the agent playing is worth 1000 words
3. **Novel contribution:** Either a new game, new algorithm application, or new record
4. **Technical depth:** Reward shaping, parallel training, algorithm comparison
5. **Reproducibility:** Open-source code with clear documentation

### Your GRPO Background is an Advantage

The mental model transfers:
- PPO ≈ GRPO's trust-region optimization, but for game actions instead of tokens
- Reward model → game score/state
- Environment rollouts → game simulation steps
- The key difference: game RL models are tiny (<10M params), and the bottleneck is CPU (running environments), not GPU

---

## Practical Implications: Recommended Approach

### Option A: "Stand on Giants' Shoulders" (Safest, Best for Learning)

1. **Start with Linesight on Trackmania Nations Forever** (free game)
2. Reproduce their results on a known track
3. Train on a track where no AI has achieved a record
4. Compare IQN vs PPO vs SAC on the same track
5. Document everything, publish code + video

**Time estimate:** 2-4 weeks to reproduce, 2-4 weeks to push further

### Option B: "Unexplored Territory" (Most Resume Impact)

1. **Pick a retro game via PyBoy or Stable-Retro** that has speedrun records but no RL work
   - Good candidates: Mega Man (NES), Kirby games, Castlevania, Metroid
   - Check speedrun.com for the game's records, verify no RL projects exist
2. Build a Gymnasium wrapper with memory-based state extraction
3. Train with PPO (Stable-Baselines3) with 8-16 parallel environments
4. Compare against human speedrun records

**Time estimate:** 1-2 weeks for environment, 2-4 weeks for training

### Option C: "Viral Potential" (Highest Risk/Reward)

1. **Tackle Getting Over It or Celeste** — games that are culturally iconic and unsolved by RL
2. Getting Over It has continuous mouse control (harder) but simpler state
3. Celeste has discrete actions but complex mechanics
4. Even partial success (e.g., beating 3 Celeste levels) would be impressive

**Time estimate:** 4-8+ weeks, with risk of failure

### Recommended Tech Stack

- **Framework:** Stable-Baselines3 (easiest start) + CleanRL (for understanding)
- **Algorithm:** PPO (most versatile) or SAC (for continuous control)
- **Parallel training:** SubprocVecEnv with 8-16 environments
- **Logging:** Weights & Biases or TensorBoard
- **Game interface:** TMInterface (Trackmania) or PyBoy (Game Boy) or Stable-Retro (multi-platform)
- **Hardware:** 8+ CPU cores, any modern GPU (4-8GB VRAM sufficient)

---

## Open Questions & Future Directions

- Which specific retro games have active speedrun communities but zero RL attempts? A systematic search of speedrun.com categories could identify the best targets.
- Could model-based RL (like DreamerV3) outperform model-free approaches on these games with less compute?
- Is there value in combining RL with LLM-based reasoning for strategy games (leveraging your GRPO experience)?
- How do you handle games where the "world record" uses glitches — should the RL agent be allowed to discover glitches?

---

## References

### Open-Source RL Game Projects
1. [Linesight — Trackmania RL AI](https://github.com/Linesight-RL/linesight)
2. [TMRL — Trackmania 2020 RL Framework](https://github.com/trackmania-rl/tmrl)
3. [QWOP-RL — World Record via RL](https://github.com/Wesleyliao/QWOP-RL)
4. [PokemonRedExperiments — Beat Pokemon Red with RL](https://github.com/PWhiddy/PokemonRedExperiments)
5. [PyBoy-RL — Mario/Kirby via Game Boy Emulator](https://github.com/lixado/PyBoy-RL)
6. [OpenGD-RL — Geometry Dash RL](https://github.com/bdang10/OpenGD-RL)
7. [celesteRL — Celeste RL (unsolved)](https://github.com/boesingerl/celesteRL)
8. [Getting Over It RL](https://github.com/nuxlear/GettingOverIt)
9. [DreamerV3 — Minecraft Diamond](https://github.com/danijar/dreamerv3)

### Frameworks & Tools
10. [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
11. [CleanRL](https://github.com/vwxyzjn/cleanrl)
12. [Gymnasium](https://gymnasium.farama.org/)
13. [Stable-Retro (Farama)](https://stable-retro.farama.org/)
14. [PyBoy — Python Game Boy Emulator](https://github.com/Baekalfen/PyBoy)
15. [EnvPool — High-Performance Env Engine](https://github.com/sail-sg/envpool)

### Key Blog Posts & Tutorials
16. [QWOP RL Blog Post (Medium)](https://medium.com/data-science/achieving-human-level-performance-in-qwop-using-reinforcement-learning-and-imitation-learning-81b0a9bbac96)
17. [37 Implementation Details of PPO](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
18. [Linesight Documentation](https://linesight-rl.github.io/linesight/build/html/)
19. [Gymnasium: Create Custom Environment](https://gymnasium.farama.org/introduction/create_custom_env/)
20. [RL Tips & Tricks — SB3 Docs](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html)

### Cautionary Reading
21. [Deep RL Doesn't Work Yet — Alex Irpan](https://www.alexirpan.com/2018/02/14/rl-hard.html)
22. [Reward Hacking in RL — Lil'Log](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/)
23. [Speedrun.com Forum: ML & Games](https://www.speedrun.com/forums/speedrunning/n5url)
