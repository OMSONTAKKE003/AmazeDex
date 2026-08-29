# AmazeDex

This folder contains all the scripts for dextrous manipulation 

## Repository Structure

```text
src/
├── deployment/
│   └── simreal.py
├── envs/
│   ├── amazedex_cube_env.py
│   ├── mujoco_env.py
│   └── register_amazedex_env.py
├── perception/
│   └── aruco_pose.py
└── training/
    └── train_ppo_cube.py
```

---

## 🏋️ Training

Four reinforcement learning algorithms were tried, in this order:

1. **PPO (Proximal Policy Optimization)** — a common, stable starting point for continuous-control tasks. It worked for basic manipulation, but was slow to train and prone to reward hacking.
2. **PPO + Curriculum Learning** — gradually increased task difficulty. The hand learned to grasp the cube, but this exposed the reward hacking issue more clearly (grasping without rotating), so this approach was dropped.
3. **DDPG + HER (Hindsight Experience Replay)** — HER lets the agent learn from failed attempts by relabeling them with the goal actually achieved. This helped with sparse rewards, but the combination still struggled with the complexity of dexterous contact dynamics.
4. **SAC (Soft Actor-Critic)** — the best performer. SAC is off-policy, reuses past experience through a replay buffer, and uses entropy-driven exploration to try out different finger-coordination strategies instead of settling too early on one behavior.

**Algorithm comparison:**

| Algorithm | Success Rate (Convergence) | Main Issue |
|---|---|---|
| PPO | 0% | No success observed |
| PPO + Curriculum | Low (2–6%) | Grasp-and-idle (reward hacking) |
| DDPG + HER | 0% | Becomes idle after some steps |
| **SAC** | **~80%** | — |

**Key tuning changes that drove the biggest improvement:**
1. Reduced cube size to 48 mm, added a 4 mm edge fillet, reduced mass, and adjusted the initial spawn position
2. Increased push reward: `0.15 → 0.50`
3. Increased reach reward: `0.05 → 0.10`
4. Reweighted the success bonus (`26 → 16`) and drop penalty (`5.1 → 2.0`)
5. Fixed the start face instead of randomizing both start and target faces, cutting the number of combinations the model had to learn from 36 down to a manageable set

<p align="center">
  <img src="docs/images/success_rate_graph.png" alt="Training results graph" width="500"/>
</p>

---