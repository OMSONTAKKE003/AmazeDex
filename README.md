# 🤖 AmazeDex

Training a multi-fingered robotic hand to autonomously rotate a cube to any target orientation using deep reinforcement learning, on the **Amazing Hand** by [Pollen Robotics](https://github.com/pollen-robotics/AmazingHand/) (4 fingers, 8 DOF).

<p align="center">
  <img src="docs/images/hand_stand.jpeg" alt="AmazeDex hand" width="550"/>
</p>

Trained fully in **MuJoCo** simulation, deployed to real hardware using AprilTag-based pose estimation. Best result: **~80% success rate** with SAC.

---

## 📂 Repository Map

```
AmazeDex/
│
├── basics/                     # RL fundamentals — toy environments
├── mjcf/                      # MuJoCo assets
│  
├── rockpaperscissors/               # Sim2real mini-task
│  
├── src/                              # Core dextrous manipulation pipeline
```

---

## 🧠 Overview

The robot receives information about the object's current orientation and target orientation and learns finger movements that gradually align the object with the target pose. The training is performed entirely in a simulated environment, where the agent learns through trial and error by maximizing rewards based on orientation accuracy and grasp stability.

This project is a union of mechanical design, computer vision, and reinforcement learning to achieve dexterity — highlighting the challenges of controlling multi-finger robotic systems for complex in-hand manipulation and deploying it on real hardware.

---

## 🖥️ Simulation

The Amazing Hand is a four-fingered hand with 8 degrees of freedom (DOF). It was simulated and trained in **MuJoCo**, chosen for being lightweight, easy to work with, and good at representing real-world physical parameters.

Two XML files define the simulation:
- **`robot.xml`** — the hand's structure: joints, actuators, control ranges, forces, colors, and physical properties
- **`scene.xml`** — the environment: ground, camera, lighting, and the manipulation cube

STL files for the hand were sourced from Onshape and Pollen Robotics' GitHub, then converted to MuJoCo XML using the open-source [`onshape-to-robot`](https://github.com/rhoban/onshape-to-robot) tool. Since no open-source stand existed to hold the hand fixed in position, a custom mechanical stand was designed and 3D printed.

<p align="center">
  <img src="docs/images/hand_sim.png" alt="MuJoCo simulation render" width="450"/>
</p>

---

## 🔧 Hardware

The physical hand is a replica of the original Amazing Hand by Pollen Robotics.

- **Material:** All 3D-printed parts in PLA filament
- **Fasteners:** M2 screws (4–26 mm), M2 hex nuts, 16 mm / 8 mm dowel pins per finger

- **Stand:** Printed in two sections (base + hand-holder), joined with M3 screws

<p align="center">
  <img src="assets/images/hand.png" alt="Assembled hand" width="450"/>
</p>

**Challenges encountered:**

| Issue | Details |
|---|---|
| Finger gimbal snapping | Broke repeatedly despite trying different print settings and filaments |
| Servo adapter failures | Suspected cause: inductive back-EMF surge — when a servo suddenly stops or jams, energy stored in its motor windings discharges as a voltage spike back into the adapter's power rail |
| Slower real-world motion | Servo speeds were intentionally limited to protect the hardware, making real movement slower than in simulation |



## 🎯 Forms of Dextrous Manipulation

Two manipulation strategies were tried:

1. **Lift and rotate** — lift the cube off the palm, then rotate it about its spawn axis.
   ❌ The hand grasped the cube but often stayed idle instead of rotating it (a form of reward hacking), and struggled to lift the cube off the palm.

2. **Rotate to target face** — rotate the cube in place toward a user-specified target face.
   ✅ This gave much better results and became the main approach for the rest of the project.



## 🚀 Usage
 
**1. Install dependencies** (this repo uses [`uv`](https://github.com/astral-sh/uv) for dependency management):
 
```bash
git clone https://github.com/<your-org>/AmazeDex.git
cd AmazeDex
uv sync
```
 
**2. Train the cube-rotation policy in simulation:**
 
```bash
uv run src/training/train_ppo_cube.py
```
 
> The environment is registered via `src/envs/register_amazedex_env.py`; swap in your own RL algorithm/config here to reproduce the PPO / DDPG+HER / SAC comparisons described below.
 
**3. Deploy a trained policy to the real hand:**
 
```bash
uv run src/deployment/simreal.py
```
 
This uses AprilTag-based pose estimation (`src/perception/aruco_pose.py`) to read the cube's real-world orientation and feed it to the policy.
 
**4. (Optional) Try the rock-paper-scissors sim2real mini-task:**
 
```bash
uv run rockpaperscissors/train_ppo_rps.py   # train in sim
uv run rockpaperscissors/sim2real_rps.py    # deploy with webcam gesture input
```
 
**5. (Optional) Explore the RL fundamentals scripts:**
 
```bash
uv run RLPractice/FrozenLake.py
```
 
> Exact script arguments (checkpoint paths, number of episodes, etc.) are defined in each file — check the top of each script for configurable options.
 
---  



## 👥 Contributors

- **Luv Bharat Jain**
- **Om Sontakke**

---

## 🙏 Acknowledgements

Built under the **Society of Robotics and Automation, VJTI**, with guidance from mentors **Arhan Chavre** and **Sahil Apage**.

Hand design based on the open-source **Amazing Hand** by [Pollen Robotics](https://www.pollen-robotics.com/).