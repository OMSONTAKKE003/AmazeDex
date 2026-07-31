from __future__ import annotations
import os
import numpy as np
from gymnasium import spaces
from mujoco_env import MujocoEnv
import mujoco

ACTUATOR_NAMES = [
    "motor_finger1_1", "motor_finger1_2",
    "motor_finger2_1", "motor_finger2_2",
    "motor_finger3_1", "motor_finger3_2",
    "motor_finger4_1", "motor_finger4_2",
]

JOINT_NAMES = [
    "finger1_motor1", "finger1_motor2",
    "finger2_motor1", "finger2_motor2",
    "finger3_motor1", "finger3_motor2",
    "finger4_motor1", "finger4_motor2",
]

# Matched to sim2real's 50Hz control loop (10 * 0.002s = 0.02s)
FRAME_SKIP = 10
MAX_STEPS = 200

PAPER = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
ROCK = (1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0)
SCISSORS = (0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 1.0, -1.0)

GESTURE_NAMES = ("rock", "paper", "scissors")
GESTURE_TARGETS = np.array([ROCK, PAPER, SCISSORS], dtype=np.float32)

SUCCESS_DIST_THRESHOLD = 0.25 # Slightly relaxed to allow steady-state convergence 

DENSE_REWARD_SCALE = 1.0
SPARSE_REWARD_CORRECT = 10.0
SPARSE_REWARD_INCORRECT = -5.0

PD_KP = 20.0
PD_ARMATURE = 0.005
PD_DAMPRATIO = 2.0
PD_KV = PD_DAMPRATIO * 2.0 * np.sqrt(PD_KP * PD_ARMATURE)
PD_TORQUE_LIMIT = 1.57

RANDOM_START_PROB = 0.75
START_POSE_NOISE = 0.25

class AmazeDexRockPaperScissorsEnv(MujocoEnv):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        model_path: str = os.path.join("resources", "rock.xml"),
        render_mode: str | None = None,
        dense_reward_scale: float = DENSE_REWARD_SCALE,
    ):
        super().__init__(model_path, FRAME_SKIP, render_mode)

        self.actuator_ids = np.array([self.model.actuator(n).id for n in ACTUATOR_NAMES])
        self.joint_qpos_adr = np.array([self.model.joint(n).qposadr[0] for n in JOINT_NAMES])
        self.joint_dof_adr = np.array([self.model.joint(n).dofadr[0] for n in JOINT_NAMES])

        # IGNORING XML actuator_ctrlrange. Enforcing position limits strictly to match sim2real.
        # This prevents PPO from demanding 180-degree hyper-extensions.
        self.ctrl_low = np.full(8, -1.39, dtype=np.float32)
        self.ctrl_high = np.full(8, 1.39, dtype=np.float32)

        self.dense_reward_scale = dense_reward_scale

        self.action_space = spaces.Box(-1.0, 1.0, shape=(8,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(27,), dtype=np.float32)

        self.prev_action = np.zeros(8, dtype=np.float32)
        self.steps = 0
        self.target_gesture_idx = 0

    def reset_model(self) -> None:
        # 1. Reset completely to 0 to wipe any constraint violations
        mujoco.mj_resetData(self.model, self.data)

        if self.np_random.random() < RANDOM_START_PROB:
            # Sample uniformly across the full ctrl range rather than biasing near
            # a gesture target. This avoids any one gesture (e.g. paper, which sits
            # at the all-zero pose) getting an artificial head start toward its own
            # target just because of how the start pose was sampled.
            start_pose = self.np_random.uniform(self.ctrl_low, self.ctrl_high, size=8).astype(np.float32)

            # 2. Safely drive the closed-loop kinematics to the random start pose
            # without tearing the linkage apart. 
            for _ in range(100):
                joint_pos = self.data.qpos[self.joint_qpos_adr]
                joint_vel = self.data.qvel[self.joint_dof_adr]
                torque = PD_KP * (start_pose - joint_pos) - PD_KV * joint_vel
                torque = np.clip(torque, -PD_TORQUE_LIMIT, PD_TORQUE_LIMIT)
                
                full_ctrl = np.zeros(self.model.nu)
                full_ctrl[self.actuator_ids] = torque
                self.data.ctrl[:] = full_ctrl
                mujoco.mj_step(self.model, self.data)
                
            self.data.qvel[:] = 0.0
            self.data.qacc[:] = 0.0
            mujoco.mj_forward(self.model, self.data)

        self.prev_action[:] = 0.0
        self.steps = 0
        self.target_gesture_idx = int(self.np_random.integers(len(GESTURE_NAMES)))

    def _target_onehot(self) -> np.ndarray:
        onehot = np.zeros(len(GESTURE_NAMES), dtype=np.float32)
        onehot[self.target_gesture_idx] = 1.0
        return onehot

    def _get_obs(self) -> np.ndarray:
        joint_pos = self.data.qpos[self.joint_qpos_adr]
        joint_vel = self.data.qvel[self.joint_dof_adr]
        obs = np.concatenate(
            [joint_pos, joint_vel, self.prev_action, self._target_onehot()]
        ).astype(np.float32)
        
        # Sanitize obs to prevent PPO from diverging if a stray collision happens
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(obs, -100.0, 100.0)

    def _dist_to_target(self, joint_pos: np.ndarray) -> float:
        target = GESTURE_TARGETS[self.target_gesture_idx]
        return float(np.linalg.norm(joint_pos - target))

    def _predicted_gesture(self, joint_pos: np.ndarray) -> int:
        dists = np.linalg.norm(GESTURE_TARGETS - joint_pos[None, :], axis=1)
        return int(np.argmin(dists))

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        target_angle = self.ctrl_low + (action + 1.0) * 0.5 * (self.ctrl_high - self.ctrl_low)

        # PD control MUST evaluate natively at physics frequency (500Hz) inside the skip loop.
        for _ in range(self.frame_skip):
            joint_pos = self.data.qpos[self.joint_qpos_adr]
            joint_vel = self.data.qvel[self.joint_dof_adr]
            torque = PD_KP * (target_angle - joint_pos) - PD_KV * joint_vel
            torque = np.clip(torque, -PD_TORQUE_LIMIT, PD_TORQUE_LIMIT)
            
            full_ctrl = np.zeros(self.model.nu)
            full_ctrl[self.actuator_ids] = torque
            self.data.ctrl[:] = full_ctrl
            mujoco.mj_step(self.model, self.data)
            
        self.steps += 1
        
        # Guard against physics explosion
        if not np.all(np.isfinite(self.data.qpos)) or not np.all(np.isfinite(self.data.qvel)):
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            obs[-len(GESTURE_NAMES):] = self._target_onehot()
            info = {
                "error": "physics_diverged",
                "target_gesture_idx": self.target_gesture_idx,
                "target_gesture_name": GESTURE_NAMES[self.target_gesture_idx],
                "predicted_gesture_idx": 0,
                "predicted_gesture_name": "unknown",
                "correct_gesture": False,
                "dist_to_target": 10.0,
                "dense_reward": -10.0,
                "sparse_reward": -50.0,
            }
            return obs, -50.0, True, False, info

        joint_pos = self.data.qpos[self.joint_qpos_adr]
        dist = self._dist_to_target(joint_pos)
        predicted_idx = self._predicted_gesture(joint_pos)
        correct = predicted_idx == self.target_gesture_idx

        success = dist < SUCCESS_DIST_THRESHOLD
        terminated = success
        truncated = self.steps >= MAX_STEPS

        # Penalize DELTA action to prevent jitter, making it completely free to HOLD a gesture
        effort_penalty = 0.05 * float(np.sum(np.square(action - self.prev_action)))
        dense_reward = self.dense_reward_scale * (-dist) - effort_penalty

        sparse_reward = 0.0
        if terminated or truncated:
            sparse_reward = SPARSE_REWARD_CORRECT if correct else SPARSE_REWARD_INCORRECT

        reward = dense_reward + sparse_reward

        self.prev_action = action.copy()

        if self.render_mode == "human":
            self.render()

        info = {
            "target_gesture_idx": self.target_gesture_idx,
            "target_gesture_name": GESTURE_NAMES[self.target_gesture_idx],
            "predicted_gesture_idx": predicted_idx,
            "predicted_gesture_name": GESTURE_NAMES[predicted_idx],
            "correct_gesture": correct,
            "dist_to_target": dist,
            "dense_reward": dense_reward,
            "sparse_reward": sparse_reward,
        }
        return self._get_obs(), reward, terminated, truncated, info

if __name__ == "__main__":
    env = AmazeDexRockPaperScissorsEnv()
    for episode in range(5):
        obs, info = env.reset(seed=episode)
        total_reward = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        print(
            f"episode {episode}: return={total_reward:.2f}, steps={env.steps}, "
            f"target={info['target_gesture_name']}, "
            f"predicted={info['predicted_gesture_name']}, "
            f"correct={info['correct_gesture']}"
        )
    env.close()