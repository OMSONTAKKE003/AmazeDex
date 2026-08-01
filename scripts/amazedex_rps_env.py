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

FRAME_SKIP = 10
MAX_STEPS = 200

# Target gesture configurations (-1.0 to 1.0 action space)
PAPER = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
ROCK = (1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0)
SCISSORS = (0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 1.0, -1.0)

GESTURE_NAMES = ("rock", "paper", "scissors")
GESTURE_TARGETS = np.array([ROCK, PAPER, SCISSORS], dtype=np.float32)

SUCCESS_DIST_THRESHOLD = 0.25 
DENSE_REWARD_SCALE = 1.0
SPARSE_REWARD_CORRECT = 10.0
SPARSE_REWARD_INCORRECT = -5.0
RANDOM_START_PROB = 0.75

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

        # Fetch joint control limits directly from XML model
        self.actuator_low = self.model.actuator_ctrlrange[self.actuator_ids, 0]
        self.actuator_high = self.model.actuator_ctrlrange[self.actuator_ids, 1]

        self.dense_reward_scale = dense_reward_scale

        self.action_space = spaces.Box(-1.0, 1.0, shape=(8,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(27,), dtype=np.float32)

        self.prev_action = np.zeros(8, dtype=np.float32)
        self.steps = 0
        self.target_gesture_idx = 0

    def _action_to_ctrl(self, action: np.ndarray) -> np.ndarray:
        """Map [-1, 1] policy action to actual XML actuator ranges."""
        return self.actuator_low + (action + 1.0) * 0.5 * (self.actuator_high - self.actuator_low)

    def reset_model(self) -> None:
        # Reset physics data back to default XML state
        mujoco.mj_resetData(self.model, self.data)

        if self.np_random.random() < RANDOM_START_PROB:
            # Sample random starting joint positions within actuator bounds.
            # NOTE: each joint is sampled independently, so some draws will
            # produce geometrically self-colliding finger configurations.
            # mj_forward alone only computes kinematics/contact detection, it
            # does NOT resolve those contacts -- if left as-is, the first
            # mj_step() of the actual episode would violently "explode" the
            # interpenetrating geometry apart in a single timestep. Instead,
            # we hold this random pose via ctrl and let the physics solver
            # settle any overlap gradually here, during reset, before the
            # episode -- and the policy -- ever sees it.
            random_ctrl = self.np_random.uniform(self.actuator_low, self.actuator_high, size=8)
            self.data.qpos[self.joint_qpos_adr] = random_ctrl

            mujoco.mj_forward(self.model, self.data)

            # Hold the sampled pose with ctrl and step physics forward a
            # short number of times so any self-collision gets resolved
            # smoothly (small per-step corrective forces) rather than all
            # at once on the first real policy step.
            self.data.ctrl[self.actuator_ids] = random_ctrl
            SETTLE_STEPS = 20
            for _ in range(SETTLE_STEPS):
                mujoco.mj_step(self.model, self.data)

            # Settling can impart residual velocity from resolving contacts;
            # zero it out so the episode starts from rest, not mid-collision-response.
            self.data.qvel[:] = 0.0

        # Forward kinematics to update body/joint positions (no-op if the
        # random branch above already ran mj_forward + settled, but needed
        # for the non-random branch to reflect the default XML qpos).
        mujoco.mj_forward(self.model, self.data)

        self.prev_action[:] = 0.0
        self.steps = 0
        self.target_gesture_idx = int(self.np_random.integers(len(GESTURE_NAMES)))

    def _get_obs(self) -> np.ndarray:
        joint_pos = self.data.qpos[self.joint_qpos_adr]
        joint_vel = self.data.qvel[self.joint_dof_adr] * 0.1  # Velocity scaling for RL stability

        onehot = np.zeros(len(GESTURE_NAMES), dtype=np.float32)
        onehot[self.target_gesture_idx] = 1.0

        obs = np.concatenate([joint_pos, joint_vel, self.prev_action, onehot]).astype(np.float32)
        return np.clip(np.nan_to_num(obs), -10.0, 10.0)

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        ctrl = self._action_to_ctrl(action)

        # Apply controls to MuJoCo simulation
        self.data.ctrl[self.actuator_ids] = ctrl
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.steps += 1

        joint_pos = self.data.qpos[self.joint_qpos_adr]
        target_ctrl = self._action_to_ctrl(GESTURE_TARGETS[self.target_gesture_idx])
        
        dist = float(np.linalg.norm(joint_pos - target_ctrl))

        all_target_ctrls = np.array([self._action_to_ctrl(g) for g in GESTURE_TARGETS])
        predicted_idx = int(np.argmin(np.linalg.norm(all_target_ctrls - joint_pos[None, :], axis=1)))
        correct = predicted_idx == self.target_gesture_idx

        terminated = dist < SUCCESS_DIST_THRESHOLD
        truncated = self.steps >= MAX_STEPS

        effort_penalty = 0.05 * float(np.sum(np.square(action - self.prev_action)))
        dense_reward = self.dense_reward_scale * (-dist) - effort_penalty

        sparse_reward = 0.0
        if terminated or truncated:
            sparse_reward = SPARSE_REWARD_CORRECT if correct else SPARSE_REWARD_INCORRECT

        self.prev_action = action.copy()

        if self.render_mode == "human":
            self.render()

        info = {
            "target_gesture_name": GESTURE_NAMES[self.target_gesture_idx],
            "predicted_gesture_name": GESTURE_NAMES[predicted_idx],
            "correct_gesture": correct,
            "dist_to_target": dist,
        }

        return self._get_obs(), dense_reward + sparse_reward, terminated, truncated, info

if __name__ == "__main__":
    env = AmazeDexRockPaperScissorsEnv()
    for episode in range(2):
        obs, info = env.reset(seed=episode)
        total_reward = 0.0
        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        print(f"Ep {episode}: return={total_reward:.2f}, steps={env.steps}, correct={info['correct_gesture']}")
    env.close()