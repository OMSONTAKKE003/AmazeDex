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
START_POSE_NOISE = 0.25
OBS_NOISE_STD = 0.005  # Simulates real-world encoder noise

class AmazeDexRockPaperScissorsEnv(MujocoEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        # FIX 1: Added '=' and missing comma ',' after default argument
        model_path: str = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "mjcf", "scene.xml")
        ),
        render_mode: str | None = None,
        dense_reward_scale: float = DENSE_REWARD_SCALE,
    ):
        super().__init__(model_path, FRAME_SKIP, render_mode)

        self.actuator_ids = np.array([self.model.actuator(n).id for n in ACTUATOR_NAMES])
        self.joint_qpos_adr = np.array([self.model.joint(n).qposadr[0] for n in JOINT_NAMES])
        self.joint_dof_adr = np.array([self.model.joint(n).dofadr[0] for n in JOINT_NAMES])

        self.base_friction = self.model.dof_frictionloss.copy()
        self.base_damping = self.model.dof_damping.copy()

        self.ctrl_low = np.full(8, -1.3962634, dtype=np.float32)
        self.ctrl_high = np.full(8, 1.3962634, dtype=np.float32)

        self.dense_reward_scale = dense_reward_scale

        # Observation space replaces 8 velocity inputs with 8 prev_pos inputs
        self.action_space = spaces.Box(-1.0, 1.0, shape=(8,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(27,), dtype=np.float32)

        self.prev_action = np.zeros(8, dtype=np.float32)
        self.prev_joint_pos = np.zeros(8, dtype=np.float32) 
        self.steps = 0
        self.target_gesture_idx = 0

    def _randomize_domain(self):
        """Randomize physics to prevent policy overfitting (Sim-to-Real best practice)."""
        friction_noise = self.np_random.uniform(0.85, 1.15, size=self.model.nv)
        damping_noise = self.np_random.uniform(0.85, 1.15, size=self.model.nv)
        self.model.dof_frictionloss[:] = self.base_friction * friction_noise
        self.model.dof_damping[:] = self.base_damping * damping_noise

    def reset_model(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self._randomize_domain()

        if self.np_random.random() < RANDOM_START_PROB:
            start_idx = int(self.np_random.integers(len(GESTURE_NAMES)))
            start_pose = GESTURE_TARGETS[start_idx] + self.np_random.uniform(
                -START_POSE_NOISE, START_POSE_NOISE, size=8
            )
            start_pose = np.clip(start_pose, self.ctrl_low, self.ctrl_high)
            
            self.data.ctrl[self.actuator_ids] = start_pose
            
            for _ in range(100):
                mujoco.mj_step(self.model, self.data)
        else:
            for _ in range(100):
                mujoco.mj_step(self.model, self.data)

        self.prev_action[:] = 0.0
        self.prev_joint_pos = self.data.qpos[self.joint_qpos_adr].copy()
        self.steps = 0
        self.target_gesture_idx = int(self.np_random.integers(len(GESTURE_NAMES)))

    def _target_onehot(self) -> np.ndarray:
        onehot = np.zeros(len(GESTURE_NAMES), dtype=np.float32)
        onehot[self.target_gesture_idx] = 1.0
        return onehot

    def _get_obs(self) -> np.ndarray:
        joint_pos = self.data.qpos[self.joint_qpos_adr].copy()
        
        # Add encoder noise during training
        if self.render_mode is None: 
            joint_pos += self.np_random.normal(0, OBS_NOISE_STD, size=8)

        obs = np.concatenate(
            [joint_pos, self.prev_joint_pos, self.prev_action, self._target_onehot()]
        ).astype(np.float32)
        
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
        
        smoothed_action = 0.8 * action + 0.2 * self.prev_action
        target_angle = self.ctrl_low + (smoothed_action + 1.0) * 0.5 * (self.ctrl_high - self.ctrl_low)

        self.data.ctrl[self.actuator_ids] = target_angle
        
        # FIX 2: mujoco.mj_step takes (model, data), it does NOT take nstep
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
            
        self.steps += 1
        
        # Keep divergence check active just in case
        if not np.all(np.isfinite(self.data.qpos)):
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            obs[-len(GESTURE_NAMES):] = self._target_onehot()
            return obs, -50.0, True, False, {"error": "physics_diverged", "correct_gesture": False}

        joint_pos = self.data.qpos[self.joint_qpos_adr]
        dist = self._dist_to_target(joint_pos)
        predicted_idx = self._predicted_gesture(joint_pos)
        correct = predicted_idx == self.target_gesture_idx

        success = dist < SUCCESS_DIST_THRESHOLD
        terminated = success
        truncated = self.steps >= MAX_STEPS

        effort_penalty = 0.05 * float(np.sum(np.square(action - self.prev_action)))
        dense_reward = self.dense_reward_scale * (-dist) - effort_penalty

        sparse_reward = 0.0
        if terminated or truncated:
            sparse_reward = SPARSE_REWARD_CORRECT if correct else SPARSE_REWARD_INCORRECT

        reward = dense_reward + sparse_reward

        self.prev_joint_pos = joint_pos.copy()
        self.prev_action = action.copy()

        if self.render_mode == "human":
            self.render()

        info = {
            "target_gesture_idx": self.target_gesture_idx,
            "target_gesture_name": GESTURE_NAMES[self.target_gesture_idx],
            "correct_gesture": correct,
            "predicted_gesture_name": GESTURE_NAMES[predicted_idx],
            "dist_to_target": dist,
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
            # FIX 3: Added 'predicted_gesture_name' to info dictionary output
            predicted_name = GESTURE_NAMES[env._predicted_gesture(env.data.qpos[env.joint_qpos_adr])]
            print(
                f"episode {episode}: return={total_reward:.2f}, steps={env.steps}, "
                f"target={info['target_gesture_name']}, "
                f"predicted={predicted_name}, "
                f"correct={info['correct_gesture']}"
            )
    env.close()