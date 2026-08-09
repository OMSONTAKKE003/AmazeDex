from __future__ import annotations

import os

import mujoco
import numpy as np
from gymnasium import spaces

from mujoco_env import MujocoEnv


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
CUBE_BODY_NAME = "cube"

FRAME_SKIP = 1
MAX_STEPS = 800

FALL_HEIGHT_THRESHOLD = 0.05

ROT_AXIS = np.array([0.0, 0.0, 1.0])
REF_VECTOR = np.array([1.0, 0.0, 0.0])
LOCAL_UP_AXIS = np.array([0.0, 0.0, 1.0])

MAX_STEP_ANGLE = np.pi / 20  # Avoid large rotation per step

NOISE_DEADZONE_DEG = 0.05  # Avoid noise for twist angle

SWING_LIMIT_DEG = 10.0  # Allow up to 10 degrees tilt without penalty

# --- REWARD & PENALTY SCALES ---
ROTATION_REWARD_SCALE = 20.0
VELOCITY_PENALTY_SCALE = 1
FALL_PENALTY = 15.0
ACTION_SMOOTHNESS_SCALE = 0.1
SWING_PENALTY_SCALE = 0.05  # Linear scale (capped at -1.0 max per step)

SUCCESS_THRESHOLD =  2 * np.pi  # Success when 1 revolutions are completed


class AmazeDexCubeEnv(MujocoEnv):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}

    def __init__(
        self,
        model_path: str = os.path.join("resources", "scene.xml"),
        render_mode: str | None = None,
    ):
        super().__init__(model_path, FRAME_SKIP, render_mode)

        self.actuator_ids = np.array([self.model.actuator(n).id for n in ACTUATOR_NAMES])
        self.joint_qpos_adr = np.array([self.model.joint(n).qposadr[0] for n in JOINT_NAMES])
        self.joint_dof_adr = np.array([self.model.joint(n).dofadr[0] for n in JOINT_NAMES])
        self.cube_body_id = self.model.body(CUBE_BODY_NAME).id

        ctrl_range = self.model.actuator_ctrlrange[self.actuator_ids]
        self.ctrl_low, self.ctrl_high = ctrl_range[:, 0], ctrl_range[:, 1]

        self.action_space = spaces.Box(-1.0, 1.0, shape=(8,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(24,), dtype=np.float32)

        self.prev_action = np.zeros(8, dtype=np.float32)
        self.steps = 0

        self._rot_dir = ROT_AXIS / np.linalg.norm(ROT_AXIS)
        self._ref_vec = REF_VECTOR / np.linalg.norm(REF_VECTOR)
        self._local_up = LOCAL_UP_AXIS / np.linalg.norm(LOCAL_UP_AXIS)

        self.prev_unit_vector: np.ndarray | None = None
        self.cum_twist = 0.0

        self.cube_init_height: float | None = None

    def _rotate_local(self, local_vec: np.ndarray, quat: np.ndarray) -> np.ndarray:
        out = np.zeros(3)
        mujoco.mju_rotVecQuat(out, local_vec, quat)
        return out

    def _project_perp(self, world_vec: np.ndarray) -> np.ndarray | None:
        v = world_vec - np.dot(world_vec, self._rot_dir) * self._rot_dir
        n = np.linalg.norm(v)
        return v / n if n > 1e-8 else None

    def _twist_and_swing(self):
        cube_quat = self.data.xquat[self.cube_body_id]
        world_ref = self._rotate_local(self._ref_vec, cube_quat)
        world_up = self._rotate_local(self._local_up, cube_quat)

        new_unit_vector = self._project_perp(world_ref)
        alignment = float(np.dot(world_up, self._rot_dir))

        if self.prev_unit_vector is None:
            self.prev_unit_vector = new_unit_vector
            return 0.0, alignment, new_unit_vector

        if new_unit_vector is None:
            return 0.0, alignment, self.prev_unit_vector

        cos_a = np.clip(np.dot(new_unit_vector, self.prev_unit_vector), -1.0, 1.0)
        cross = np.cross(self.prev_unit_vector, new_unit_vector)
        sin_a = np.dot(cross, self._rot_dir)
        angle = np.arctan2(sin_a, cos_a)
        angle = np.clip(angle, -MAX_STEP_ANGLE, MAX_STEP_ANGLE)
        if abs(angle) < np.deg2rad(NOISE_DEADZONE_DEG):
            angle = 0.0

        self.prev_unit_vector = new_unit_vector
        return float(angle), alignment, new_unit_vector

    def reset_model(self) -> None:
        self.data.qpos[self.joint_qpos_adr] = 0.0
        self.data.qvel[self.joint_dof_adr] = 0.0
        self.prev_action[:] = 0.0
        self.steps = 0

        self.prev_unit_vector = None
        self.cum_twist = 0.0
        self.cube_init_height = None

    def _get_obs(self) -> np.ndarray:
        joint_pos = self.data.qpos[self.joint_qpos_adr]
        joint_vel = self.data.qvel[self.joint_dof_adr]
        return np.concatenate([joint_pos, joint_vel, self.prev_action]).astype(np.float32)

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        ctrl = self.ctrl_low + (action + 1.0) * 0.5 * (self.ctrl_high - self.ctrl_low)

        full_ctrl = np.zeros(self.model.nu)
        full_ctrl[self.actuator_ids] = ctrl
        self.do_simulation(full_ctrl, self.frame_skip)
        self.steps += 1

        # ---- REWARD & METRICS COMPUTATION ----
        angle, alignment, _ = self._twist_and_swing()
        self.cum_twist += angle
        success = bool(self.cum_twist > SUCCESS_THRESHOLD)

        cube_z = float(self.data.xpos[self.cube_body_id][2])
        if self.cube_init_height is None:
            self.cube_init_height = cube_z

        # Check if the cube has dropped
        fallen = bool(cube_z < (self.cube_init_height - FALL_HEIGHT_THRESHOLD))

        # Calculate swing tilt
        swing_deg = float(np.degrees(np.arccos(np.clip(alignment, -1.0, 1.0))))
        swing_excess = max(0.0, swing_deg - SWING_LIMIT_DEG)

        # Reward terms
        rotation_reward = ROTATION_REWARD_SCALE * angle

        obj_linear_vel = float(np.linalg.norm(self.data.cvel[self.cube_body_id, 3:6]))
        velocity_penalty = -VELOCITY_PENALTY_SCALE * obj_linear_vel

        fall_penalty = -FALL_PENALTY * float(fallen)

        action_delta = action - self.prev_action
        smoothness_penalty = -ACTION_SMOOTHNESS_SCALE * float(np.sum(np.square(action_delta)))

        # Linear, capped swing penalty (Max penalty capped at -1.0 per step)
        swing_penalty = -min(1.0, SWING_PENALTY_SCALE * swing_excess)

        reward = (
            rotation_reward
            + velocity_penalty
            + fall_penalty
            + smoothness_penalty
            + swing_penalty
        )

        # Episode terminations
        terminated = fallen
        truncated = self.steps >= MAX_STEPS

        self.prev_action = action.copy()

        if self.render_mode == "human":
            self.render()

        info = {
            "fallen": fallen,
            "cube_z": cube_z,
            "twist_angle": angle,
            "cum_twist": self.cum_twist,
            "alignment": alignment,
            "swing_deg": swing_deg,
            "success": success,
            "rotation_reward": rotation_reward,
            "velocity_penalty": velocity_penalty,
            "fall_penalty": fall_penalty,
            "smoothness_penalty": smoothness_penalty,
            "swing_penalty": swing_penalty,
        }

        return self._get_obs(), reward, terminated, truncated, info


if __name__ == "__main__":
    env = AmazeDexCubeEnv()
    for episode in range(5):
        obs, info = env.reset(seed=episode)
        total_reward = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        reason = "cube fell" if info["fallen"] else "max steps reached"
        print(
            f"episode {episode}: return={total_reward:.2f}, steps={env.steps}, "
            f"cum_twist={np.degrees(env.cum_twist):.1f} deg, ended: {reason}"
        )
    env.close()