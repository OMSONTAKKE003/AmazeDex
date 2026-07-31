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
CAMERA_NAME = "tracking_camera" 

FRAME_SKIP = 1
MAX_STEPS = 500

FACE_NORMALS_BODY = np.array([
    [1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, -1.0],
])

FACE_NUMBERS = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5", 5: "6"}

MIN_FACE_CONFIDENCE = 0.5

DROP_HEIGHT_DROP_M = 0.05      # cube sank this far below its start height
MAX_CUBE_CAMERA_DIST_M = 0.35  # cube moved this far away from the camera


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
        self.camera_id = self.model.camera(CAMERA_NAME).id

        ctrl_range = self.model.actuator_ctrlrange[self.actuator_ids]
        self.ctrl_low, self.ctrl_high = ctrl_range[:, 0], ctrl_range[:, 1]

        self.action_space = spaces.Box(-1.0, 1.0, shape=(8,), dtype=np.float32)
        # joint_pos(8) + joint_vel(8) + prev_action(8) -- everything the
        # real robot can actually know.
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(24,), dtype=np.float32)

        self.prev_action = np.zeros(8, dtype=np.float32)
        self.steps = 0
        self.initial_cube_z = None

    def reset_model(self) -> None:
        self.data.qpos[self.joint_qpos_adr] = 0.0
        self.data.qvel[self.joint_dof_adr] = 0.0
        self.prev_action[:] = 0.0
        self.steps = 0
        self.initial_cube_z = None

    def _get_obs(self) -> np.ndarray:
        joint_pos = self.data.qpos[self.joint_qpos_adr]
        joint_vel = self.data.qvel[self.joint_dof_adr]
        return np.concatenate([joint_pos, joint_vel, self.prev_action]).astype(np.float32)

    def _get_cube_face_toward_camera(self) -> tuple[int, str, float]:
     
        cube_pos = self.data.xpos[self.cube_body_id]
        cube_quat = self.data.xquat[self.cube_body_id]  # (w, x, y, z)

        rot_flat = np.zeros(9)
        mujoco.mju_quat2Mat(rot_flat, cube_quat)
        rot = rot_flat.reshape(3, 3)

        cam_pos = self.data.cam_xpos[self.camera_id]
        view_dir = cam_pos - cube_pos
        view_dir = view_dir / (np.linalg.norm(view_dir) + 1e-8)

        world_normals = FACE_NORMALS_BODY @ rot.T 
        dots = world_normals @ view_dir

        face_idx = int(np.argmax(dots))
        return face_idx, FACE_NUMBERS[face_idx], float(dots[face_idx])

    def _cube_dropped(self) -> bool:
    
        if self.initial_cube_z is None:
            self.initial_cube_z = float(self.data.xpos[self.cube_body_id][2])

        cube_pos = self.data.xpos[self.cube_body_id]
        cam_pos = self.data.cam_xpos[self.camera_id]

        fell = (self.initial_cube_z - cube_pos[2]) > DROP_HEIGHT_DROP_M
        too_far = float(np.linalg.norm(cube_pos - cam_pos)) > MAX_CUBE_CAMERA_DIST_M
        _, _, face_conf = self._get_cube_face_toward_camera()
        unreadable = face_conf < MIN_FACE_CONFIDENCE

        return bool(fell or too_far or unreadable)

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        ctrl = self.ctrl_low + (action + 1.0) * 0.5 * (self.ctrl_high - self.ctrl_low)

        full_ctrl = np.zeros(self.model.nu)
        full_ctrl[self.actuator_ids] = ctrl
        self.do_simulation(full_ctrl, self.frame_skip)
        self.steps += 1

        dropped = self._cube_dropped()
        face_idx, face_number, face_conf = self._get_cube_face_toward_camera()

        cube_spin = self.data.cvel[self.cube_body_id, 2]  # angular velocity about z (sim-only, reward shaping)
        effort_penalty = 0.01 * float(np.sum(np.square(action)))
        reward = cube_spin - effort_penalty - (10.0 if dropped else 0.0)

        terminated = dropped
        truncated = self.steps >= MAX_STEPS

        self.prev_action = action.copy()

        if self.render_mode == "human":
            self.render()

        info = {
            "cube_dropped": dropped,
            "face_up_index": face_idx,
            "face_up_number": face_number,
            "face_confidence": face_conf,
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
        reason = "cube dropped/lost from frame" if terminated else "max steps reached"
        print(
            f"episode {episode}: return={total_reward:.2f}, steps={env.steps}, "
            f"ended: {reason}, last face up: {info['face_up_number']} "
            f"(conf={info['face_confidence']:.2f})"
        )
    env.close()