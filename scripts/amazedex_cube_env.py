from __future__ import annotations

import os

import numpy as np
from gymnasium import spaces

from mujoco_env import MujocoEnv

# Actuator/joint names, in servo-ID order (see motorflash.py, IDS = [1..8]).
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
CAMERA_NAME = "tracking_camera"  # the one real camera, defined in scene.xml

FRAME_SKIP = 1
MAX_STEPS = 500

# Shrink the camera's usable field of view a bit so "cube near the edge of
# frame" already counts as lost, before it's actually clipped out of the
# image. 1.0 = full FOV, lower = more conservative.
FOV_MARGIN = 1


class AmazeDexCubeEnv(MujocoEnv):
    """Rotate a cube in-hand using only what one fixed camera + joint
    encoders can actually give you on the real robot.

    Sim2real notes
    --------------
    - Observation is joint angles + joint velocities + previous action.
      No cube position/orientation is exposed -- the real hand has no pose
      sensor for the cube, so a policy trained on privileged pose info
      would never transfer.
    - "Dropped" is decided by whether the cube is still inside the fixed
      camera's field of view (using the camera's real pose + fovy from the
      model), not by a height threshold. A single camera can't measure
      metric height/depth, but "is the cube still visible in frame" is
      exactly the kind of check a simple color/marker blob detector can
      answer on the real hand.
    - Reward is still allowed to use the cube's true angular velocity from
      the simulator: reward only has to exist during training in sim, the
      policy itself never sees it (only obs matters for sim2real transfer).
    """

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

    def reset_model(self) -> None:
        self.data.qpos[self.joint_qpos_adr] = 0.0
        self.data.qvel[self.joint_dof_adr] = 0.0
        self.prev_action[:] = 0.0
        self.steps = 0

    def _get_obs(self) -> np.ndarray:
        joint_pos = self.data.qpos[self.joint_qpos_adr]
        joint_vel = self.data.qvel[self.joint_dof_adr]
        return np.concatenate([joint_pos, joint_vel, self.prev_action]).astype(np.float32)

    def _cube_visible(self) -> bool:
        """Is the cube inside the fixed camera's frustum?"""
        cam_pos = self.data.cam_xpos[self.camera_id]
        cam_mat = self.data.cam_xmat[self.camera_id].reshape(3, 3)  # columns = camera axes in world frame
        cube_pos = self.data.xpos[self.cube_body_id]

        p_cam = cam_mat.T @ (cube_pos - cam_pos)  # cube position in camera-local frame
        depth = -p_cam[2]  # MuJoCo cameras look down their local -z axis
        if depth <= 1e-6:
            return False  # behind or on top of the camera

        fovy = np.deg2rad(self.model.cam_fovy[self.camera_id])
        half_v = np.tan(fovy / 2.0) * depth * FOV_MARGIN
        # No aspect ratio available here, so treat the horizontal bound the
        # same as the vertical one -- slightly conservative on wide images,
        # which is the safe direction for a "did we lose the cube" check.
        half_h = half_v

        return bool(abs(p_cam[0]) < half_h and abs(p_cam[1]) < half_v)

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        ctrl = self.ctrl_low + (action + 1.0) * 0.5 * (self.ctrl_high - self.ctrl_low)

        full_ctrl = np.zeros(self.model.nu)
        full_ctrl[self.actuator_ids] = ctrl
        self.do_simulation(full_ctrl, self.frame_skip)
        self.steps += 1

        dropped = not self._cube_visible()

        cube_spin = self.data.cvel[self.cube_body_id, 2]  # angular velocity about z (sim-only, reward shaping)
        effort_penalty = 0.01 * float(np.sum(np.square(action)))
        reward = cube_spin - effort_penalty - (10.0 if dropped else 0.0)

        terminated = dropped
        truncated = self.steps >= MAX_STEPS

        self.prev_action = action.copy()

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, {"cube_dropped": dropped}


if __name__ == "__main__":
    # Quick smoke test: random actions for a few episodes.
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
        print(f"episode {episode}: return={total_reward:.2f}, steps={env.steps}, ended: {reason}")
    env.close()