from __future__ import annotations

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np


class MujocoEnv(gym.Env):
    """Loads a MuJoCo model"""

    def __init__(self, model_path: str, frame_skip: int, render_mode: str | None = None):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        self.render_mode = render_mode

        self.init_qpos = self.data.qpos.copy()
        self.init_qvel = self.data.qvel.copy()

        self._viewer = None
        self._renderer = None

    def reset_model(self) -> None:
        """Set self.data.qpos / qvel for a new episode."""
        raise NotImplementedError

    def _get_obs(self) -> np.ndarray:
        """Build the observation from self.data."""
        raise NotImplementedError

    # Gymnasium API 
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.reset_model()
        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        if self.render_mode == "human":
            self.render()
        return obs, {}

    def do_simulation(self, ctrl: np.ndarray, n_frames: int) -> None:
        """Apply a full-length ctrl vector and step physics n_frames times."""
        self.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            mujoco.mj_step(self.model, self.data)

    def render(self):
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                # Increased resolution (1080p) for high-quality evaluation videos
                self._renderer = mujoco.Renderer(self.model, height=1080, width=1080)
            try:
                camera_id = self.model.camera("tracking_camera").id
            except Exception as e:
                print(f"Warning: Failed to set XML camera: {e}")
                camera_id = -1  # falls back to the default free camera
            
            self._renderer.update_scene(self.data, camera=camera_id)
            return self._renderer.render()
            
        elif self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
                try:
                    camera_id = self.model.camera("tracking_camera").id
                    self._viewer.cam.fixedcamid = camera_id
                    self._viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                except Exception as e:
                    print(f"Warning: Failed to set XML camera for human viewer: {e}")
            
            self._viewer.sync()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None