from __future__ import annotations

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np


class MujocoEnv(gym.Env):
    """Loads a MuJoCo model """

    def __init__(self, model_path: str, frame_skip: int, render_mode: str | None = None):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        self.render_mode = render_mode

        self.init_qpos = self.data.qpos.copy()
        self.init_qvel = self.data.qvel.copy()

        self._viewer = None
        self._renderer = None
        self._cam_renderer = None  # separate small renderer used for OCR checks

   
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

   # Line 61: The function definition
    def render(self):
        # Line 62: Indented by 4 spaces
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
                
                try:
                    camera_id = self.model.camera("tracking_camera").id
                    with self._viewer.lock():
                        self._viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                        self._viewer.cam.fixedcamid = camera_id
                except Exception as e:
                    print(f"Warning: Failed to set XML camera: {e}")
            
            if self._viewer.is_running():
                self._viewer.sync()
            return None

        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model)
            self._renderer.update_scene(self.data)
            return self._renderer.render()

        return None

    def render_camera_frame(self, camera_id: int, width: int = 240, height: int = 240) -> np.ndarray:
        """Render an RGB frame from a specific camera.

        Independent of render_mode/self.render() -- this needs to work every
        step (e.g. for the OCR-based "is the cube visible" check) even when
        training with render_mode=None.
        """
        if self._cam_renderer is None:
            self._cam_renderer = mujoco.Renderer(self.model, height=height, width=width)
        self._cam_renderer.update_scene(self.data, camera=camera_id)
        return self._cam_renderer.render()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._cam_renderer is not None:
            self._cam_renderer.close()
            self._cam_renderer = None