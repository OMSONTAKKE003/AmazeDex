"""
amazedex_cube_env.py

Gymnasium-style environment: rotate a numbered cube (faces 1-6) in-hand using
the Pollen Robotics "Amazing Hand" (AmazeDex1), simulated in MuJoCo.

Follows the Gymnasium custom-env API:
https://gymnasium.farama.org/introduction/create_custom_env

Files expected next to this script (produced by the onshape-to-robot export):
    my-robot/
        scene.xml               <- includes robot.xml, spawns the cube + floor + camera
        robot.xml                <- hand kinematics, actuators, equality constraints
        joints_properties.xml    <- actuator "feel" defaults (referenced by robot.xml)
        additional.xml
        assets/                  <- STL meshes

Hardware mapping (matches motorflash.py, which drives servo IDs 1-8):
    Servo ID -> MuJoCo actuator (declaration order in robot.xml <actuator>):
        1 -> motor_finger1_1   5 -> motor_finger3_1
        2 -> motor_finger1_2   6 -> motor_finger3_2
        3 -> motor_finger2_1   7 -> motor_finger4_1
        4 -> motor_finger2_2   8 -> motor_finger4_2
    action[i] always drives the servo with ID (i + 1).

WHAT CHANGED VS THE OLD rusty.py
---------------------------------
1. "Runs for 3 episodes then crashes" -> two real bugs, both fixed:
     a) The old __main__ block had a hardcoded `for episode in range(3)`.
        That's just a smoke test, not a training loop -- it was never meant
        to run more than 3 episodes. This file makes the episode count a
        parameter instead of a magic number.
     b) scene.xml declared `<size njmax="500" nconmax="200"/>`. This hand has
        ~30 bodies with several ball/hinge joints each, so once a grasp gets
        messy the contact/constraint count can exceed those limits. Hitting
        that limit is a *fatal* MuJoCo error that kills the whole process --
        not a catchable Python exception in older bindings. Fixed by raising
        the buffer sizes in scene.xml (see the accompanying updated file).
        This env also adds a belt-and-suspenders runtime guard (see
        `_physics_is_finite` and the try/except around `mj_step`) so a bad
        step ends *one* episode instead of the whole run.
2. Visual observations. The real hand has no cube-pose sensor -- the only
   way to know where the cube is / which face is up is a camera, which is
   why the cube has numbers 1-6 printed on it. So the observation the policy
   receives is now a Dict:
       {
         "image":       (H, W, 3) uint8 RGB frame from the FIXED
                         "tracking_camera" defined in scene.xml,
         "joint_pos":   (8,) float32 -- finger joint angles (rad),
         "joint_vel":   (8,) float32 -- finger joint velocities (rad/s),
         "prev_action": (8,) float32 -- last commanded action,
       }
   The cube's true position/orientation from MuJoCo is still used
   internally (reward shaping, drop/visibility checks) but is deliberately
   NOT exposed in the observation -- that mirrors what the real robot can
   know, and it's the standard "privileged simulator info for reward only"
   pattern used when training vision-based policies for sim-to-real.
3. "Cube not visible" is now a real geometric check against the fixed
   camera's frustum (not just a fixed-radius workspace check), plus a hard
   floor/anchor-drift safety net in case the frustum math is ever wrong.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces

# --------------------------------------------------------------------------- #
# Names pulled straight from robot.xml / scene.xml. If you rename anything in
# the XML, update it here too.
# --------------------------------------------------------------------------- #
ACTUATOR_NAMES = [
    "motor_finger1_1", "motor_finger1_2",
    "motor_finger2_1", "motor_finger2_2",
    "motor_finger3_1", "motor_finger3_2",
    "motor_finger4_1", "motor_finger4_2",
]  # order == servo ID 1..8, see header comment

ACTUATED_JOINT_NAMES = [
    "finger1_motor1", "finger1_motor2",
    "finger2_motor1", "finger2_motor2",
    "finger3_motor1", "finger3_motor2",
    "finger4_motor1", "finger4_motor2",
]

CUBE_BODY_NAME = "cube"
CUBE_FREEJOINT_NAME = "cube_joint"
PALM_BODY_NAME = "r_wrist_interface"
CAMERA_NAME = "tracking_camera"        # fixed camera declared in scene.xml

# --------------------------------------------------------------------------- #
# Tuning knobs -- deliberately simple thresholds so the env stays easy to
# reason about. Tune for your setup.
# --------------------------------------------------------------------------- #
IMAGE_SIZE = (84, 84)          # (height, width) of the vision observation
CAMERA_FOV_MARGIN = 0.85       # shrink the usable frustum to this fraction of
                                # the real FOV, so "cube near the edge" counts
                                # as lost a little before it's truly clipped
DROP_FALL_MARGIN = 0.12        # m, cube center this far *below* its anchor -> "dropped"
ANCHOR_DRIFT_MARGIN = 0.20     # m, hard safety net on top of the camera check
FLOOR_SAFETY_Z = -0.14         # m, absolute floor safety net (floor top ~ -0.15)
MAX_EPISODE_STEPS = 1000
CONTROL_DECIMATION = 5         # physics substeps per env.step() call
SETTLE_STEPS = 200             # physics steps to let the hand/cube settle on reset

# Constant torque bias (N*m, one entry per actuator, ACTUATOR_NAMES order)
# ramped in during the settle phase so the hand actually closes around the
# cube instead of leaving it to free-fall (these are torque motors, so
# ctrl=0 means "limp"). Placeholder values -- verify signs/magnitudes for
# your hand in the viewer (render_mode="human") and adjust.
GRASP_BIAS = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
GRASP_RAMP_STEPS = 40
GRASP_MAX_ATTEMPTS = 5          # reset() retries with new randomization if the grasp fails
GRASP_FAIL_DROP = 0.05          # m the cube may sink during settle before we call it a failed grasp


class AmazeDexCubeRotateEnv(gym.Env):
    """Rotate a numbered cube in-hand with the Amazing Hand, MuJoCo-backed.

    Action space:
        Box(-1, 1, shape=(8,)) -- one normalized command per finger motor
        (index i == servo ID i+1). Internally rescaled to each motor's
        ctrlrange from robot.xml.

    Observation space (Dict, see module docstring for rationale):
        image        Box(0, 255, (H, W, 3), uint8)
        joint_pos    Box(-inf, inf, (8,), float32)
        joint_vel    Box(-inf, inf, (8,), float32)
        prev_action  Box(-1, 1, (8,), float32)

    Episode ends (terminated=True) when:
        - the cube is dropped (falls well below where it settled, or hits
          the floor safety height), or
        - the cube is not visible to the fixed camera (out of its frustum,
          or drifted far enough from its anchor to be considered lost), or
        - the physics went unstable (NaN/Inf state, or MuJoCo raised a
          fatal error mid-step) -- treated as an automatic "restart",
          never propagated up to crash the training loop.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        model_path: str = os.path.join("my-robot", "scene.xml"),
        render_mode: Optional[str] = None,
        rotation_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
        max_episode_steps: int = MAX_EPISODE_STEPS,
        image_size: tuple[int, int] = IMAGE_SIZE,
        use_camera_obs: bool = True,
    ):
        """
        use_camera_obs: if True (default), observations include an "image"
            key rendered from the fixed camera -- this needs a working
            MuJoCo offscreen renderer (mujoco.Renderer / MjrContext) on this
            machine. If your install can't create one (see the error message
            you'd get otherwise), pass False to train on joint state only
            while you fix the MuJoCo/OpenGL install -- the "cube not
            visible" check is purely geometric (camera pose + fovy) and
            works either way, it doesn't need the renderer.
        """
        super().__init__()

        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.rotation_axis = np.asarray(rotation_axis, dtype=np.float64)
        self.rotation_axis /= np.linalg.norm(self.rotation_axis)
        self._img_h, self._img_w = image_size
        self.use_camera_obs = use_camera_obs

        # ------------------------------------------------------------------- #
        # Load the MuJoCo model + data
        # ------------------------------------------------------------------- #
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # Cache ids/addresses once so step()/reset() stay fast and readable.
        self._actuator_ids = np.array([self.model.actuator(n).id for n in ACTUATOR_NAMES])
        self._joint_qpos_adr = np.array([self.model.joint(n).qposadr[0] for n in ACTUATED_JOINT_NAMES])
        self._joint_dof_adr = np.array([self.model.joint(n).dofadr[0] for n in ACTUATED_JOINT_NAMES])
        self._cube_body_id = self.model.body(CUBE_BODY_NAME).id
        self._cube_qpos_adr = self.model.joint(CUBE_FREEJOINT_NAME).qposadr[0]
        self._cube_dof_adr = self.model.joint(CUBE_FREEJOINT_NAME).dofadr[0]
        self._camera_id = self.model.camera(CAMERA_NAME).id  # raises if missing -- camera is required now

        self._ctrl_low = self.model.actuator_ctrlrange[self._actuator_ids, 0]
        self._ctrl_high = self.model.actuator_ctrlrange[self._actuator_ids, 1]

        # Remember the cube's spawn pose (from the XML) so reset() can restore
        # / randomize around it before the settle phase runs.
        self._init_cube_qpos = self.data.qpos[self._cube_qpos_adr: self._cube_qpos_adr + 7].copy()
        self._cube_anchor = self._init_cube_qpos[:3].copy()

        # ------------------------------------------------------------------- #
        # Gymnasium spaces
        # ------------------------------------------------------------------- #
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)

        obs_spaces = {
            "joint_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32),
            "joint_vel": spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32),
            "prev_action": spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32),
        }
        if self.use_camera_obs:
            obs_spaces["image"] = spaces.Box(
                low=0, high=255, shape=(self._img_h, self._img_w, 3), dtype=np.uint8
            )
        self.observation_space = spaces.Dict(obs_spaces)

        self._prev_action = np.zeros(8, dtype=np.float32)
        self._elapsed_steps = 0

        # `_viewer` / `_renderer` (human / rgb_array render() calls) are
        # created lazily on first use. `_obs_renderer` -- the one that
        # produces the "image" observation every step -- is created *now*,
        # eagerly, so a broken MuJoCo/OpenGL install fails loudly right here
        # with an actionable message instead of crashing deep inside the
        # first reset() with a cryptic AttributeError.
        self._viewer = None
        self._renderer = None
        self._obs_renderer = None
        if self.use_camera_obs:
            self._obs_renderer = self._make_renderer_or_raise()
        self._sim_dt = self.model.opt.timestep * CONTROL_DECIMATION

    # ------------------------------------------------------------------- #
    # Gymnasium API
    # ------------------------------------------------------------------- #
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)

        spawn_z = self._init_cube_qpos[2]
        grasp_ok = False

        for _attempt in range(GRASP_MAX_ATTEMPTS):
            mujoco.mj_resetData(self.model, self.data)

            # Neutral hand pose: all finger joints at 0 rad.
            self.data.qpos[self._joint_qpos_adr] = 0.0
            self.data.qvel[self._joint_dof_adr] = 0.0

            # Restore the cube to its spawn pose with a little randomization
            # so the policy doesn't overfit to one exact starting pose.
            cube_pos_noise = self.np_random.uniform(low=-0.005, high=0.005, size=3)
            yaw_noise = self.np_random.uniform(low=-np.pi, high=np.pi)
            self.data.qpos[self._cube_qpos_adr: self._cube_qpos_adr + 3] = (
                self._init_cube_qpos[:3] + cube_pos_noise
            )
            self.data.qpos[self._cube_qpos_adr + 3: self._cube_qpos_adr + 7] = _quat_from_z_rotation(yaw_noise)
            self.data.qvel[self._cube_dof_adr: self._cube_dof_adr + 6] = 0.0

            mujoco.mj_forward(self.model, self.data)

            # Let the hand settle around the cube before the episode
            # "officially" starts. Torque actuators are limp at ctrl=0, so we
            # ramp in GRASP_BIAS to close the fingers instead of letting the
            # cube free-fall through an open hand.
            settle_ok = True
            for i in range(SETTLE_STEPS):
                ramp = min(1.0, (i + 1) / GRASP_RAMP_STEPS)
                self.data.ctrl[self._actuator_ids] = ramp * GRASP_BIAS
                if not self._safe_mj_step():
                    settle_ok = False
                    break
            if not settle_ok:
                continue  # physics blew up mid-settle -- retry with a fresh randomization

            # Kill residual settle velocity so it doesn't carry into step 1.
            self.data.qvel[self._cube_dof_adr: self._cube_dof_adr + 6] = 0.0
            mujoco.mj_forward(self.model, self.data)

            settled_z = self.data.xpos[self._cube_body_id, 2]
            if (spawn_z - settled_z) < GRASP_FAIL_DROP and settled_z > FLOOR_SAFETY_Z:
                grasp_ok = True
                break

        if not grasp_ok:
            import warnings
            warnings.warn(
                f"Cube fell during the settle phase in all {GRASP_MAX_ATTEMPTS} reset "
                "attempts. GRASP_BIAS is likely the wrong sign/magnitude for one or "
                "more fingers -- open the env with render_mode='human' and tune it.",
                RuntimeWarning,
            )

        # Anchor "dropped" / "lost" checks to where the cube actually settled
        # this episode, not a hardcoded world position.
        self._cube_anchor = self.data.xpos[self._cube_body_id].copy()
        self._prev_action[:] = 0.0
        self._elapsed_steps = 0

        observation = self._get_obs()
        info = self._get_info()
        info["grasp_ok"] = grasp_ok

        if self.render_mode == "human":
            self.render()

        return observation, info

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        action = np.clip(action, self.action_space.low, self.action_space.high)
        ctrl = self._ctrl_low + (action + 1.0) * 0.5 * (self._ctrl_high - self._ctrl_low)
        self.data.ctrl[self._actuator_ids] = ctrl

        physics_ok = True
        for _ in range(CONTROL_DECIMATION):
            if not self._safe_mj_step():
                physics_ok = False
                break

        self._elapsed_steps += 1
        self._prev_action[:] = action

        cube_dropped = (not physics_ok) or self._cube_is_dropped()
        cube_not_visible = (not physics_ok) or self._cube_is_not_visible()
        terminated = bool(cube_dropped or cube_not_visible)

        observation = self._get_obs()
        info = self._get_info()
        info["cube_dropped"] = cube_dropped
        info["cube_not_visible"] = cube_not_visible
        info["physics_diverged"] = not physics_ok

        reward = self._compute_reward(action, terminated)
        truncated = self._elapsed_steps >= self.max_episode_steps

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
                # Lock the viewer onto the same fixed camera used for the
                # vision observation, so what you watch matches what the
                # policy (would) see. This is the built-in equivalent of
                # manually setting viewer.cam.azimuth/elevation/distance/
                # lookat every frame: mjCAMERA_FIXED + fixedcamid just tells
                # the viewer "use camera N from the XML", and MuJoCo keeps it
                # pinned there without any per-frame bookkeeping.
                with self._viewer.lock():
                    self._viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    self._viewer.cam.fixedcamid = self._camera_id
            if self._viewer.is_running():
                self._viewer.sync()
            return None

        if self.render_mode == "human":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._renderer.update_scene(self.data, camera=CAMERA_NAME)
            return self._renderer.render()

        return None

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._obs_renderer is not None:
            self._obs_renderer.close()
            self._obs_renderer = None

    # ------------------------------------------------------------------- #
    # Physics safety
    # ------------------------------------------------------------------- #
    def _safe_mj_step(self) -> bool:
        """Step the physics once. Returns False (instead of crashing the
        process) if the state diverged or MuJoCo raised a fatal error --
        e.g. the constraint/contact buffers in scene.xml being exceeded."""
        try:
            mujoco.mj_step(self.model, self.data)
        except Exception:
            return False
        return self._physics_is_finite()

    def _physics_is_finite(self) -> bool:
        return bool(np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all())

    # ------------------------------------------------------------------- #
    # Observation / camera-visibility helpers
    # ------------------------------------------------------------------- #
    def _make_renderer_or_raise(self) -> "mujoco.Renderer":
        try:
            return mujoco.Renderer(self.model, height=self._img_h, width=self._img_w)
        except Exception as exc:
            raise RuntimeError(
                "Could not create MuJoCo's offscreen renderer, which is needed "
                "for the camera 'image' observation. This is a MuJoCo/OpenGL "
                "install issue on this machine, not a bug in this environment "
                f"(underlying error: {exc!r}).\n"
                "Things to check:\n"
                "  1. Duplicate/mismatched installs -- run "
                "`python -c \"import mujoco; print(mujoco.__file__, mujoco.__version__)\"` "
                "and make sure only ONE mujoco package is on this env's path "
                "(having it installed via both pip AND conda in the same "
                "environment is a common cause). Try: `pip uninstall mujoco -y` "
                "then `pip install --upgrade --force-reinstall mujoco`.\n"
                "  2. Rendering backend -- MuJoCo picks a backend (wgl/glfw/"
                "egl/osmesa) via the MUJOCO_GL env var at import time. If no "
                "GPU/OpenGL context is available (e.g. a remote desktop or "
                "some VMs), try forcing one explicitly before running, e.g. "
                "in PowerShell: `$env:MUJOCO_GL = \"wgl\"`.\n"
                "  3. Immediate workaround so training isn't blocked while you "
                "sort this out: construct the env with `use_camera_obs=False` "
                "to train on joint state only until rendering is fixed."
            ) from exc

    def _render_camera_image(self) -> np.ndarray:
        self._obs_renderer.update_scene(self.data, camera=CAMERA_NAME)
        return self._obs_renderer.render()

    def _get_obs(self) -> dict[str, np.ndarray]:
        obs = {
            "joint_pos": self.data.qpos[self._joint_qpos_adr].astype(np.float32),
            "joint_vel": self.data.qvel[self._joint_dof_adr].astype(np.float32),
            "prev_action": self._prev_action.copy(),
        }
        if self.use_camera_obs:
            obs["image"] = self._render_camera_image()
        return obs

    def _get_info(self) -> dict[str, Any]:
        cube_pos_world = self.data.xpos[self._cube_body_id]
        return {
            "cube_pos": cube_pos_world.copy(),          # privileged sim info, for logging/reward only
            "distance_from_anchor": float(np.linalg.norm(cube_pos_world - self._cube_anchor)),
            "elapsed_steps": self._elapsed_steps,
        }

    def _cube_in_camera_frustum(self) -> bool:
        """Geometric check: is the cube center within the fixed camera's
        field of view and in front of it? Uses the camera's actual pose and
        fovy from the model, so it stays correct even if the camera in
        scene.xml is moved."""
        cam_pos = self.data.cam_xpos[self._camera_id]
        cam_mat = self.data.cam_xmat[self._camera_id].reshape(3, 3)  # columns = camera's local axes in world frame

        cube_pos = self.data.xpos[self._cube_body_id]
        p_cam = cam_mat.T @ (cube_pos - cam_pos)  # cube position in camera-local frame

        depth = -p_cam[2]  # MuJoCo cameras look down their local -z axis
        if depth <= 1e-6:
            return False  # behind or on top of the camera

        fovy = np.deg2rad(self.model.cam_fovy[self._camera_id])
        half_v = np.tan(fovy / 2.0) * depth * CAMERA_FOV_MARGIN
        half_h = half_v * (self._img_w / self._img_h)

        return bool(abs(p_cam[0]) < half_h and abs(p_cam[1]) < half_v)

    def _cube_is_dropped(self) -> bool:
        """Cube fell out of the hand: well below where it started, or hit
        the absolute floor safety height."""
        cube_z = self.data.xpos[self._cube_body_id, 2]
        fell_from_anchor = (self._cube_anchor[2] - cube_z) > DROP_FALL_MARGIN
        hit_floor = cube_z < FLOOR_SAFETY_Z
        return bool(fell_from_anchor or hit_floor)

    def _cube_is_not_visible(self) -> bool:
        """Cube isn't visible to the fixed camera: outside its frustum, or
        (safety net) drifted far enough from its anchor to be considered
        lost regardless of the frustum math."""
        if not self._cube_in_camera_frustum():
            return True
        cube_pos_world = self.data.xpos[self._cube_body_id]
        distance = np.linalg.norm(cube_pos_world - self._cube_anchor)
        return bool(distance > ANCHOR_DRIFT_MARGIN)

    # ------------------------------------------------------------------- #
    # Reward
    # ------------------------------------------------------------------- #
    def _compute_reward(self, action: np.ndarray, terminated: bool) -> float:
        if terminated:
            return -10.0  # flat penalty for dropping / losing the cube

        cube_angvel = self.data.cvel[self._cube_body_id, 0:3]
        spin_reward = float(np.dot(cube_angvel, self.rotation_axis))  # turn the cube

        cube_pos_world = self.data.xpos[self._cube_body_id]
        distance = np.linalg.norm(cube_pos_world - self._cube_anchor)
        grasp_bonus = 1.0 - np.clip(distance / ANCHOR_DRIFT_MARGIN, 0.0, 1.0)  # hold it tight

        effort_penalty = 0.01 * float(np.sum(np.square(action)))  # smooth control

        return float(spin_reward + 0.1 * grasp_bonus - effort_penalty)


def _quat_from_z_rotation(angle_rad: float) -> np.ndarray:
    """MuJoCo-style (w, x, y, z) quaternion for a rotation about world z."""
    half = angle_rad / 2.0
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)


# --------------------------------------------------------------------------- #
# Register with Gymnasium so it can be created via gym.make(...)
# --------------------------------------------------------------------------- #
try:
    gym.register(
        id="AmazeDexCubeRotate-v0",
        entry_point=f"{__name__}:AmazeDexCubeRotateEnv",
        max_episode_steps=MAX_EPISODE_STEPS,
    )
except gym.error.Error:
    pass  # already registered (e.g. this module was imported more than once)


if __name__ == "__main__":
    # Smoke test: random-action rollout. `--episodes` is a real parameter now
    # (the old script's `for episode in range(3)` was hardcoded, which is
    # probably why it looked like it "crashed" after exactly 3 runs).
    import argparse

    parser = argparse.ArgumentParser(description="AmazeDex cube-rotate env smoke test")
    parser.add_argument("--model-path", default=os.path.join("my-robot", "scene.xml"))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--render", choices=["none", "human"], default="none")
    args = parser.parse_args()

    env = AmazeDexCubeRotateEnv(
        model_path=args.model_path,
        render_mode=None if args.render == "none" else args.render,
    )

    for episode in range(args.episodes):
        observation, info = env.reset(seed=episode)
        episode_reward = 0.0
        terminated = truncated = False

        # Each episode is isolated in its own try/except: one bad episode
        # (e.g. a MuJoCo error that slips past `_safe_mj_step`) logs and
        # moves on to the next reset instead of killing the whole run.
        try:
            while not (terminated or truncated):
                action = env.action_space.sample()
                observation, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
        except Exception as exc:  # pragma: no cover - defensive, see docstring
            print(f"Episode {episode}: crashed mid-step ({exc!r}); moving on to next episode.")
            continue

        if info.get("cube_dropped"):
            reason = "cube dropped"
        elif info.get("cube_not_visible"):
            reason = "cube not visible to camera"
        else:
            reason = "max steps reached"

        print(
            f"Episode {episode}: return={episode_reward:.2f}, "
            f"steps={info['elapsed_steps']}, ended because: {reason}"
        )

    env.close()