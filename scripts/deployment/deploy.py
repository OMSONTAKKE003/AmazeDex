import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

# Automatically add project root (~/AmazeDex) and parent directories to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1] if len(SCRIPT_DIR.parents) > 1 else SCRIPT_DIR.parent
for p in (SCRIPT_DIR, SCRIPT_DIR.parent, PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import cv2
import mujoco
import mujoco.viewer
import numpy as np
from pupil_apriltags import Detector
from rustypot import Scs0009PyController

try:
    import env.register_amazedex_env  # noqa: F401
    from env.amazedex_cube_env import (
        ACTUATORS,
        ANGVEL_SCALE,
        CFG,
        CUBE_LOCAL_CENTER,
        FACE_NORMALS,
        JOINTS,
        JVEL_SCALE,
        MODEL_PATH,
        REACH_NORM,
        TIP_SITES,
    )
except ModuleNotFoundError:
    import envs.register_amazedex_env  # noqa: F401
    from envs.amazedex_cube_env import (
        ACTUATORS,
        ANGVEL_SCALE,
        CFG,
        CUBE_LOCAL_CENTER,
        FACE_NORMALS,
        JOINTS,
        JVEL_SCALE,
        MODEL_PATH,
        REACH_NORM,
        TIP_SITES,
    )

from sbx import SAC
from scipy.spatial.transform import Rotation as R

CONTROL_DT = 0.02
SERVO_IDS = [1, 2, 3, 4, 5, 6, 7, 8]
GOAL_SPEED = 200  # Raw SCS0009 integer speed matching script convention
CALIBRATION_PATH = str(SCRIPT_DIR / "hand_calibration.json")

# World reference orientation (Z-axis pointing UP in the ref tag's own frame)
REF_Z_AXIS = np.array([0.0, 0.0, 1.0], dtype=np.float32)
DIGIT_TO_FACE_INDEX = {6: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
FACE_INDEX_TO_DIGIT = {v: k for k, v in DIGIT_TO_FACE_INDEX.items()}
# ===========================================================================
# Reference-tag & Cube configuration
# ===========================================================================

REF_TAG_FAMILY = "tag36h11"
REFERENCE_TAG_ID = 6
REFERENCE_TAG_SIZE = 0.05  # meters = 5 cm
CUBE_TAG_SIZE = 0.01       # meters = 1 cm
CUBE_HALF_EDGE = 0.024     # meters = 2.4 cm

DEFAULT_MAX_HOLD_FRAMES = 15

# Cube geometry definition relative to cube center
h = CUBE_HALF_EDGE
FACE_OFFSET_FROM_CENTER = {
    0: np.array([0, 0, -h]),
    1: np.array([h, 0, 0]),
    2: np.array([-h, 0, 0]),
    3: np.array([0, h, 0]),
    4: np.array([0, -h, 0]),
    5: np.array([0, 0, h]),
}

FACE_QUATS_WXYZ = {
    0: (-1, 0, 0, 0),
    1: (0.7071068, 0, 0.7071068, 0),
    2: (0.7071068, 0, -0.7071068, 0),
    3: (0.7071068, -0.7071068, 0, 0),
    4: (0.7071068, 0.7071068, 0, 0),
    5: (1, 0, 0, 0),
}

FACE_ROTMATS = {
    tag_id: R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    for tag_id, q in FACE_QUATS_WXYZ.items()
}
CUBE_TAG_IDS = set(FACE_ROTMATS.keys())


class CameraCalib:
    def __init__(self, mtx: np.ndarray, dist: np.ndarray):
        self.mtx = mtx
        self.dist = dist


def load_camera_calib(path: str) -> CameraCalib:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Calibration file not found: {path}")

    if path.endswith(".npz"):
        data = np.load(path)
        mtx = data.get("camMatrix", data.get("mtx", data.get("K", data.get("camera_matrix"))))
        dist = data.get("distCoeff", data.get("dist", data.get("D", data.get("distortion_coefficients", data.get("distortion")))))

        if mtx is None or dist is None:
            raise KeyError(f"Could not find matrix or distortion keys in '{path}'. Available keys: {data.files}")

        return CameraCalib(mtx, dist)

    raise ValueError(f"Unsupported calibration format: {path}")


def is_valid_rotation(rotmat, tol: float = 1e-3) -> bool:
    rotmat = np.asarray(rotmat)
    if rotmat.shape != (3, 3):
        return False
    determinant = np.linalg.det(rotmat)
    orthogonal_error = np.linalg.norm(rotmat @ rotmat.T - np.eye(3))
    return determinant > 0.5 and orthogonal_error < tol


def face_pose_to_cube_pose(face_pos_cam, face_rot_cam, tag_id):
    face_rot_local = FACE_ROTMATS[tag_id]
    offset_local = FACE_OFFSET_FROM_CENTER[tag_id]
    cube_rot_cam = face_rot_cam @ face_rot_local.T
    cube_pos_cam = face_pos_cam - cube_rot_cam @ offset_local
    return cube_pos_cam, cube_rot_cam


class ApriltagCubeTracker:
    def __init__(self, camera_calib: CameraCalib, face_normals=None):
        self.calib = camera_calib
        mtx = camera_calib.mtx
        self.camera_params = [mtx[0, 0], mtx[1, 1], mtx[0, 2], mtx[1, 2]]
        self.detector = Detector(
            families=REF_TAG_FAMILY,
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )

    def update_quat(self, frame_bgr: np.ndarray):
        frame_undist = cv2.undistort(frame_bgr, self.calib.mtx, self.calib.dist)
        gray = cv2.cvtColor(frame_undist, cv2.COLOR_BGR2GRAY)
        detections = self.detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=self.camera_params,
            tag_size=CUBE_TAG_SIZE,
        )

        cube_rot_estimates = []
        for tag in detections:
            if tag.tag_id in CUBE_TAG_IDS:
                face_pos_cam = np.asarray(tag.pose_t, dtype=np.float32).flatten()
                face_rot_cam = np.asarray(tag.pose_R, dtype=np.float32)

                _, cube_rot_cam = face_pose_to_cube_pose(face_pos_cam, face_rot_cam, tag.tag_id)
                if is_valid_rotation(cube_rot_cam):
                    cube_rot_estimates.append(cube_rot_cam)

        if not cube_rot_estimates:
            return None

        if len(cube_rot_estimates) == 1:
            fused_rot_cam = cube_rot_estimates[0]
        else:
            try:
                fused_rot_cam = R.from_matrix(cube_rot_estimates).mean().as_matrix()
            except ValueError:
                fused_rot_cam = cube_rot_estimates[0]

        return rotmat_to_quat_wxyz(fused_rot_cam)


class ReferenceTagTracker:
    def __init__(self, camera_calib: CameraCalib, max_hold_frames: int = DEFAULT_MAX_HOLD_FRAMES):
        self.calib = camera_calib
        mtx = camera_calib.mtx
        self.camera_params = [mtx[0, 0], mtx[1, 1], mtx[0, 2], mtx[1, 2]]
        self.detector = Detector(
            families=REF_TAG_FAMILY,
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )
        self.max_hold_frames = max_hold_frames
        self._last_rot = None
        self._last_pos = None
        self._missing_frames = 0

    def update(self, frame_bgr: np.ndarray):
        frame_undist = cv2.undistort(frame_bgr, self.calib.mtx, self.calib.dist)
        gray = cv2.cvtColor(frame_undist, cv2.COLOR_BGR2GRAY)
        detections = self.detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=self.camera_params,
            tag_size=REFERENCE_TAG_SIZE,
        )

        for tag in detections:
            if tag.tag_id != REFERENCE_TAG_ID:
                continue
            pos = np.asarray(tag.pose_t, dtype=np.float32).flatten()
            rot = np.asarray(tag.pose_R, dtype=np.float32)
            if not is_valid_rotation(rot):
                break
            self._last_pos = pos
            self._last_rot = rot
            self._missing_frames = 0
            return pos, rot, True

        self._missing_frames += 1
        if self._last_rot is not None and self._missing_frames <= self.max_hold_frames:
            return self._last_pos, self._last_rot, False
        return None, None, False


class CubeQuatTracker:
    def __init__(self, tracker: ApriltagCubeTracker, max_hold_frames: int = DEFAULT_MAX_HOLD_FRAMES):
        self.tracker = tracker
        self.max_hold_frames = max_hold_frames
        self._last_quat = None
        self._missing_frames = 0

    def update(self, frame_bgr: np.ndarray):
        quat = self.tracker.update_quat(frame_bgr)

        if quat is not None:
            self._last_quat = np.asarray(quat, dtype=np.float32).copy()
            self._missing_frames = 0
            return self._last_quat, True

        self._missing_frames += 1
        if self._last_quat is not None and self._missing_frames <= self.max_hold_frames:
            return self._last_quat, False
        return None, False


def rotmat_to_quat_wxyz(rotmat: np.ndarray) -> np.ndarray:
    xyzw = R.from_matrix(rotmat).as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float32)


def quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return R.from_quat([x, y, z, w]).as_matrix()


def cube_quat_to_ref_frame(cube_quat_cam_wxyz: np.ndarray, ref_rot_cam: np.ndarray) -> np.ndarray:
    R_cube_cam = quat_wxyz_to_rotmat(cube_quat_cam_wxyz)
    R_cube_ref = ref_rot_cam.T @ R_cube_cam
    return rotmat_to_quat_wxyz(R_cube_ref)


def get_current_facing_digit(cube_quat_ref: np.ndarray, ref_z: np.ndarray = REF_Z_AXIS) -> int:
    """Determines which cube face digit is currently facing up in the reference frame."""
    R_cube = quat_wxyz_to_rotmat(cube_quat_ref)
    dots = [np.dot(R_cube @ FACE_NORMALS[f_idx], ref_z) for f_idx in range(6)]
    current_face_idx = int(np.argmax(dots))
    return FACE_INDEX_TO_DIGIT[current_face_idx]


# ===========================================================================
# Kinematics / action mapping / servo I/O
# ===========================================================================


class HandKinematics:
    def __init__(self, model_path: str = MODEL_PATH):
        if not os.path.exists(model_path):
            alt_path = SCRIPT_DIR / "resources" / "scene.xml"
            if alt_path.exists():
                model_path = str(alt_path)
            else:
                raise FileNotFoundError(
                    f"MuJoCo XML file not found at '{model_path}' or '{alt_path}'."
                )

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.qposids = np.array(
            [self.model.joint(n).qposadr[0] for n in JOINTS]
        )
        self.cubeid = self.model.body("cube").id
        self.tip_sites = [self.model.site(n).id for n in TIP_SITES]
        actids = np.array([self.model.actuator(n).id for n in ACTUATORS])
        lims = self.model.actuator_ctrlrange[actids]
        self.ctrl_lo, self.ctrl_hi = lims[:, 0], lims[:, 1]
        self.ctrl_mid = (self.ctrl_lo + self.ctrl_hi) / 2
        self.ctrl_half = (self.ctrl_hi - self.ctrl_lo) / 2
        mujoco.mj_forward(self.model, self.data)
        cube_body_pos = np.array(self.model.body("cube").pos, dtype=np.float32)
        self.nominal_cube_pos = cube_body_pos + CUBE_LOCAL_CENTER

    def tip_to_cube(self, joint_pos_rad: np.ndarray) -> np.ndarray:
        self.data.qpos[self.qposids] = np.clip(
            joint_pos_rad, self.ctrl_lo, self.ctrl_hi
        )
        mujoco.mj_forward(self.model, self.data)
        cpos = self.nominal_cube_pos
        return np.array(
            [cpos - self.data.site_xpos[s] for s in self.tip_sites],
            dtype=np.float32,
        )

    def sync_pose(self, joint_pos_rad: np.ndarray) -> None:
        """Mirror a real joint-position reading into this model's qpos and
        run forward kinematics, so the model reflects the real hand's current
        pose (for viewer display, or as a cheap FK utility)."""
        self.data.qpos[self.qposids] = np.clip(
            joint_pos_rad, self.ctrl_lo, self.ctrl_hi
        )
        mujoco.mj_forward(self.model, self.data)


class ActionMapper:
    def __init__(self, kin: HandKinematics, cfg=CFG):
        self.kin = kin
        self.cfg = cfg
        self.grasp_frac = None
        self.filtered_ctrl = None

    def start(self, current_joint_pos_rad: np.ndarray) -> None:
        frac = np.clip(
            (current_joint_pos_rad - self.kin.ctrl_mid) / self.kin.ctrl_half,
            -1.0,
            1.0,
        )
        self.grasp_frac = frac.copy()
        self.filtered_ctrl = frac.copy()

    def action_to_ctrl(self, action: np.ndarray) -> np.ndarray:
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        # NOTE: anchored to self.filtered_ctrl (not self.grasp_frac) to match
        # the training-env fix in amazedex_cube_env.py's step(). Anchoring to
        # a fixed pose captured once at episode/rollout start makes
        # target_frac converge to a fixed point for a constant action, so
        # filtered_ctrl freezes within a handful of steps regardless of what
        # the policy outputs afterward -- the "one movement then stop" bug.
        # Anchoring to filtered_ctrl makes this a genuine velocity command,
        # matching what the policy was actually trained against.
        target_frac = np.clip(
            self.filtered_ctrl + action * self.cfg.grasp_band_frac, -1.0, 1.0
        )
        lpf_ctrl = (
            self.cfg.action_lpf * target_frac
            + (1 - self.cfg.action_lpf) * self.filtered_ctrl
        )
        rate = self.cfg.max_ctrl_rate_frac
        step_delta = np.clip(lpf_ctrl - self.filtered_ctrl, -rate, rate)
        self.filtered_ctrl = self.filtered_ctrl + step_delta
        return self.kin.ctrl_mid + self.filtered_ctrl * self.kin.ctrl_half


def load_calibration(path: str = CALIBRATION_PATH) -> dict:
    if not os.path.exists(path):
        print("[WARN] No hand calibration file found. Using default offsets.")
        return {"offset_rad": [0.0] * 8, "sign": [1.0] * 8}
    with open(path) as f:
        return json.load(f)


class HandInterface:
    MAX_RAW_DELTA_PER_TICK = None
    SPEED_MIN, SPEED_MAX = 1, 1023

    def __init__(
        self,
        serial_port: str = "/dev/ttyACM0",
        baudrate: int = 1_000_000,
        timeout: float = 0.5,
        calibration: dict | None = None,
        goal_speed: int = GOAL_SPEED,
    ):
        self.controller = Scs0009PyController(
            serial_port=serial_port, baudrate=baudrate, timeout=timeout
        )

        for servo_id in SERVO_IDS:
            self.controller.write_torque_enable(servo_id, 1)
            time.sleep(0.0002)

        self.goal_speed = None
        self.set_goal_speed(goal_speed)

        calib = calibration or {"offset_rad": [0.0] * 8, "sign": [1.0] * 8}
        self.offset = np.asarray(calib["offset_rad"], dtype=np.float32)
        self.sign = np.asarray(calib["sign"], dtype=np.float32)
        self._last_raw_cmd = None
        self._last_raw_positions = None

    def set_goal_speed(
        self, speed: int, servo_ids: list[int] | None = None
    ) -> None:
        speed = int(speed)
        if not (self.SPEED_MIN <= speed <= self.SPEED_MAX):
            raise ValueError(
                f"goal_speed must be an integer in [{self.SPEED_MIN},"
                f" {self.SPEED_MAX}]. Got: {speed!r}"
            )
        ids = servo_ids if servo_ids is not None else SERVO_IDS
        failed = []
        for servo_id in ids:
            try:
                self.controller.write_goal_speed(servo_id, speed)
            except RuntimeError as e:
                failed.append(servo_id)
                print(f"[WARN] set_goal_speed: servo {servo_id} failed: {e}")
            time.sleep(0.0002)

        if servo_ids is None:
            self.goal_speed = speed
        if failed:
            print(f"[WARN] Servo goal speed set for {ids}, but {failed} did not respond.")
        else:
            print(f"[INFO] Servo goal speed set to {speed} for servos {ids}")

    def read_joint_positions(self) -> np.ndarray:
        raw_vals = []
        for i in SERVO_IDS:
            try:
                pos = self.controller.read_present_position(i)
                if hasattr(pos, "__len__"):
                    pos = pos[0]
                raw_vals.append(float(pos))
            except RuntimeError as e:
                print(f"[WARN] read_joint_positions: servo {i} failed: {e}")
                # Fall back to last known raw value for this servo so the
                # control loop can keep running instead of crashing outright.
                if self._last_raw_positions is not None:
                    raw_vals.append(float(self._last_raw_positions[len(raw_vals)]))
                else:
                    raw_vals.append(0.0)
            time.sleep(0.0002)

        raw = np.array(raw_vals, dtype=np.float32)
        self._last_raw_positions = raw.copy()
        return self.sign * (raw - self.offset)

    def send_ctrl(self, ctrl_sim_frame: np.ndarray) -> None:
        raw = self.offset + self.sign * ctrl_sim_frame
        if self.MAX_RAW_DELTA_PER_TICK is not None:
            if self._last_raw_cmd is None:
                self._last_raw_cmd = raw.copy()
            delta = np.clip(
                raw - self._last_raw_cmd,
                -self.MAX_RAW_DELTA_PER_TICK,
                self.MAX_RAW_DELTA_PER_TICK,
            )
            raw = self._last_raw_cmd + delta
            self._last_raw_cmd = raw.copy()

        failed = []
        for servo_id, goal_pos in zip(SERVO_IDS, raw):
            try:
                self.controller.write_goal_position(servo_id, float(goal_pos))
            except RuntimeError as e:
                failed.append(servo_id)
                print(f"[WARN] send_ctrl: servo {servo_id} failed: {e}")
            time.sleep(0.0002)

        if failed:
            print(f"[WARN] send_ctrl: servos {failed} did not respond this tick.")

        time.sleep(0.005)

    def _safe_servo_write(
        self, action_func, sid: int, *args, action_name: str = "write command"
    ) -> None:
        """Helper to catch write errors without repeating boilerplate."""
        try:
            action_func(sid, *args)
        except RuntimeError as e:
            print(
                f"[WARN] reset_to_home: could not {action_name} for servo {sid}: {e}"
            )

    def reset_to_home(self) -> None:
        """Slowly moves odd servos (1,3,5,7) to -511 and even servos (2,4,6,8) to 511, then disables torque."""
        print(
            "\n[INFO] Safe shutdown: odd IDs returning to -511, even IDs to 511..."
        )
        slow_speed = 0.5

        # Set slow speed for all servos
        for sid in SERVO_IDS:
            self._safe_servo_write(
                self.controller.write_goal_speed,
                sid,
                slow_speed,
                action_name="set slow speed",
            )
            time.sleep(0.001)

        # Command target positions (-511 for odd, 511 for even)
        for sid in SERVO_IDS:
            target_pos = -511 if (sid % 2 != 0) else 511
            try:
                if hasattr(self.controller, "write_raw_goal_position"):
                    self.controller.write_raw_goal_position(sid, target_pos)
                else:
                    self.controller.write_goal_position(
                        sid, float(np.deg2rad(target_pos))
                    )
            except RuntimeError as e:
                print(
                    f"[WARN] reset_to_home: servo {sid} position write failed: {e}"
                )
            time.sleep(0.001)

        time.sleep(2.0)  # Allow time for servos to complete movement

        # Disable torque for all servos
        for sid in SERVO_IDS:
            self._safe_servo_write(
                self.controller.write_torque_enable,
                sid,
                0,
                action_name="disable torque",
            )
            time.sleep(0.0002)

    def close(self) -> None:
        """Safely shuts down the robot and releases controller resources."""
        try:
            self.reset_to_home()
        except Exception as e:
            print(f"[WARN] close: reset_to_home failed entirely ({e}), forcing torque off...")
            for sid in SERVO_IDS:
                self._safe_servo_write(
                    self.controller.write_torque_enable,
                    sid,
                    0,
                    action_name="force disable torque",
                )
                time.sleep(0.0002)
        finally:
            print("[INFO] Safe shutdown complete.")


# ===========================================================================
# Quaternion-first observation building
# ===========================================================================


def desired_rot_axis(
    R_cube: np.ndarray, target_face: int, ref_z: np.ndarray = REF_Z_AXIS
) -> np.ndarray | None:
    n = R_cube @ FACE_NORMALS[target_face]
    axis = np.cross(n, ref_z)
    norm = np.linalg.norm(axis)
    if norm < 1e-6:
        if np.dot(n, ref_z) < 0:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        return None
    return (axis / norm).astype(np.float32)


def cube_angvel_from_quats(
    quat_prev: np.ndarray, quat_curr: np.ndarray, dt: float
) -> np.ndarray:
    R_prev = quat_wxyz_to_rotmat(quat_prev)
    R_curr = quat_wxyz_to_rotmat(quat_curr)
    dR = R_curr @ R_prev.T
    rvec, _ = cv2.Rodrigues(dR)
    return (rvec.flatten() / dt).astype(np.float32)


def build_cube_obs(
    jpos_raw,
    jvel_raw,
    last_action,
    target_face,
    kin: HandKinematics,
    cube_quat: np.ndarray,
    cube_angvel_raw: np.ndarray,
    tip_to_cube_n: np.ndarray | None = None,
) -> np.ndarray:
    R_cube = quat_wxyz_to_rotmat(cube_quat)

    jpos = np.clip((jpos_raw - kin.ctrl_mid) / kin.ctrl_half, -1.0, 1.0)
    jvel = np.clip(jvel_raw / JVEL_SCALE, -1.0, 1.0)

    target_world_normal = R_cube @ FACE_NORMALS[target_face]
    cube_quat_norm = cube_quat.astype(np.float32) / np.linalg.norm(cube_quat)

    cube_angvel = np.clip(cube_angvel_raw / ANGVEL_SCALE, -1.0, 1.0)

    one_hot = np.zeros(6, dtype=np.float32)
    one_hot[target_face] = 1.0

    # NOTE: previously hardcoded to zeros here, but _get_obs() in the
    # training env (amazedex_cube_env.py) computes this live every step via
    # self._tip_to_cube(), so the policy was trained expecting real values.
    # Now computed from forward kinematics on the real joint reading
    # (kin.tip_to_cube), same normalization as training (REACH_NORM, clip
    # to [-3, 3]). This assumes the cube sits at its nominal in-hand
    # position (kin.nominal_cube_pos) rather than the tracked AprilTag
    # pose -- same approximation the FK-only tip_to_cube already makes.
    if tip_to_cube_n is None:
        tip_to_cube_n = np.zeros(12, dtype=np.float32)

    axis = desired_rot_axis(R_cube, target_face)
    axis_obs = axis if axis is not None else np.zeros(3, dtype=np.float32)

    return np.concatenate([
        jpos,
        jvel,
        last_action,
        cube_quat_norm,
        cube_angvel,
        target_world_normal,
        one_hot,
        tip_to_cube_n,
        axis_obs,
    ]).astype(np.float32)


def theta_rad(
    cube_quat: np.ndarray, target_face: int, ref_z: np.ndarray = REF_Z_AXIS
) -> float:
    R_cube = quat_wxyz_to_rotmat(cube_quat)
    n = R_cube @ FACE_NORMALS[target_face]
    return float(np.arccos(np.clip(np.dot(n, ref_z), -1.0, 1.0)))


def predict_action(model: SAC, obs: np.ndarray) -> np.ndarray:
    action, _ = model.predict(obs, deterministic=True)
    return np.clip(action, -1.0, 1.0)


# ===========================================================================
# Validation Helper
# ===========================================================================


def run_vision_validation(
    cube_tracker: CubeQuatTracker,
    ref_tracker: ReferenceTagTracker,
    cap: cv2.VideoCapture,
    camera_calib: CameraCalib,
) -> None:
    print(
        "\n[VALIDATION] Running AprilTag Tracker Test (ref-tag world frame)."
        " Press 'q' in window to exit."
    )
    while True:
        ok, raw_frame = cap.read()
        if not ok:
            print("[ERROR] Failed to grab frame from camera.")
            break

        display = cv2.undistort(raw_frame, camera_calib.mtx, camera_calib.dist)

        cube_quat_cam, cube_fresh = cube_tracker.update(raw_frame)
        ref_pos, ref_rot, ref_fresh = ref_tracker.update(raw_frame)

        if cube_quat_cam is None:
            text = "CUBE TAG NOT VISIBLE -- no cube pose"
            color = (0, 0, 255)
        elif ref_rot is None:
            text = "REF TAG NOT VISIBLE -- no world frame"
            color = (0, 0, 255)
        else:
            cube_quat_world = cube_quat_to_ref_frame(cube_quat_cam, ref_rot)
            current_digit = get_current_facing_digit(cube_quat_world)
            status = "fresh" if (cube_fresh and ref_fresh) else "held"
            text = f"Facing: {current_digit} | Quat[ref,{status}] [w,x,y,z]: {np.round(cube_quat_world, 3)}"
            color = (0, 255, 0) if status == "fresh" else (0, 165, 255)

        cv2.putText(
            display, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )
        cv2.imshow("AprilTag Validation", display)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
            break
    cv2.destroyAllWindows()


# ===========================================================================
# Main control loop
# ===========================================================================


def execute_target(
    model,
    hand: HandInterface,
    cube_tracker: CubeQuatTracker,
    ref_tracker: ReferenceTagTracker,
    target_face: int,
    kin: HandKinematics,
    cap: cv2.VideoCapture,
    camera_calib: CameraCalib,
    cfg=CFG,
    sim_viewer=None,
) -> bool:
    print(f"\n--- Target: face {FACE_INDEX_TO_DIGIT[target_face]} ---")

    mapper = ActionMapper(kin, cfg)
    joint_pos = hand.read_joint_positions()
    mapper.start(joint_pos)
    prev_joint_pos = joint_pos.copy()
    prev_action = np.zeros(8, dtype=np.float32)

    identity_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    ok, first_frame = cap.read()
    if ok and first_frame is not None:
        cube_quat_cam, _ = cube_tracker.update(first_frame)
        _, ref_rot, _ = ref_tracker.update(first_frame)
        prev_quat = (
            cube_quat_to_ref_frame(cube_quat_cam, ref_rot)
            if (ref_rot is not None and cube_quat_cam is not None)
            else identity_quat
        )
    else:
        prev_quat = identity_quat
    hold = 0

    for step in range(cfg.max_steps):
        loop_start = time.perf_counter()

        joint_pos = hand.read_joint_positions()
        joint_vel = (joint_pos - prev_joint_pos) / CONTROL_DT

        # Live tip-to-cube-center vectors from forward kinematics on the
        # real joint reading, matching what the policy saw in training
        # (see build_cube_obs note above). This call also leaves kin.data
        # synced to the current real pose, which the sim viewer below reuses
        # directly -- no extra FK pass needed.
        tip_to_cube_raw = kin.tip_to_cube(joint_pos)
        tip_to_cube_n = np.clip(tip_to_cube_raw / REACH_NORM, -3.0, 3.0).flatten()

        if sim_viewer is not None and sim_viewer.is_running():
            sim_viewer.sync()

        ok, raw_frame = cap.read()
        if ok and raw_frame is not None:
            cube_quat_cam, cube_fresh = cube_tracker.update(raw_frame)
            _, ref_rot, ref_fresh = ref_tracker.update(raw_frame)
        else:
            cube_quat_cam, cube_fresh = None, False
            ref_rot, ref_fresh = None, False

        if ref_rot is not None and cube_quat_cam is not None:
            cube_quat = cube_quat_to_ref_frame(cube_quat_cam, ref_rot)
        else:
            cube_quat = prev_quat

        both_fresh = cube_fresh and ref_fresh
        if both_fresh:
            cube_angvel = cube_angvel_from_quats(prev_quat, cube_quat, CONTROL_DT)
        else:
            cube_angvel = np.zeros(3, dtype=np.float32)
        prev_quat = cube_quat.copy()

        obs = build_cube_obs(
            joint_pos,
            joint_vel,
            prev_action,
            target_face,
            kin,
            cube_quat=cube_quat,
            cube_angvel_raw=cube_angvel,
            tip_to_cube_n=tip_to_cube_n,
        )
        action = predict_action(model, obs)
        target_ctrl = mapper.action_to_ctrl(action)

        # Output raw policy action and resulting servo targets (in radians)
        print(
            f"[Step {step:03d}] Policy Action: {np.round(action, 2)} | "
            f"Servo Ctrl Target (rad): {np.round(target_ctrl, 3)}"
        )

        hand.send_ctrl(target_ctrl)

        prev_joint_pos = joint_pos.copy()
        prev_action = action.copy()

        theta = theta_rad(cube_quat, target_face)
        settled = (
            theta < cfg.success_theta_rad
            and np.linalg.norm(cube_angvel) < cfg.success_max_angvel
        )
        hold = hold + 1 if settled else 0

        if raw_frame is not None:
            if cube_quat_cam is None:
                cube_status = "LOST"
            elif cube_fresh:
                cube_status = "fresh"
            else:
                cube_status = "held"
            ref_status = "fresh" if ref_fresh else ("held" if ref_rot is not None else "LOST")
            any_lost = cube_quat_cam is None or ref_rot is None

            # Compute current facing digit
            current_digit = get_current_facing_digit(cube_quat)

            # Displays the undistorted camera frame during execution
            display = cv2.undistort(raw_frame, camera_calib.mtx, camera_calib.dist)

            # Line 1: Target vs Currently Facing & Status info
            cv2.putText(
                display,
                f"Target: {FACE_INDEX_TO_DIGIT[target_face]}  Facing: {current_digit} "
                f" theta={np.degrees(theta):.1f}deg  hold={hold}"
                f"  cube={cube_status}  ref={ref_status}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255) if any_lost else (0, 255, 0),
                2,
            )

            # Line 2: Quaternion values overlay
            quat_text = f"Cube Quat [w,x,y,z]: {np.round(cube_quat, 3)}"
            cv2.putText(
                display,
                quat_text,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )

            cv2.imshow("AmazeDex Cube Execution", display)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                print("--- ABORTED (quit) ---")
                return False

        if sim_viewer is not None and not sim_viewer.is_running():
            print("--- ABORTED (sim viewer closed) ---")
            return False

        if hold >= cfg.success_hold_steps:
            print(
                f"--- SUCCESS: face {FACE_INDEX_TO_DIGIT[target_face]} reached"
                f" and held ({step + 1} steps) ---"
            )
            return True

        elapsed = time.perf_counter() - loop_start
        if elapsed < CONTROL_DT:
            time.sleep(CONTROL_DT - elapsed)

    print(
        f"--- TIMEOUT: target face {FACE_INDEX_TO_DIGIT[target_face]} not"
        f" reached in {cfg.max_steps} steps ---"
    )
    return False


def prompt_target_face(default: int | None) -> int | None:
    raw = input(
        "Target face -- digit as printed on the cube [0-5] (blank to quit): "
    ).strip()
    if raw == "":
        return None
    if raw.isdigit() and 0 <= int(raw) <= 5:
        return DIGIT_TO_FACE_INDEX[int(raw)]
    print("Invalid input, expected 0-5.")
    return default


def main() -> None:
    global REFERENCE_TAG_ID, REFERENCE_TAG_SIZE

    parser = argparse.ArgumentParser(
        description=(
            "Quat-driven real-hardware deployment for AmazeDex cube rotation,"
            " with world origin defined by a fixed scene reference AprilTag."
        )
    )
    parser.add_argument(
        "--model",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "models",
            "sac_cube_final_80",
        ),
    )
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--camera", type=int, default=2)
    parser.add_argument(
        "--calib",
        default=str(SCRIPT_DIR / "calibration.npz"),
        help="AprilTag camera calibration file (.npz)",
    )
    parser.add_argument(
        "--ref-tag-id",
        type=int,
        default=REFERENCE_TAG_ID,
        help="AprilTag ID of the fixed scene reference tag defining world origin.",
    )
    parser.add_argument(
        "--ref-tag-size",
        type=float,
        default=REFERENCE_TAG_SIZE,
        help="Physical size (meters) of the reference tag.",
    )
    parser.add_argument(
        "--target-face",
        type=int,
        default=None,
        choices=range(6),
        help="Digit as printed on the cube (0-5).",
    )
    parser.add_argument(
        "--goal-speed",
        type=int,
        default=GOAL_SPEED,
        help=(
            f"Servo goal speed integer in range 1-1023 (default {GOAL_SPEED})."
        ),
    )
    parser.add_argument(
        "--max-hold-frames",
        type=int,
        default=DEFAULT_MAX_HOLD_FRAMES,
        help=(
            "Max consecutive frames to hold the last-known pose (cube tag or"
            f" reference tag) before treating it as lost (default {DEFAULT_MAX_HOLD_FRAMES})."
        ),
    )
    parser.add_argument(
        "--test-vision",
        action="store_true",
        help="Run vision tracking test loop without connecting to hardware.",
    )
    parser.add_argument(
        "--no-sim-viewer",
        action="store_true",
        help=(
            "Disable the live MuJoCo viewer that mirrors real joint readings"
            " onto the sim model during deployment (open by default so you"
            " can visually compare finger motion against sim)."
        ),
    )
    args = parser.parse_args()

    REFERENCE_TAG_ID = args.ref_tag_id
    REFERENCE_TAG_SIZE = args.ref_tag_size

    # Configure camera to match realpose6dref settings
    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30.0)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera at index {args.camera}.")

    camera_calib = load_camera_calib(args.calib)
    tracker = ApriltagCubeTracker(camera_calib, FACE_NORMALS)
    cube_tracker = CubeQuatTracker(tracker, max_hold_frames=args.max_hold_frames)
    ref_tracker = ReferenceTagTracker(camera_calib, max_hold_frames=args.max_hold_frames)

    print(
        f"[INFO] World origin defined by reference tag ID {REFERENCE_TAG_ID}"
        f" (size {REFERENCE_TAG_SIZE * 100:.1f} cm)."
    )

    if args.test_vision:
        run_vision_validation(cube_tracker, ref_tracker, cap, camera_calib)
        cap.release()
        return

    model = SAC.load(args.model, device="cpu")
    run_cfg = dataclasses.replace(CFG, max_steps=CFG.max_steps * 10)

    hand = None
    sim_viewer = None
    kin = None
    try:
        hand = HandInterface(
            serial_port=args.port,
            calibration=load_calibration(),
            goal_speed=args.goal_speed,
        )
        kin = HandKinematics()

        if not args.no_sim_viewer:
            # Launched passively so it just mirrors state we push into
            # kin.data -- no physics runs in this viewer, it's a pure
            # visual readout of real servo positions via forward kinematics.
            sim_viewer = mujoco.viewer.launch_passive(kin.model, kin.data)
            print(
                "[INFO] Sim viewer open -- finger motion mirrors real servo"
                " readings each control tick. Close the viewer window or"
                " pass --no-sim-viewer to disable."
            )

        print("\n=== AmazeDex cube rotation (ref-tag world-frame deployment) ===")

        target_face = (
            DIGIT_TO_FACE_INDEX[args.target_face]
            if args.target_face is not None
            else None
        )

        while True:
            if target_face is None:
                target_face = prompt_target_face(None)
                if target_face is None:
                    break
            execute_target(
                model, hand, cube_tracker, ref_tracker, target_face, kin, cap,
                camera_calib, cfg=run_cfg, sim_viewer=sim_viewer,
            )
            target_face = None

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt caught via Ctrl+C.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if sim_viewer is not None:
            sim_viewer.close()
        if hand is not None:
            hand.close()


if __name__ == "__main__":
    main()