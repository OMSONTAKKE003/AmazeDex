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
GOAL_SPEED = 200  # Raw SCS0009 integer speed matching script convention (default 200)
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


def grab_latest_frame(cap: cv2.VideoCapture):
    """Flushes stale buffered frames so detection runs on real-time camera feeds."""
    for _ in range(2):
        cap.grab()
    return cap.read()


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
            nthreads=4,  # Parallelized detection
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )

    def update_quat(self, frame_bgr: np.ndarray):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray_undist = cv2.undistort(gray, self.calib.mtx, self.calib.dist)
        detections = self.detector.detect(
            gray_undist,
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
            nthreads=4,  # Parallelized detection
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
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray_undist = cv2.undistort(gray, self.calib.mtx, self.calib.dist)
        detections = self.detector.detect(
            gray_undist,
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
    """Lightweight forward-kinematics helper used to compute tip-to-cube
    features for the REAL-hardware observation. This model is only ever
    driven via mj_forward (no physics stepping) and is not shared with any
    viewer -- it exists purely to compute FK-derived features.
    """

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


class ActionMapper:
    DEPLOY_ACTION_LPF = 0.10
    DEPLOY_MAX_CTRL_RATE_FRAC = 0.10
    BLEND_STEPS = 60

    def __init__(self, kin: HandKinematics, cfg=CFG, action_lpf=None,
                 max_ctrl_rate_frac=None, blend_steps=None):
        self.kin = kin
        self.cfg = cfg
        self.action_lpf = self.DEPLOY_ACTION_LPF if action_lpf is None else action_lpf
        self.max_ctrl_rate_frac = (
            self.DEPLOY_MAX_CTRL_RATE_FRAC if max_ctrl_rate_frac is None else max_ctrl_rate_frac
        )
        self.blend_steps = self.BLEND_STEPS if blend_steps is None else blend_steps
        self.grasp_frac = None
        self.filtered_ctrl = None
        self.safe_frac = None
        self._step_count = 0

    def start(self, current_joint_pos_rad: np.ndarray) -> None:
        frac = np.clip(
            (current_joint_pos_rad - self.kin.ctrl_mid) / self.kin.ctrl_half,
            -1.0,
            1.0,
        )
        self.grasp_frac = frac.copy()
        self.filtered_ctrl = frac.copy()
        self.safe_frac = frac.copy()
        self._step_count = 0

    def action_to_ctrl(self, action: np.ndarray) -> np.ndarray:
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        target_frac = np.clip(
            self.filtered_ctrl + action * self.cfg.grasp_band_frac, -1.0, 1.0
        )
        lpf_ctrl = (
            self.action_lpf * target_frac
            + (1 - self.action_lpf) * self.filtered_ctrl
        )
        rate = self.max_ctrl_rate_frac
        step_delta = np.clip(lpf_ctrl - self.filtered_ctrl, -rate, rate)
        policy_frac = self.filtered_ctrl + step_delta

        progress = min(1.0, self._step_count / float(self.blend_steps))
        self._step_count += 1
        blended_frac = (1.0 - progress) * self.safe_frac + progress * policy_frac
        blended_frac = np.clip(blended_frac, -1.0, 1.0)

        self.filtered_ctrl = blended_frac
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
        try:
            action_func(sid, *args)
        except RuntimeError as e:
            print(
                f"[WARN] reset_to_home: could not {action_name} for servo {sid}: {e}"
            )

    def reset_to_home(self) -> None:
        print(
            "\n[INFO] Safe shutdown: odd IDs returning to -511, even IDs to 511..."
        )
        slow_speed = 1

        for sid in SERVO_IDS:
            self._safe_servo_write(
                self.controller.write_goal_speed,
                sid,
                slow_speed,
                action_name="set slow speed",
            )
            time.sleep(0.001)

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

        time.sleep(2.0)

        for sid in SERVO_IDS:
            self._safe_servo_write(
                self.controller.write_torque_enable,
                sid,
                0,
                action_name="disable torque",
            )
            time.sleep(0.0002)

    def close(self) -> None:
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
# Parallel sim evaluator
#
# Runs its own fully-stepped MuJoCo simulation (own model/data/actuators),
# driven by the same policy, in lockstep with the real-hardware rollout (or
# fully standalone in --sim-only mode). This is a genuine physics
# evaluation (mj_step every tick), not a passive mirror of real joint
# readings.
# ===========================================================================


class SimHandEvaluator:
    # Training's AmazeDexCubeEnv.step() calls do_simulation(ctrl, frame_skip=10)
    # -- 10 physics substeps per control step. Used as the default here so
    # deploy-time dynamics match training unless explicitly overridden.
    DEFAULT_FRAME_SKIP = 10

    def __init__(self, model_path: str = MODEL_PATH, cfg=CFG):
        if not os.path.exists(model_path):
            alt_path = SCRIPT_DIR / "resources" / "scene.xml"
            model_path = str(alt_path) if alt_path.exists() else model_path
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.cfg = cfg

        self.qposids = np.array([self.model.joint(n).qposadr[0] for n in JOINTS])
        self.qvelids = np.array([self.model.joint(n).dofadr[0] for n in JOINTS])
        self.cubeid = self.model.body("cube").id
        self.tip_sites = [self.model.site(n).id for n in TIP_SITES]
        actids = np.array([self.model.actuator(n).id for n in ACTUATORS])
        self.actids = actids
        lims = self.model.actuator_ctrlrange[actids]
        self.ctrl_lo, self.ctrl_hi = lims[:, 0], lims[:, 1]
        self.ctrl_mid = (self.ctrl_lo + self.ctrl_hi) / 2
        self.ctrl_half = (self.ctrl_hi - self.ctrl_lo) / 2

        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        cube_body_pos = np.array(self.model.body("cube").pos, dtype=np.float32)
        self.nominal_cube_pos = cube_body_pos + CUBE_LOCAL_CENTER

        self.mapper = None
        self.prev_action = np.zeros(8, dtype=np.float32)
        self.prev_joint_pos = None
        self.prev_quat = None

    def start(
        self,
        action_lpf: float = 0.10,
        max_ctrl_rate: float = 0.10,
        blend_steps: int = 60,
    ) -> None:
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.mapper = ActionMapper(
            self,
            self.cfg,
            action_lpf=action_lpf,
            max_ctrl_rate_frac=max_ctrl_rate,
            blend_steps=blend_steps,
        )
        joint_pos = self.data.qpos[self.qposids].copy()
        self.mapper.start(joint_pos)
        self.prev_joint_pos = joint_pos.copy()
        self.prev_action = np.zeros(8, dtype=np.float32)
        self.prev_quat = self._read_cube_quat()

    def _read_cube_quat(self) -> np.ndarray:
        wxyz = np.asarray(self.data.xquat[self.cubeid], dtype=np.float32)
        return wxyz.copy()

    def tip_to_cube(self, joint_pos_rad: np.ndarray) -> np.ndarray:
        cpos = self.nominal_cube_pos
        return np.array(
            [cpos - self.data.site_xpos[s] for s in self.tip_sites],
            dtype=np.float32,
        )

    def step(self, target_face: int):
        """Runs one control tick's worth of observation-building for the
        parallel sim rollout (physics stepping happens in apply_action)."""
        joint_pos = self.data.qpos[self.qposids].copy()
        joint_vel = self.data.qvel[self.qvelids].copy()

        tip_to_cube_raw = self.tip_to_cube(joint_pos)
        tip_to_cube_n = np.clip(tip_to_cube_raw / REACH_NORM, -3.0, 3.0).flatten()

        cube_quat = self._read_cube_quat()
        cube_angvel = cube_angvel_from_quats(self.prev_quat, cube_quat, CONTROL_DT)
        self.prev_quat = cube_quat.copy()

        obs = build_cube_obs(
            joint_pos,
            joint_vel,
            self.prev_action,
            target_face,
            self,
            cube_quat=cube_quat,
            cube_angvel_raw=cube_angvel,
            tip_to_cube_n=tip_to_cube_n,
        )

        return obs, cube_quat, cube_angvel

    def apply_action(self, action: np.ndarray, frame_skip: int | None = None) -> np.ndarray:
        """Applies the policy action and steps physics forward `frame_skip`
        substeps (defaults to DEFAULT_FRAME_SKIP=10, matching training's
        do_simulation(ctrl, frame_skip=10))."""
        n_substeps = self.DEFAULT_FRAME_SKIP if frame_skip is None else frame_skip
        target_ctrl = self.mapper.action_to_ctrl(action)
        self.data.ctrl[self.actids] = target_ctrl
        for _ in range(n_substeps):
            mujoco.mj_step(self.model, self.data)
        self.prev_action = action.copy()
        return target_ctrl


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
        ok, raw_frame = grab_latest_frame(cap)
        if not ok or raw_frame is None:
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
# Main control loop -- real hardware + parallel sim eval, driven together
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
    sim_eval: SimHandEvaluator | None = None,
    sim_viewer=None,
    action_lpf: float = 0.10,
    max_ctrl_rate: float = 0.10,
    blend_steps: int = 60,
    sim_frame_skip: int | None = None,
) -> bool:
    print(f"\n--- Target: face {FACE_INDEX_TO_DIGIT[target_face]} ---")

    mapper = ActionMapper(
        kin,
        cfg,
        action_lpf=action_lpf,
        max_ctrl_rate_frac=max_ctrl_rate,
        blend_steps=blend_steps,
    )
    joint_pos = hand.read_joint_positions()
    mapper.start(joint_pos)
    prev_joint_pos = joint_pos.copy()
    prev_action = np.zeros(8, dtype=np.float32)

    if sim_eval is not None:
        sim_eval.start(
            action_lpf=action_lpf,
            max_ctrl_rate=max_ctrl_rate,
            blend_steps=blend_steps,
        )

    identity_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    ok, first_frame = grab_latest_frame(cap)
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
    sim_hold = 0

    for step in range(cfg.max_steps):
        loop_start = time.perf_counter()

        # ---------------- REAL-HARDWARE tick ----------------
        joint_pos = hand.read_joint_positions()
        joint_vel = (joint_pos - prev_joint_pos) / CONTROL_DT

        tip_to_cube_raw = kin.tip_to_cube(joint_pos)
        tip_to_cube_n = np.clip(tip_to_cube_raw / REACH_NORM, -3.0, 3.0).flatten()

        ok, raw_frame = grab_latest_frame(cap)
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

        hand.send_ctrl(target_ctrl)

        prev_joint_pos = joint_pos.copy()
        prev_action = action.copy()

        theta = theta_rad(cube_quat, target_face)
        settled = (
            theta < cfg.success_theta_rad
            and np.linalg.norm(cube_angvel) < cfg.success_max_angvel
        )
        hold = hold + 1 if settled else 0

        # ---------------- PARALLEL SIM tick ----------------
        sim_theta = None
        sim_settled = False
        sim_cube_quat = None
        if sim_eval is not None:
            sim_obs, sim_cube_quat, sim_angvel = sim_eval.step(target_face)
            sim_action = predict_action(model, sim_obs)
            sim_eval.apply_action(sim_action, frame_skip=sim_frame_skip)

            sim_theta = theta_rad(sim_cube_quat, target_face)
            sim_settled = (
                sim_theta < cfg.success_theta_rad
                and np.linalg.norm(sim_angvel) < cfg.success_max_angvel
            )
            sim_hold = sim_hold + 1 if sim_settled else 0

            if sim_viewer is not None and sim_viewer.is_running():
                sim_viewer.sync()

        print(
            f"[Step {step:03d}] "
            f"REAL theta={np.degrees(theta):5.1f}deg hold={hold} | "
            + (
                f"SIM theta={np.degrees(sim_theta):5.1f}deg hold={sim_hold}"
                if sim_theta is not None
                else "SIM disabled"
            )
        )

        if raw_frame is not None:
            if cube_quat_cam is None:
                cube_status = "L"
            elif cube_fresh:
                cube_status = "f"
            else:
                cube_status = "h"
            ref_status = "f" if ref_fresh else ("h" if ref_rot is not None else "L")
            any_lost = cube_quat_cam is None or ref_rot is None

            current_digit = get_current_facing_digit(cube_quat)

            display = cv2.undistort(raw_frame, camera_calib.mtx, camera_calib.dist)

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

            if sim_theta is not None:
                cv2.putText(
                    display,
                    f"SIM theta={np.degrees(sim_theta):.1f}deg hold={sim_hold}",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 255),
                    2,
                )

            cv2.imshow("AmazeDex Cube Execution (REAL)", display)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                print("--- ABORTED (quit) ---")
                return False

        if sim_viewer is not None and not sim_viewer.is_running():
            print("--- ABORTED (sim viewer closed) ---")
            return False

        if hold >= cfg.success_hold_steps:
            print(
                f"--- REAL SUCCESS: face {FACE_INDEX_TO_DIGIT[target_face]} reached"
                f" and held ({step + 1} steps) ---"
            )
            if sim_eval is not None:
                status = "SUCCESS" if sim_hold >= cfg.success_hold_steps else "NOT YET SETTLED"
                print(f"--- SIM at same tick: {status} (sim_hold={sim_hold}) ---")
            return True

        elapsed = time.perf_counter() - loop_start
        if elapsed < CONTROL_DT:
            time.sleep(CONTROL_DT - elapsed)

    print(
        f"--- TIMEOUT: target face {FACE_INDEX_TO_DIGIT[target_face]} not"
        f" reached in {cfg.max_steps} steps ---"
    )
    return False


def execute_target_sim_only(
    model,
    sim_eval: SimHandEvaluator,
    target_face: int,
    cfg=CFG,
    sim_viewer=None,
    action_lpf: float = 0.10,
    max_ctrl_rate: float = 0.10,
    blend_steps: int = 60,
    sim_frame_skip: int | None = None,
) -> bool:
    """Runs the policy against ONLY the parallel MuJoCo sim -- no camera,
    no AprilTags, no servo hardware. Useful for quickly sanity-checking a
    checkpoint or the deploy-time action mapping before going anywhere near
    the real hand."""
    print(f"\n--- [SIM-ONLY] Target: face {FACE_INDEX_TO_DIGIT[target_face]} ---")

    sim_eval.start(
        action_lpf=action_lpf,
        max_ctrl_rate=max_ctrl_rate,
        blend_steps=blend_steps,
    )
    hold = 0

    for step in range(cfg.max_steps):
        loop_start = time.perf_counter()

        obs, cube_quat, cube_angvel = sim_eval.step(target_face)
        action = predict_action(model, obs)
        sim_eval.apply_action(action, frame_skip=sim_frame_skip)

        theta = theta_rad(cube_quat, target_face)
        settled = (
            theta < cfg.success_theta_rad
            and np.linalg.norm(cube_angvel) < cfg.success_max_angvel
        )
        hold = hold + 1 if settled else 0

        if sim_viewer is not None and sim_viewer.is_running():
            sim_viewer.sync()

        print(f"[SIM Step {step:03d}] theta={np.degrees(theta):5.1f}deg hold={hold}")

        if sim_viewer is not None and not sim_viewer.is_running():
            print("--- ABORTED (sim viewer closed) ---")
            return False

        if hold >= cfg.success_hold_steps:
            print(
                f"--- SIM SUCCESS: face {FACE_INDEX_TO_DIGIT[target_face]} reached"
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
            " Runs a genuine parallel MuJoCo sim evaluation of the same"
            " policy alongside the real-hardware rollout for side-by-side"
            " comparison (or standalone with --sim-only)."
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
        "--action-lpf",
        type=float,
        default=0.10,
        help="Action low-pass filter coefficient (default 0.10, raise toward training's 0.75).",
    )
    parser.add_argument(
        "--max-ctrl-rate",
        type=float,
        default=0.10,
        help="Max control rate step delta (default 0.10, raise toward training's 0.70).",
    )
    parser.add_argument(
        "--blend-steps",
        type=int,
        default=60,
        help="Number of initial steps to ramp in policy targets from safe initial pose (default 60).",
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
        "--sim-frame-skip",
        type=int,
        default=None,
        help=(
            "Physics substeps per control tick for the parallel/sim-only"
            f" MuJoCo rollout (default {SimHandEvaluator.DEFAULT_FRAME_SKIP},"
            " matching training's do_simulation frame_skip)."
        ),
    )
    parser.add_argument(
        "--test-vision",
        action="store_true",
        help="Run vision tracking test loop without connecting to hardware.",
    )
    parser.add_argument(
        "--no-sim-eval",
        action="store_true",
        help=(
            "Disable the parallel MuJoCo sim evaluation. By default a fully"
            " stepped sim rollout of the same policy runs alongside the"
            " real-hardware rollout each tick, shown in its own viewer."
        ),
    )
    parser.add_argument(
        "--sim-only",
        action="store_true",
        help=(
            "Run ONLY the MuJoCo sim rollout -- no camera, no AprilTags, no"
            " servo/hardware connection at all. Useful for a quick sanity"
            " check of a checkpoint before touching the real hand."
        ),
    )
    args = parser.parse_args()

    REFERENCE_TAG_ID = args.ref_tag_id
    REFERENCE_TAG_SIZE = args.ref_tag_size

    if args.sim_only:
        model = SAC.load(args.model, device="cpu")
        run_cfg = dataclasses.replace(CFG, max_steps=CFG.max_steps * 10)
        sim_eval = SimHandEvaluator(cfg=run_cfg)
        sim_viewer = mujoco.viewer.launch_passive(sim_eval.model, sim_eval.data)
        print("\n=== AmazeDex cube rotation (SIM-ONLY) ===")

        target_face = (
            DIGIT_TO_FACE_INDEX[args.target_face]
            if args.target_face is not None
            else None
        )
        try:
            while True:
                if target_face is None:
                    target_face = prompt_target_face(None)
                    if target_face is None:
                        break
                execute_target_sim_only(
                    model,
                    sim_eval,
                    target_face,
                    cfg=run_cfg,
                    sim_viewer=sim_viewer,
                    action_lpf=args.action_lpf,
                    max_ctrl_rate=args.max_ctrl_rate,
                    blend_steps=args.blend_steps,
                    sim_frame_skip=args.sim_frame_skip,
                )
                target_face = None
        except KeyboardInterrupt:
            print("\n[INFO] KeyboardInterrupt caught via Ctrl+C.")
        finally:
            if sim_viewer is not None:
                sim_viewer.close()
        return

    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30.0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Keep camera buffer at size 1 to prevent frame lag

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
    sim_eval = None
    try:
        hand = HandInterface(
            serial_port=args.port,
            calibration=load_calibration(),
            goal_speed=args.goal_speed,
        )
        kin = HandKinematics()

        if not args.no_sim_eval:
            sim_eval = SimHandEvaluator(cfg=run_cfg)
            sim_viewer = mujoco.viewer.launch_passive(sim_eval.model, sim_eval.data)
            print(
                "[INFO] Parallel sim eval running -- a fully stepped MuJoCo"
                " rollout of the same policy runs each tick alongside the"
                " real hardware, independently, for side-by-side comparison."
                " Close the viewer window or pass --no-sim-eval to disable."
            )

        print("\n=== AmazeDex cube rotation (ref-tag world-frame deployment"
              " + parallel sim eval) ===")

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
                model,
                hand,
                cube_tracker,
                ref_tracker,
                target_face,
                kin,
                cap,
                camera_calib,
                cfg=run_cfg,
                sim_eval=sim_eval,
                sim_viewer=sim_viewer,
                action_lpf=args.action_lpf,
                max_ctrl_rate=args.max_ctrl_rate,
                blend_steps=args.blend_steps,
                sim_frame_skip=args.sim_frame_skip,
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