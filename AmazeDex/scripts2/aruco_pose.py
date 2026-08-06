"""ArUco/AprilTag-based cube pose estimation for AmazeDex sim2real.

The physical cube has six tag36h11 markers (ids 0-5), one per face. This
module detects whichever tag the camera can see, recovers its pose, and
composes it with (a) the tag's known orientation relative to the cube body
frame - copied from the tag geoms in scene.xml - and (b) a calibrated
camera-to-world rotation, to produce the cube orientation in the same
Z-up world frame `AmazeDexCubeEnv` uses internally.

Before trusting this on real hardware:
  1. Edit TAG_SIZE_M_DEFAULT to your printed tag's edge length (meters).
  2. Run calibrate_camera_intrinsics() once for your webcam (or supply your
     own camera_matrix/dist_coeffs) and save it into camera_calibration.json.
  3. Run calibrate_extrinsic() once with the cube held in a known
     orientation to correct for how the camera is actually mounted.
  4. TAG_TO_CUBE_QUAT below matches scene.xml's simulated tag placement.
     If your physical cube's stickers are arranged differently, re-derive
     this table - getting the chirality wrong here will make the policy
     think the cube is oriented incorrectly.
"""
from __future__ import annotations

import json
import os

import cv2
import cv2.aruco as aruco
import numpy as np

TAG_DICTIONARY = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
TAG_SIZE_M_DEFAULT = 0.010  # EDIT ME: physical printed tag edge length, meters
CALIBRATION_PATH = "camera_calibration.json"

# wxyz quats, copied verbatim from the tag geoms in scene.xml
TAG_TO_CUBE_QUAT = {
    0: (1.0, 0.0, 0.0, 0.0),
    1: (0.7071068, 0.0, 0.7071068, 0.0),
    2: (0.7071068, 0.7071068, 0.0, 0.0),
    3: (0.7071068, 0.0, -0.7071068, 0.0),
    4: (0.7071068, -0.7071068, 0.0, 0.0),
    5: (1.0, 0.0, 0.0, 0.0),
}


def quat_to_rotmat(q) -> np.ndarray:
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - w * z), s * (x * z + w * y)],
        [s * (x * y + w * z), 1 - s * (x * x + z * z), s * (y * z - w * x)],
        [s * (x * z - w * y), s * (y * z + w * x), 1 - s * (x * x + y * y)],
    ])


TAG_TO_CUBE_R = {tag_id: quat_to_rotmat(q) for tag_id, q in TAG_TO_CUBE_QUAT.items()}


def load_camera_calibration(path: str = CALIBRATION_PATH) -> dict:
    if not os.path.exists(path):
        print(
            f"[WARN] {path} not found - using rough default intrinsics and an "
            "uncalibrated (identity) camera mount. Real pose estimates will be "
            "unreliable until you calibrate (see aruco_pose.py docstring)."
        )
        return {
            "camera_matrix": [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]],
            "dist_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
            "cam_extrinsic_quat": [1.0, 0.0, 0.0, 0.0],
            "tag_size_m": TAG_SIZE_M_DEFAULT,
        }
    with open(path) as f:
        return json.load(f)


class CubePoseTracker:
    """Tracks cube orientation (world frame, Z-up) from webcam AprilTag detections."""

    def __init__(self, calibration: dict | None = None, smoothing: float = 0.5):
        calib = calibration or load_camera_calibration()
        self.camera_matrix = np.array(calib["camera_matrix"], dtype=np.float64)
        self.dist_coeffs = np.array(calib["dist_coeffs"], dtype=np.float64)
        self.cam_extrinsic_R = quat_to_rotmat(calib["cam_extrinsic_quat"])
        self.tag_size_m = float(calib.get("tag_size_m", TAG_SIZE_M_DEFAULT))
        self.smoothing = smoothing
        self._last_R = np.eye(3)
        self._detector = None
        if hasattr(aruco, "ArucoDetector"):
            self._detector = aruco.ArucoDetector(TAG_DICTIONARY, aruco.DetectorParameters())

    @property
    def last_R(self) -> np.ndarray:
        return self._last_R

    def _detect(self, gray: np.ndarray):
        if self._detector is not None:
            corners, ids, _ = self._detector.detectMarkers(gray)
        else:  # older opencv-contrib fallback
            corners, ids, _ = aruco.detectMarkers(gray, TAG_DICTIONARY)
        return corners, ids

    def update(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Feed one camera frame; returns the current best estimate of R_cube_in_world (3x3)."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        corners, ids = self._detect(gray)
        if ids is None or len(ids) == 0:
            return self._last_R

        # Prefer the largest (closest, most reliably detected) known tag in view.
        best_idx, best_area = None, -1.0
        for i, tag_id in enumerate(ids.flatten()):
            if int(tag_id) not in TAG_TO_CUBE_R:
                continue
            area = cv2.contourArea(corners[i].reshape(-1, 2).astype(np.float32))
            if area > best_area:
                best_area, best_idx = area, i
        if best_idx is None:
            return self._last_R

        tag_id = int(ids.flatten()[best_idx])
        half = self.tag_size_m / 2.0
        obj_pts = np.array(
            [[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]],
            dtype=np.float64,
        )
        img_pts = corners[best_idx].reshape(-1, 2).astype(np.float64)

        ok, rvec, _ = cv2.solvePnP(obj_pts, img_pts, self.camera_matrix, self.dist_coeffs)
        if not ok:
            return self._last_R

        R_tag_in_cam, _ = cv2.Rodrigues(rvec)
        R_tag_to_cube = TAG_TO_CUBE_R[tag_id]
        R_cube_in_cam = R_tag_in_cam @ R_tag_to_cube.T
        R_cube_in_world = self.cam_extrinsic_R @ R_cube_in_cam

        # Exponential smoothing to damp jitter, then re-orthonormalize (a blend
        # of two rotation matrices isn't itself a valid rotation matrix).
        blended = self.smoothing * self._last_R + (1 - self.smoothing) * R_cube_in_world
        u, _, vt = np.linalg.svd(blended)
        self._last_R = u @ vt
        return self._last_R


def calibrate_extrinsic(tracker: CubePoseTracker, frame_bgr: np.ndarray, known_face_up: int):
    """One-shot calibration helper.

    Hold the cube with `known_face_up` (0-5, matching FACENORMALS in
    amazedex_cube_env.py) pointing straight up, then call this once. It
    returns a wxyz quat: save it into camera_calibration.json's
    "cam_extrinsic_quat" field so future estimates account for how your
    camera is actually mounted relative to gravity.
    """
    from amazedex_cube_env import FACENORMALS

    raw_R = tracker.update(frame_bgr)  # uncorrected estimate (cam_extrinsic assumed identity)
    current = raw_R @ FACENORMALS[known_face_up]
    target = np.array([0.0, 0.0, 1.0])

    axis = np.cross(current, target)
    s = np.linalg.norm(axis)
    c = float(np.dot(current, target))
    if s < 1e-8:
        return (1.0, 0.0, 0.0, 0.0) if c > 0 else (0.0, 1.0, 0.0, 0.0)
    axis = axis / s
    angle = np.arctan2(s, c)
    w = np.cos(angle / 2)
    xyz = axis * np.sin(angle / 2)
    return (float(w), float(xyz[0]), float(xyz[1]), float(xyz[2]))
