

from __future__ import annotations

import argparse
import os
import re
import sys
import time
import traceback
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models



DEFAULT_MODEL_PATH = r"C:\Users\luvja\Desktop\updatedwithstand\AmazeDex\face_cnn.pt"
DEFAULT_SCENE_PATH = r"C:\Users\luvja\Desktop\updatedwithstand\AmazeDex\resources\scene.xml"

IMG_SIZE    = 224    
CONF_THRESH = 0.45    
ENTROPY_MAX = 1.1     
VOTE_WINDOW = 4        
VOTE_MIN    = 3     

TAG_SIZE_M = 0.010  

CUBE_CENTER_LOCAL = np.array([0.0, -0.035, 0.0])   
CUBE_HALF_SIZE = 0.025                             

ARUCO_DICT_NAME = "DICT_APRILTAG_36H11"

DEFAULT_CAM_UP_IN_CAM_FRAME = np.array([0.0, -1.0, 0.0])


@dataclass
class TagFace:
    tag_id: int
    name: str
    axis: int              
    sign: int            
    pos: np.ndarray    
    quat_wxyz: np.ndarray    


TAG_FACE_INFO: Dict[int, TagFace] = {
    5: TagFace(5, "top(+Z)",    axis=2, sign=+1,
               pos=np.array([-0.02, -0.035, 0.025]),
               quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0])),
    0: TagFace(0, "bottom(-Z)", axis=2, sign=-1,
               pos=np.array([-0.02, -0.035, -0.025]),
               quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0])),
    1: TagFace(1, "right(+X)",  axis=0, sign=+1,
               pos=np.array([0.025, -0.055, 0.0]),
               quat_wxyz=np.array([0.7071068, 0.0, 0.7071068, 0.0])),
    2: TagFace(2, "left(-X)",   axis=0, sign=-1,
               pos=np.array([-0.025, -0.015, 0.0]),
               quat_wxyz=np.array([0.7071068, 0.0, -0.7071068, 0.0])),
    3: TagFace(3, "front(+Y)",  axis=1, sign=+1,
               pos=np.array([0.02, -0.01, 0.0]),
               quat_wxyz=np.array([0.7071068, -0.7071068, 0.0, 0.0])),
    4: TagFace(4, "back(-Y)",   axis=1, sign=-1,
               pos=np.array([-0.02, -0.06, 0.0]),
               quat_wxyz=np.array([0.7071068, 0.7071068, 0.0, 0.0])),
}


_GEOM_RE = re.compile(
    r'<geom\s+name="tag_number_(\d+)"[^>]*?\bpos="([^"]+)"'
    r'(?:[^>]*?\bquat="([^"]+)")?[^>]*?/?>',
    re.DOTALL,
)


def verify_tag_geometry_against_scene(scene_path: str, pos_tol: float = 1e-4,
                                       quat_tol: float = 1e-3) -> bool:

    print("[SELF-CHECK] Verifying TAG_FACE_INFO against scene.xml ...")

    if not scene_path or not os.path.exists(scene_path):
        print(f"[SELF-CHECK] scene.xml not found at '{scene_path}' -- skipping "
              f"geometry verification (expected in --mode real without --scene).")
        return True

    try:
        with open(scene_path, "r", encoding="utf-8") as f:
            xml_text = f.read()
    except Exception as e:
        print(f"[SELF-CHECK][WARN] Could not read scene.xml: {e}. Skipping check.")
        return True

    found: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for m in _GEOM_RE.finditer(xml_text):
        tag_id = int(m.group(1))
        pos = np.array([float(x) for x in m.group(2).split()])
        quat_str = m.group(3)
        quat = np.array([float(x) for x in quat_str.split()]) if quat_str else np.array([1.0, 0.0, 0.0, 0.0])
        found[tag_id] = (pos, quat)

    all_ok = True
    for tag_id, face in sorted(TAG_FACE_INFO.items()):
        if tag_id not in found:
            print(f"[SELF-CHECK][WARN] tag_number_{tag_id} not found in scene.xml "
                  f"(expected face '{face.name}'). Skipping this tag's check.")
            continue
        scene_pos, scene_quat = found[tag_id]
        pos_ok = np.allclose(scene_pos, face.pos, atol=pos_tol)
        # quat and -quat represent the same rotation
        quat_ok = (np.allclose(scene_quat, face.quat_wxyz, atol=quat_tol) or
                   np.allclose(scene_quat, -face.quat_wxyz, atol=quat_tol))
        if pos_ok and quat_ok:
            print(f"[SELF-CHECK]   OK    tag{tag_id} ({face.name}): pos/quat match scene.xml")
        else:
            all_ok = False
            print(f"[SELF-CHECK]   MISMATCH tag{tag_id} ({face.name}):")
            print(f"                script pos={face.pos}  quat={face.quat_wxyz}")
            print(f"                scene  pos={scene_pos}  quat={scene_quat}")

    if all_ok:
        print("[SELF-CHECK] All tag geometry matches scene.xml. Proceeding.")
    else:
        print("[SELF-CHECK][WARNING] One or more tags in scene.xml no longer match "
              "TAG_FACE_INFO in this script. Cube pose / face assignment WILL BE "
              "WRONG for those tags until you update TAG_FACE_INFO to match. "
              "Continuing anyway, but treat pose output with suspicion.")
    return all_ok

=

def quat_wxyz_to_R(q: np.ndarray) -> np.ndarray:
    q = q / (np.linalg.norm(q) + 1e-12)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


def R_to_euler_xyz_deg(R: np.ndarray) -> Tuple[float, float, float]:
    sy = -R[2, 0]
    sy = np.clip(sy, -1.0, 1.0)
    pitch = np.arcsin(sy)
    if abs(np.cos(pitch)) > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        yaw = 0.0
    return tuple(np.degrees([roll, pitch, yaw]))


def R_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / (np.linalg.norm(q) + 1e-12)


def invert_transform(R: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    R_inv = R.T
    t_inv = -R_inv @ t
    return R_inv, t_inv


def compose_transform(R1: np.ndarray, t1: np.ndarray,
                       R2: np.ndarray, t2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Returns T1 . T2 (apply T2 first, then T1)."""
    R = R1 @ R2
    t = R1 @ t2 + t1
    return R, t


def average_rotations_weighted(rots: List[np.ndarray], weights: List[float]) -> np.ndarray:
    """Chordal L2 mean of rotation matrices, projected back onto SO(3) via SVD."""
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / (weights.sum() + 1e-12)
    A = np.zeros((3, 3))
    for R, w in zip(rots, weights):
        A += w * R
    U, _, Vt = np.linalg.svd(A)
    R_avg = U @ Vt
    if np.linalg.det(R_avg) < 0:
        U[:, -1] *= -1
        R_avg = U @ Vt
    return R_avg


def load_model(model_path: str, device: torch.device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"CNN checkpoint not found at: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    classes = [str(c) for c in checkpoint["classes"]]

    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, len(classes))

    raw_state = checkpoint["model_state"]
    clean_state = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in raw_state.items()}
    model.load_state_dict(clean_state, strict=True)
    model.to(device)
    model.eval()
    return model, classes


CNN_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def classify_face_crop(model, classes, device, bgr_crop: np.ndarray):
    """Runs the CNN on a rectified BGR face crop. Returns (pred_class, conf, entropy, probs)."""
    rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
    tensor = CNN_TRANSFORM(rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)[0]
        conf, pred_idx = torch.max(probs, dim=0)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()
    return classes[pred_idx.item()], conf.item(), entropy, probs.cpu().numpy()


class TagDetector:
    def __init__(self, dict_name: str = ARUCO_DICT_NAME):
        if not hasattr(cv2, "aruco"):
            raise RuntimeError(
                "cv2.aruco not found. Install opencv-contrib-python:\n"
                "    pip install opencv-contrib-python"
            )
        dict_id = getattr(cv2.aruco, dict_name)
        self.dictionary = cv2.aruco.getPredefinedDictionary(dict_id)

        self._use_new_api = hasattr(cv2.aruco, "ArucoDetector")
        if self._use_new_api:
            params = cv2.aruco.DetectorParameters()
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            self.detector = cv2.aruco.ArucoDetector(self.dictionary, params)
        else:
            self.params = cv2.aruco.DetectorParameters_create()
            self.params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    def detect(self, gray: np.ndarray):
        """Returns (corners_list, ids_array or None). Never raises."""
        try:
            if self._use_new_api:
                corners, ids, _ = self.detector.detectMarkers(gray)
            else:
                corners, ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self.params)
            return corners, ids
        except Exception as e:
            print(f"[WARN] Tag detection failed this frame: {e}")
            return [], None



class CameraIntrinsics:
    def __init__(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray, width: int, height: int):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.width = width
        self.height = height


class RealCameraSource:


    def __init__(self, camera_index: int, calib_path: Optional[str] = None,
                 width: int = 1280, height: int = 720):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self._open_capture()

        if calib_path and os.path.exists(calib_path):
            data = np.load(calib_path)
            camera_matrix = data["camera_matrix"]
            dist_coeffs = data["dist_coeffs"]
            print(f"[INFO] Loaded real camera calibration from {calib_path}")
        else:
            if calib_path:
                print(f"[WARN] Calibration file not found at {calib_path}; falling back to a rough "
                      f"default estimate. Run an OpenCV chessboard calibration for accurate metric "
                      f"pose (`cv2.calibrateCamera`), then pass --calib your_file.npz")
            fx = fy = self.actual_w / (2 * np.tan(np.radians(70) / 2))
            camera_matrix = np.array([[fx, 0, self.actual_w / 2],
                                       [0, fy, self.actual_h / 2],
                                       [0, 0, 1]], dtype=np.float64)
            dist_coeffs = np.zeros(5)

        self.intr = CameraIntrinsics(camera_matrix, dist_coeffs, self.actual_w, self.actual_h)
        self.cam_up_in_cam_frame = DEFAULT_CAM_UP_IN_CAM_FRAME.copy()

    def _open_capture(self):
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open webcam at index {self.camera_index}.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.width
        self.actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.height

    def read(self) -> Optional[np.ndarray]:
        ret, frame = self.cap.read()
        if not ret or frame is None:
            # transient camera hiccup -- try one reconnect before giving up on this frame
            print("[WARN] Dropped frame from webcam; attempting reconnect...")
            try:
                self.cap.release()
                time.sleep(0.3)
                self._open_capture()
                ret, frame = self.cap.read()
            except Exception as e:
                print(f"[WARN] Reconnect failed: {e}")
                return None
            if not ret or frame is None:
                return None
        return frame

    def close(self):
        self.cap.release()


class MuJoCoCameraSource:

    def __init__(self, scene_path: str, width: int = 960, height: int = 720,
                 camera_name: str = "tracking_camera", step_physics: bool = True):
        try:
            import mujoco
        except ImportError as e:
            raise RuntimeError(
                "mujoco python package not found. Install with:\n    pip install mujoco"
            ) from e
        self._mujoco = mujoco

        if not os.path.exists(scene_path):
            raise FileNotFoundError(f"MuJoCo scene not found at: {scene_path}")

        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.step_physics = step_physics

        self.cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if self.cam_id < 0:
            raise RuntimeError(f"Camera '{camera_name}' not found in {scene_path}")
        self.camera_name = camera_name

        fovy_deg = float(self.model.cam_fovy[self.cam_id])
        fy = height / (2 * np.tan(np.radians(fovy_deg) / 2))
        fx = fy  # square pixels, standard MuJoCo assumption
        camera_matrix = np.array([[fx, 0, width / 2],
                                   [0, fy, height / 2],
                                   [0, 0, 1]], dtype=np.float64)
        dist_coeffs = np.zeros(5)  # MuJoCo's pinhole render has no lens distortion
        self.intr = CameraIntrinsics(camera_matrix, dist_coeffs, width, height)

        mujoco.mj_forward(self.model, self.data)

    def read(self) -> Optional[np.ndarray]:
        m, d = self.model, self.data
        if self.step_physics:
            self._mujoco.mj_step(m, d)
        else:
            self._mujoco.mj_forward(m, d)
        self.renderer.update_scene(d, camera=self.camera_name)
        rgb = self.renderer.render()
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def world_up_in_camera_frame(self) -> np.ndarray:
        """Exact world-up direction expressed in the camera frame, computed
        live from MuJoCo's own camera pose (used for the die-value estimate)."""
        R_world_cam = self.data.cam_xmat[self.cam_id].reshape(3, 3)  # cols = cam axes in world
        R_cam_world = R_world_cam.T
        up_world = np.array([0.0, 0.0, 1.0])
        v = R_cam_world @ up_world
        return v / (np.linalg.norm(v) + 1e-12)

    def reset(self):
        self._mujoco.mj_resetData(self.model, self.data)
        self._mujoco.mj_forward(self.model, self.data)

    def close(self):
        pass



def _tag_object_points(tag_size: float) -> np.ndarray:
    s = tag_size / 2.0
    return np.array([
        [-s,  s, 0],
        [ s,  s, 0],
        [ s, -s, 0],
        [-s, -s, 0],
    ], dtype=np.float64)


@dataclass
class CubePoseEstimate:
    R: np.ndarray            
    t: np.ndarray           
    n_tags_used: int
    per_tag_area: Dict[int, float] = field(default_factory=dict)


def estimate_cube_pose(corners_list, ids, intr: CameraIntrinsics,
                        tag_size: float) -> Optional[CubePoseEstimate]:
    if ids is None or len(ids) == 0:
        return None

    obj_pts = _tag_object_points(tag_size)
    cand_R, cand_t, cand_w = [], [], []
    per_tag_area = {}

    for corners, tag_id in zip(corners_list, ids.flatten()):
        tag_id = int(tag_id)
        if tag_id not in TAG_FACE_INFO:
            continue  # unknown tag id -> ignore, don't crash

        img_pts = corners.reshape(4, 2).astype(np.float64)

        try:
            ok, rvec, tvec = cv2.solvePnP(
                obj_pts, img_pts, intr.camera_matrix, intr.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
        except Exception:
            ok = False

        if not ok:
            continue

        R_cam_tag, _ = cv2.Rodrigues(rvec)
        t_cam_tag = tvec.reshape(3)

        face = TAG_FACE_INFO[tag_id]
        R_cube_tag = quat_wxyz_to_R(face.quat_wxyz)
        t_cube_tag = face.pos
        R_tag_cube, t_tag_cube = invert_transform(R_cube_tag, t_cube_tag)

        R_cam_cube, t_cam_cube = compose_transform(R_cam_tag, t_cam_tag, R_tag_cube, t_tag_cube)

        area = cv2.contourArea(img_pts.astype(np.float32))
        if area <= 1.0:
            continue

        cand_R.append(R_cam_cube)
        cand_t.append(t_cam_cube)
        cand_w.append(area)
        per_tag_area[tag_id] = area

    if not cand_R:
        return None

    weights = np.array(cand_w, dtype=np.float64)
    weights = weights / weights.sum()
    t_fused = np.sum([w * t for w, t in zip(weights, cand_t)], axis=0)
    R_fused = average_rotations_weighted(cand_R, cand_w)

    return CubePoseEstimate(R=R_fused, t=t_fused, n_tags_used=len(cand_R), per_tag_area=per_tag_area)


def face_corners_local(axis: int, sign: int) -> np.ndarray:
 
    n = axis
    u_axis = (n + 1) % 3
    v_axis = (n + 2) % 3
    h = CUBE_HALF_SIZE

    def make(u_val, v_val):
        p = CUBE_CENTER_LOCAL.copy()
        p[n] += sign * h
        p[u_axis] += u_val
        p[v_axis] += sign * v_val   
        return p

    return np.array([
        make(-h, -h),  # BL
        make(h, -h),   # BR
        make(h, h),    # TR
        make(-h, h),   # TL
    ], dtype=np.float64)


def project_points(pts_local: np.ndarray, R_cam_cube: np.ndarray, t_cam_cube: np.ndarray,
                    intr: CameraIntrinsics) -> np.ndarray:
    pts_cam = (R_cam_cube @ pts_local.T).T + t_cam_cube
    rvec, _ = cv2.Rodrigues(np.eye(3))
    img_pts, _ = cv2.projectPoints(pts_cam, rvec, np.zeros(3), intr.camera_matrix, intr.dist_coeffs)
    return img_pts.reshape(-1, 2)


def warp_face_crop(frame: np.ndarray, img_quad: np.ndarray, out_size: int) -> Optional[np.ndarray]:
    h, w = frame.shape[:2]
    if not np.all(np.isfinite(img_quad)):
        return None
    area = cv2.contourArea(img_quad.astype(np.float32))
    if area < 25 or area > (w * h * 4):
        return None

    dst = np.array([[0, out_size - 1], [out_size - 1, out_size - 1],
                     [out_size - 1, 0], [0, 0]], dtype=np.float32)
    try:
        M = cv2.getPerspectiveTransform(img_quad.astype(np.float32), dst)
        crop = cv2.warpPerspective(frame, M, (out_size, out_size), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE)
    except Exception:
        return None
    return crop

class FaceMemory:
    """Per-tag temporal smoothing of CNN number predictions, keyed by tag id
    (since each physical face has a fixed, unchanging number)."""

    def __init__(self, quiet: bool = False):
        self.buffers: Dict[int, deque] = {tid: deque(maxlen=VOTE_WINDOW) for tid in TAG_FACE_INFO}
        self.locked: Dict[int, Optional[str]] = {tid: None for tid in TAG_FACE_INFO}
        self.quiet = quiet

    def update(self, tag_id: int, pred_class: Optional[str], conf: float = 0.0):
        buf = self.buffers[tag_id]
        buf.append(pred_class if pred_class is not None else "None")
        counts = Counter(buf)
        most_common, count = counts.most_common(1)[0]
        prev = self.locked[tag_id]
        if most_common != "None" and count >= VOTE_MIN:
            self.locked[tag_id] = most_common
        elif most_common == "None" and count >= VOTE_MIN:
            self.locked[tag_id] = None

        if not self.quiet and self.locked[tag_id] != prev and self.locked[tag_id] is not None:
            face = TAG_FACE_INFO[tag_id]
            print(f"[LOCK] tag{tag_id} {face.name:11s} -> number {self.locked[tag_id]} "
                  f"(last conf {conf*100:5.1f}%)")


class DetectionAnnouncer:
  

    def __init__(self):
        self._last_up: Optional[Tuple[int, Optional[str]]] = None
        self._last_front: Optional[Tuple[int, Optional[str]]] = None

    def announce(self, label: str, tag_id: Optional[int], memory: FaceMemory,
                 conf: float = 0.0):
        attr = "_last_up" if label == "UP" else "_last_front"
        prev = getattr(self, attr)
        if tag_id is None:
            setattr(self, attr, None)
            return
        number = memory.locked[tag_id]
        current = (tag_id, number)
        if current != prev and number is not None:
            face = TAG_FACE_INFO[tag_id]
            print(f"[DETECT] {label:5s} face -> tag{tag_id} {face.name:11s} = {number}"
                  f"  (conf {conf*100:5.1f}%)")
        setattr(self, attr, current)


def determine_up_face(pose: CubePoseEstimate, up_in_cam: np.ndarray) -> int:
    """Which of the 6 canonical faces has its outward normal most aligned
    with the given 'up' direction (expressed in camera frame)."""
    best_tid, best_dot = None, -2.0
    for tid, face in TAG_FACE_INFO.items():
        normal_local = np.zeros(3)
        normal_local[face.axis] = face.sign
        normal_cam = pose.R @ normal_local
        d = float(np.dot(normal_cam, up_in_cam))
        if d > best_dot:
            best_dot = d
            best_tid = tid
    return best_tid


def determine_front_face(pose: CubePoseEstimate) -> int:
    cam_forward_from_face = np.array([0.0, 0.0, -1.0])
    best_tid, best_dot = None, -2.0
    for tid, face in TAG_FACE_INFO.items():
        normal_local = np.zeros(3)
        normal_local[face.axis] = face.sign
        normal_cam = pose.R @ normal_local
        d = float(np.dot(normal_cam, cam_forward_from_face))
        if d > best_dot:
            best_dot = d
            best_tid = tid
    return best_tid



def draw_hud(pose: Optional[CubePoseEstimate], face_conf: Dict[int, Tuple[Optional[str], float]],
             memory: FaceMemory, up_face_id: Optional[int], front_face_id: Optional[int],
             fps: float) -> np.ndarray:
    hud_w, hud_h = 480, 460
    hud = np.zeros((hud_h, hud_w, 3), dtype=np.uint8)
    y = 30
    line_h = 22

    def put(text, color=(255, 255, 255), scale=0.55, dy=line_h):
        nonlocal y
        cv2.putText(hud, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        y += dy

    put("CUBE POSE + TAG/CNN NUMBER FUSION", (0, 255, 255), 0.6, 30)
    put(f"FPS: {fps:.1f}")

    if pose is None:
        put("No tags visible -- cube not localized.", (0, 0, 255))
    else:
        tx, ty, tz = pose.t
        put(f"Position (cam frame, m): x={tx:+.3f} y={ty:+.3f} z={tz:+.3f}", (0, 255, 0))
        roll, pitch, yaw = R_to_euler_xyz_deg(pose.R)
        put(f"Orientation (deg): roll={roll:+.1f} pitch={pitch:+.1f} yaw={yaw:+.1f}", (0, 255, 0))
        qw, qx, qy, qz = R_to_quat_wxyz(pose.R)
        put(f"Quat (wxyz): {qw:+.3f} {qx:+.3f} {qy:+.3f} {qz:+.3f}")
        put(f"Tags used for fused pose: {pose.n_tags_used}/6")

    y += 6
    put("Per-face tag -> number (locked) -> live conf:", (255, 255, 0))
    for tid in sorted(TAG_FACE_INFO):
        face = TAG_FACE_INFO[tid]
        locked = memory.locked[tid]
        pred, conf = face_conf.get(tid, (None, 0.0))
        visible = tid in face_conf
        if not visible:
            color = (110, 110, 110)
            txt = f"  tag{tid} {face.name:11s} not visible   locked={locked}"
        else:
            color = (0, 255, 0) if locked is not None else (0, 165, 255)
            txt = f"  tag{tid} {face.name:11s} pred={pred!s:>4} conf={conf*100:5.1f}%  locked={locked}"
        put(txt, color, 0.48, 20)

    y += 6
    if up_face_id is not None:
        put(f"UP face:    tag{up_face_id} ({TAG_FACE_INFO[up_face_id].name}) "
            f"= {memory.locked.get(up_face_id)}", (0, 255, 255), 0.55, 22)
    else:
        put("UP face:    unknown (need pose)", (150, 150, 150))

    if front_face_id is not None:
        put(f"FRONT face: tag{front_face_id} ({TAG_FACE_INFO[front_face_id].name}) "
            f"= {memory.locked.get(front_face_id)}", (0, 255, 255), 0.55, 22)
    else:
        put("FRONT face: unknown (need pose)", (150, 150, 150))

    return hud


def parse_args():
    p = argparse.ArgumentParser(description="Cube AprilTag pose + CNN number detector (sim2real)")
    p.add_argument("--mode", choices=["sim", "real"], default="sim")
    p.add_argument("--scene", default=DEFAULT_SCENE_PATH, help="Path to scene.xml (sim mode; also used "
                                                                 "for the geometry self-check in real mode if present)")
    p.add_argument("--camera-index", type=int, default=0, help="Webcam index (real mode)")
    p.add_argument("--calib", default=None, help="Path to .npz with camera_matrix/dist_coeffs (real mode)")
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Path to face_cnn.pt checkpoint")
    p.add_argument("--tag-size", type=float, default=TAG_SIZE_M, help="Physical tag size in meters")
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--no-physics", action="store_true", help="Sim mode: freeze physics, just visualize")
    p.add_argument("--cam-up", type=float, nargs=3, default=None,
                    help="Real mode override: world-up direction expressed in camera frame, e.g. 0 -1 0")
    p.add_argument("--quiet", action="store_true", help="Suppress per-tag [LOCK] terminal spam; "
                                                          "[DETECT] up/front announcements still print")
    p.add_argument("--skip-self-check", action="store_true",
                    help="Skip the scene.xml vs TAG_FACE_INFO geometry self-check")
    p.add_argument("--debug-dir", default=None,
                    help="If set, pressing 'c' during a run saves every currently-visible "
                         "face's rectified crop (exactly what the CNN sees) to this folder, "
                         "with prediction/confidence/full-probability-vector printed to the "
                         "terminal. Use this to diagnose misaligned crops or unexpected classes.")
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Mode: {args.mode} | device: {device}")

    if not args.skip_self_check:
        verify_tag_geometry_against_scene(args.scene)

    try:
        model, classes = load_model(args.model_path, device)
        print(f"[INFO] CNN loaded. Classes: {classes}")
    except Exception as e:
        print(f"[FATAL] Could not load CNN checkpoint: {e}")
        sys.exit(1)

    try:
        detector = TagDetector()
    except Exception as e:
        print(f"[FATAL] {e}")
        sys.exit(1)

    cam_up_override = np.array(args.cam_up) if args.cam_up else None

    try:
        if args.mode == "sim":
            source = MuJoCoCameraSource(args.scene, width=args.width, height=args.height,
                                         step_physics=not args.no_physics)
        else:
            source = RealCameraSource(args.camera_index, calib_path=args.calib,
                                       width=args.width, height=args.height)
    except Exception as e:
        print(f"[FATAL] Could not start camera source: {e}")
        sys.exit(1)

    memory = FaceMemory(quiet=args.quiet)
    announcer = DetectionAnnouncer()
    prev_time = time.time()

    win_main = "Cube Camera View (tags + pose)"
    win_hud = "Cube State (pose / probability / orientation)"
    cv2.namedWindow(win_main, cv2.WINDOW_NORMAL)
    cv2.namedWindow(win_hud, cv2.WINDOW_NORMAL)

    if args.debug_dir:
        os.makedirs(args.debug_dir, exist_ok=True)
        print(f"[INFO] Debug capture enabled. Press 'c' during a run to dump the exact "
              f"crops the CNN is scoring to: {args.debug_dir}")

    print("\n[DETECTOR READY] Press 'q' or ESC to exit."
          + ("  Press 'r' to reset sim, SPACE to pause physics." if args.mode == "sim" else "")
          + ("  Press 'c' to capture debug crops." if args.debug_dir else ""))

    paused = False
    consecutive_errors = 0
    last_crops: Dict[int, np.ndarray] = {}
    last_probs: Dict[int, Tuple[np.ndarray, float, float]] = {}  # tid -> (probs, entropy, conf)
    try:
        while True:
            try:
                if args.mode == "sim" and paused:
                    source.step_physics = False
                    frame = source.read()
                    source.step_physics = not args.no_physics
                else:
                    frame = source.read()

                if frame is None:
                    print("[WARN] Empty frame from camera source; retrying...")
                    consecutive_errors += 1
                    if consecutive_errors > 30:
                        print("[FATAL] Too many consecutive empty frames; stopping.")
                        break
                    time.sleep(0.05)
                    continue
                consecutive_errors = 0

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                corners_list, ids = detector.detect(gray)

                pose = estimate_cube_pose(corners_list, ids, source.intr, args.tag_size)

                face_conf: Dict[int, Tuple[Optional[str], float]] = {}
                vis = frame.copy()

                if ids is not None and len(ids) > 0:
                    cv2.aruco.drawDetectedMarkers(vis, corners_list, ids)

                if pose is not None:
                    rvec, _ = cv2.Rodrigues(pose.R)
                    try:
                        cv2.drawFrameAxes(vis, source.intr.camera_matrix, source.intr.dist_coeffs,
                                           rvec, pose.t.reshape(3, 1), 0.03)
                    except Exception:
                        pass

                    visible_ids = set(int(i) for i in ids.flatten()) if ids is not None else set()
                    for tid in visible_ids:
                        if tid not in TAG_FACE_INFO:
                            continue
                        face = TAG_FACE_INFO[tid]
                        quad_local = face_corners_local(face.axis, face.sign)
                        img_quad = project_points(quad_local, pose.R, pose.t, source.intr)
                        crop = warp_face_crop(frame, img_quad, IMG_SIZE)

                        pred, conf = None, 0.0
                        if crop is not None:
                            try:
                                pred_cls, conf, entropy, probs_vec = classify_face_crop(model, classes, device, crop)
                                if conf > CONF_THRESH and entropy < ENTROPY_MAX:
                                    pred = pred_cls
                                if args.debug_dir:
                                    last_crops[tid] = crop
                                    last_probs[tid] = (probs_vec, entropy, conf)
                            except Exception as e:
                                print(f"[WARN] CNN inference failed for tag {tid}: {e}")

                        face_conf[tid] = (pred, conf)
                        memory.update(tid, pred, conf)

                        pts = img_quad.astype(int)
                        cv2.polylines(vis, [pts], True, (255, 0, 255), 2)
                        label = f"tag{tid}:{memory.locked[tid]}"
                        cv2.putText(vis, label, tuple(pts[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (255, 0, 255), 2, cv2.LINE_AA)
                else:
                    for tid in TAG_FACE_INFO:
                        memory.update(tid, None)

                up_face_id, front_face_id = None, None
                up_conf, front_conf = 0.0, 0.0
                if pose is not None:
                    if args.mode == "sim" and cam_up_override is None:
                        up_in_cam = source.world_up_in_camera_frame()
                    else:
                        up_in_cam = cam_up_override if cam_up_override is not None else DEFAULT_CAM_UP_IN_CAM_FRAME
                        up_in_cam = up_in_cam / (np.linalg.norm(up_in_cam) + 1e-12)
                    up_face_id = determine_up_face(pose, up_in_cam)
                    front_face_id = determine_front_face(pose)
                    up_conf = face_conf.get(up_face_id, (None, 0.0))[1]
                    front_conf = face_conf.get(front_face_id, (None, 0.0))[1]

                announcer.announce("UP", up_face_id, memory, up_conf)
                announcer.announce("FRONT", front_face_id, memory, front_conf)

                curr_time = time.time()
                fps = 1.0 / max(curr_time - prev_time, 1e-6)
                prev_time = curr_time

                hud = draw_hud(pose, face_conf, memory, up_face_id, front_face_id, fps)

                cv2.putText(vis, f"FPS: {fps:.1f}", (10, vis.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow(win_main, vis)
                cv2.imshow(win_hud, hud)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if args.mode == "sim" and key == ord("r"):
                    source.reset()
                    print("[INFO] Simulation reset.")
                if args.mode == "sim" and key == ord(" "):
                    paused = not paused
                    print(f"[INFO] Physics {'paused' if paused else 'resumed'}.")
                if args.debug_dir and key == ord("c"):
                    ts = time.strftime("%Y%m%d-%H%M%S")
                    if not last_crops:
                        print("[DEBUG] No face crops available yet (no tags visible this frame).")
                    for tid, crop_img in last_crops.items():
                        face = TAG_FACE_INFO[tid]
                        probs_vec, entropy, conf = last_probs[tid]
                        top3_idx = np.argsort(probs_vec)[::-1][:3]
                        top3 = ", ".join(f"{classes[i]}={probs_vec[i]*100:.1f}%" for i in top3_idx)
                        fname = os.path.join(args.debug_dir, f"tag{tid}_{face.name.split('(')[0]}_{ts}.png")
                        cv2.imwrite(fname, crop_img)
                        print(f"[DEBUG] Saved {fname}")
                        print(f"[DEBUG]   tag{tid} {face.name:11s} entropy={entropy:.3f}  top3: {top3}")

            except KeyboardInterrupt:
                raise
            except Exception as e:
                # Bullet-proofing: one bad frame should never kill the whole run.
                print(f"[ERROR] Unhandled exception in main loop, skipping this frame: {e}")
                traceback.print_exc()
                consecutive_errors += 1
                if consecutive_errors > 30:
                    print("[FATAL] Too many consecutive errors; stopping.")
                    break
                time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        source.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()