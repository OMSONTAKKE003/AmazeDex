
import os
import csv
import numpy as np
import cv2
import mujoco
from tqdm import tqdm

# --- Configuration Paths ---
XML_PATH = r"C:\Users\luvja\Desktop\updatedwithstand\AmazeDex\resources\scene.xml"
IMAGE_DIR = r"C:\Users\luvja\Desktop\updatedwithstand\dataset\images"
OUTPUT_CSV = r"C:\Users\luvja\Desktop\updatedwithstand\dataset\labels.csv"

NUM_SAMPLES = 2    
UNKNOWN_FRACTION = 0.02     

CAMERA_NAME = "tracking_camera"
CUBE_BODY_NAME = "cube"

# MAPPING: 5=Top, 6=Bottom, 3=Front, 1=Back, 2=Right, 4=Left
FACE_NORMALS = {
    "1": np.array([ 1.0,  0.0,  0.0]),  # +X
    "3": np.array([-1.0,  0.0,  0.0]),  # -X
    "2": np.array([ 0.0, -1.0,  0.0]),  # -Y
    "4": np.array([ 0.0,  1.0,  0.0]),  # +Y
    "5": np.array([ 0.0,  0.0,  1.0]),  # +Z
    "6": np.array([ 0.0,  0.0, -1.0])   # -Z
}
ALL_FACES = list(FACE_NORMALS.keys())


def _axis_sign_from_normal(normal: np.ndarray):
    axis = int(np.argmax(np.abs(normal)))
    sign = int(np.sign(normal[axis]))
    return axis, sign


FACE_AXIS_SIGN = {label: _axis_sign_from_normal(n) for label, n in FACE_NORMALS.items()}

# --- Diversity knobs ---
BASE_POS_CENTER = np.array([-0.07, 0.023, 0.07])
BASE_POS_JITTER_XY = 0.012      # meters, randomizes hand/cube "anchor" per-sample (was fixed before)
BASE_POS_JITTER_Z = 0.008

POS_JITTER_XY = 0.0016           # was 0.015 -- wider spread of poses around the anchor
POS_JITTER_Z_MIN = 0.00
POS_JITTER_Z_MAX = 0.03       # was 0.04

WIGGLE_PROB = 0.60              # was 0.85 -- favor pure-random search more, for orientation diversity
WIGGLE_ANGLE_MAX = 0.88          # was 0.6 -- allow bigger wiggles when we do reuse a known-good quat

CAM_POS_JITTER = 0.004          # meters, small extrinsic perturbation (mounting/calibration tolerance)
CAM_ANGLE_JITTER_DEG = 1.880      # degrees

FINGER_OCCLUSION_PROB = 0.5     # chance to randomize hand/finger joints away from "rest" per sample

CUBE_CENTER_LOCAL = np.array([0.0, -0.035, 0.0])   # must match scene.xml / detector script
CUBE_HALF_SIZE = 0.025                              # must match scene.xml / detector script
CROP_OUT_SIZE = 256   # saved slightly larger than train_face_cnn.py's IMG_SIZE=224 so
                   
INCLUDE_HAND_EMPTY_UNKNOWN = False

SAVE_DEBUG_FULL_SCENE = False


# ==========================================
# Quaternion / geometry helpers
# ==========================================

def quat_multiply(q1, q2):
    """Multiplies two quaternions [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def random_unit_quat():
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    angle = np.random.uniform(-np.pi, np.pi)
    return np.array([np.cos(angle / 2.0), *(axis * np.sin(angle / 2.0))])


def face_scores(model, data, cube_body_id, cam_id):
    """Returns dict {face_label: cos-angle score to camera}, same math used
    everywhere else in this project (dot of world-frame face normal with the
    cube->camera unit vector)."""
    body_xmat = data.xmat[cube_body_id].reshape(3, 3)
    cam_vec = data.cam_xpos[cam_id] - data.xpos[cube_body_id]
    cam_vec /= (np.linalg.norm(cam_vec) + 1e-8)
    scores = {}
    for f in ALL_FACES:
        world_n = body_xmat @ FACE_NORMALS[f]
        scores[f] = float(np.dot(world_n, cam_vec))
    return scores



_MJ_TO_CV_AXIS_FLIP = np.diag([1.0, -1.0, -1.0])  # mujoco cam frame (x-right,y-up,
                                                    # z-out-of-screen) -> OpenCV cam
                                                    # frame (x-right,y-down,z-forward)


def get_camera_matrix(model, cam_id, width, height):
    """Same pinhole model cube_tag_pose_cnn_detector.py's MuJoCoCameraSource uses,
    so a crop rectified here lines up with how the real detector would do it."""
    fovy_deg = float(model.cam_fovy[cam_id])
    fy = height / (2 * np.tan(np.radians(fovy_deg) / 2))
    fx = fy
    return np.array([[fx, 0, width / 2],
                      [0, fy, height / 2],
                      [0, 0, 1]], dtype=np.float64)


def get_cv_cube_pose(data, cam_id, cube_body_id):
    """True cube pose in OpenCV camera-frame convention (x-right, y-down,
    z-forward), computed from MuJoCo's ground-truth world-frame state. Call
    this AFTER any camera jitter for the current frame has been applied
    (and mj_forward'd), so it matches whatever was actually rendered."""
    cam_pos = data.cam_xpos[cam_id]
    R_world_mjcam = data.cam_xmat[cam_id].reshape(3, 3)  # columns = mj cam axes in world
    R_mjcam_world = R_world_mjcam.T
    R_cvcam_world = _MJ_TO_CV_AXIS_FLIP @ R_mjcam_world
    t_cvcam_world = -R_cvcam_world @ cam_pos

    cube_pos = data.xpos[cube_body_id]
    R_world_cube = data.xmat[cube_body_id].reshape(3, 3)

    R_cvcam_cube = R_cvcam_world @ R_world_cube
    t_cvcam_cube = R_cvcam_world @ cube_pos + t_cvcam_world
    return R_cvcam_cube, t_cvcam_cube


def face_corners_local(axis: int, sign: int) -> np.ndarray:
    """Identical to cube_tag_pose_cnn_detector.py's face_corners_local: the
    4 true corners (CCW from outside) of a cube face in cube-body-local
    coordinates, using the cube's real bounding box, not the tiny tag."""
    n = axis
    u_axis, v_axis = (n + 1) % 3, (n + 2) % 3
    h = CUBE_HALF_SIZE

    def make(u_val, v_val):
        p = CUBE_CENTER_LOCAL.copy()
        p[n] += sign * h
        p[u_axis] += u_val
        p[v_axis] += sign * v_val
        return p

    return np.array([make(-h, -h), make(h, -h), make(h, h), make(-h, h)], dtype=np.float64)


def project_face_points(pts_local, R_cvcam_cube, t_cvcam_cube, camera_matrix) -> np.ndarray:
    pts_cam = (R_cvcam_cube @ pts_local.T).T + t_cvcam_cube
    rvec, _ = cv2.Rodrigues(np.eye(3))
    img_pts, _ = cv2.projectPoints(pts_cam, rvec, np.zeros(3), camera_matrix, np.zeros(5))
    return img_pts.reshape(-1, 2)


def rectify_crop(bgr_img, img_quad, out_size: int):
    """Perspective-warp img_quad (4 image-space corners) into a canonical
    out_size x out_size square. Returns None for degenerate projections
    (behind camera, off-frame, absurd scale) -- same guard as the
    detector's warp_face_crop, so anything that WOULDN'T produce a usable
    crop for the real detector is skipped here too, instead of being
    saved as a bogus training image."""
    h, w = bgr_img.shape[:2]
    if not np.all(np.isfinite(img_quad)):
        return None
    area = cv2.contourArea(img_quad.astype(np.float32))
    if area < 25 or area > (w * h * 4):
        return None
    dst = np.array([[0, out_size - 1], [out_size - 1, out_size - 1],
                     [out_size - 1, 0], [0, 0]], dtype=np.float32)
    try:
        M = cv2.getPerspectiveTransform(img_quad.astype(np.float32), dst)
        crop = cv2.warpPerspective(bgr_img, M, (out_size, out_size), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE)
    except Exception:
        return None
    return crop


def rectify_target_face(bgr_img, model, data, cam_id, cube_body_id, face_label, out_size):
    """One-call convenience: ground-truth-rectify the crop for `face_label`
    (a key of FACE_NORMALS/FACE_AXIS_SIGN) from the current sim state and
    the already-rendered bgr_img. Returns None if the projection is
    degenerate (e.g. the face ended up entirely off-frame)."""
    axis, sign = FACE_AXIS_SIGN[face_label]
    camera_matrix = get_camera_matrix(model, cam_id, bgr_img.shape[1], bgr_img.shape[0])
    R_cvcam_cube, t_cvcam_cube = get_cv_cube_pose(data, cam_id, cube_body_id)
    quad_local = face_corners_local(axis, sign)
    img_quad = project_face_points(quad_local, R_cvcam_cube, t_cvcam_cube, camera_matrix)
    return rectify_crop(bgr_img, img_quad, out_size)


# ==========================================
# Hand / finger joint discovery + randomization
# ==========================================

def discover_finger_joints(model):
    """Finds every joint whose name contains 'finger' (matches the naming
    convention used throughout this project, e.g. finger1_motor1). Returns
    (qpos_addrs, ranges) so we can randomize hand pose for occlusion
    diversity without touching amazedex_cube_env.py's ACTUATOR_NAMES list."""
    qpos_adrs, ranges = [], []
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name and "finger" in name:
            adr = model.jnt_qposadr[j]
            jr = model.jnt_range[j]
            qpos_adrs.append(adr)
            # Fall back to a generic +-60 deg range if the XML has no limit set
            if jr[0] == 0.0 and jr[1] == 0.0:
                ranges.append((-1.0, 1.0))
            else:
                ranges.append((jr[0], jr[1]))
    return np.array(qpos_adrs, dtype=int), ranges


def randomize_fingers(data, qpos_adrs, ranges, strength=1.0):
    """strength in [0,1]: 0 = leave at rest, 1 = full random within joint
    limits. Used both for mild partial-occlusion diversity and for the
    'heavily occluded' unknown-class generator (strength close to 1)."""
    for adr, (lo, hi) in zip(qpos_adrs, ranges):
        rest = 0.0
        target = np.random.uniform(lo, hi)
        data.qpos[adr] = (1 - strength) * rest + strength * target


def reset_fingers(data, qpos_adrs):
    for adr in qpos_adrs:
        data.qpos[adr] = 0.0


# ==========================================
# Camera + lighting jitter (per-sample domain randomization in sim)
# ==========================================

def jitter_camera(model, cam_id, base_pos, base_quat):
    dpos = np.random.uniform(-CAM_POS_JITTER, CAM_POS_JITTER, size=3)
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    ang = np.deg2rad(np.random.uniform(-CAM_ANGLE_JITTER_DEG, CAM_ANGLE_JITTER_DEG))
    dq = np.array([np.cos(ang / 2.0), *(axis * np.sin(ang / 2.0))])
    new_quat = quat_multiply(dq, base_quat)
    new_quat /= np.linalg.norm(new_quat)
    model.cam_pos[cam_id] = base_pos + dpos
    model.cam_quat[cam_id] = new_quat


def reset_camera(model, cam_id, base_pos, base_quat):
    model.cam_pos[cam_id] = base_pos
    model.cam_quat[cam_id] = base_quat

def jitter_lighting(model):
    if model.nlight == 0:
        return
    for i in range(model.nlight):
        # Aggressively cut base diffuse to remove the "field light" glare[cite: 1]
        base_diffuse = np.array([0.25, 0.25, 0.25])
        
        # Restrict intensity to strictly darkening multipliers
        intensity = np.random.uniform(0.6, 0.9)
        
        subtle_tint = np.random.uniform(0.97, 1.03, size=3)
        model.light_diffuse[i] = np.clip(base_diffuse * intensity * subtle_tint, 0, 1.0)
        
        # Lower ambient light to restore realistic indoor shadows
        model.light_ambient[i] = np.clip(
            np.array([0.15, 0.15, 0.15]) * np.random.uniform(0.7, 1.0), 0, 1.0
        )
        
        d = model.light_dir[i].copy()
        jitter = np.random.uniform(-0.15, 0.15, size=3)
        d2 = d + jitter
        model.light_dir[i] = d2 / (np.linalg.norm(d2) + 1e-8)
# ==========================================
# Post-render camera-realism augmentation
# ==========================================

def _motion_blur(img):
    ksize = np.random.choice([3, 5, 7, 9])
    angle = np.random.uniform(0, 180)
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    kernel[ksize // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((ksize / 2 - 0.5, ksize / 2 - 0.5), angle, 1)
    kernel = cv2.warpAffine(kernel, M, (ksize, ksize))
    kernel /= (kernel.sum() + 1e-8)
    return cv2.filter2D(img, -1, kernel)

def _exposure(img):
    # Cap max exposure below 1.0 to guarantee a darker, indoor look
    factor = np.random.uniform(0.6, 0.95)
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

def _gamma(img):
    # Shift gamma down (0.7 to 1.0) to gently darken midtones without harsh contrast
    gamma = np.random.uniform(0.7, 1.0)
    inv = 1.0 / gamma
    table = ((np.arange(256) / 255.0) ** inv * 255).astype(np.uint8)
    return cv2.LUT(img, table)

def _white_balance(img):
    # Tightened gain range from 0.85-1.15 to 0.97-1.03 to avoid obvious color casts
    gains = np.random.uniform(0.97, 1.03, size=3)
    out = img.astype(np.float32)
    for c in range(3):
        out[:, :, c] *= gains[c]
    return np.clip(out, 0, 255).astype(np.uint8)

def _gaussian_noise(img):
    std = np.random.uniform(3, 14)
    noise = np.random.normal(0, std, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _jpeg_roundtrip(img):
    quality = int(np.random.uniform(30, 85))
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return img
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def apply_camera_artifacts(bgr_img, heavy_blur=False):
    img = bgr_img.copy()
    if np.random.rand() < 0.5:
        img = _exposure(img)
    if np.random.rand() < 0.4:
        img = _gamma(img)
    if np.random.rand() < 0.4:
        img = _white_balance(img)
    if heavy_blur or np.random.rand() < 0.30:
        for _ in range(2 if heavy_blur else 1):
            img = _motion_blur(img)
    if np.random.rand() < 0.5:
        img = _gaussian_noise(img)
    if np.random.rand() < 0.5:
        img = _jpeg_roundtrip(img)
    return img


# ==========================================
# Original pose-search (kept, lightly parameterized for wider diversity)
# ==========================================
def is_cube_centered_and_in_frame(model, data, cube_body_id, cam_id, img_w=512, img_h=512, margin=100):
    """
    Projects the 3D center of the cube into 2D pixel space and checks if 
    it falls inside a centered safe-zone box.
    """
    cube_pos = data.xpos[cube_body_id]
    cam_pos = data.cam_xpos[cam_id]
    cam_mat = data.cam_xmat[cam_id].reshape(3, 3)
    
    # Vector from camera to cube in camera's local coordinate frame
    p_cam = cam_mat.T @ (cube_pos - cam_pos)
    
    # In MuJoCo camera convention: +X is right, +Y is up, -Z is view direction
    x_cam, y_cam, z_cam = p_cam[0], p_cam[1], -p_cam[2]
    
    # Reject if cube is behind camera or too close
    if z_cam <= 0.1:
        return False
        
    fovy = model.cam_fovy[cam_id]
    focal_length = (img_h / 2.0) / np.tan(np.deg2rad(fovy) / 2.0)
    
    # Project to 2D pixel coordinates
    u = img_w / 2.0 + focal_length * (x_cam / z_cam)
    v = img_h / 2.0 - focal_length * (y_cam / z_cam)
    
    # Check if the cube's center is within our safe inner frame box
    return (margin <= u <= img_w - margin) and (margin <= v <= img_h - margin)

def find_valid_pose_for_target_face(model, data, target_face, base_pos, qpos_adr,
                                     cube_body_id, cam_id, last_known_quat=None):
    """Same collision-guard + face-prominence-check pipeline as the original
    script. Wider position jitter and less reliance on `last_known_quat` so
    the accepted poses span a much larger part of SO(3), not many near-copies
    of the same orientation."""
    for _ in range(2000):
        sample_pos = base_pos + np.array([
            np.random.uniform(-POS_JITTER_XY, POS_JITTER_XY),
            np.random.uniform(-POS_JITTER_XY, POS_JITTER_XY),
            np.random.uniform(POS_JITTER_Z_MIN, POS_JITTER_Z_MAX)
        ])

        if last_known_quat is not None and np.random.rand() < WIGGLE_PROB:
            axis = np.random.randn(3)
            axis /= np.linalg.norm(axis)
            angle = np.random.uniform(-WIGGLE_ANGLE_MAX, WIGGLE_ANGLE_MAX)
            q_delta = np.array([np.cos(angle / 2.0), *(axis * np.sin(angle / 2.0))])
            quat = quat_multiply(q_delta, last_known_quat)
            quat /= np.linalg.norm(quat)
        else:
            quat = random_unit_quat()

        data.qpos[qpos_adr:qpos_adr+3] = sample_pos
        data.qpos[qpos_adr+3:qpos_adr+7] = quat
        mujoco.mj_forward(model, data)
        if not is_cube_centered_and_in_frame(model, data, cube_body_id, cam_id, margin=100):
            continue
        severe_clipping = False
        for i in range(data.ncon):
            geom1 = data.contact[i].geom1
            geom2 = data.contact[i].geom2
            if cube_body_id in (model.geom_bodyid[geom1], model.geom_bodyid[geom2]):
                if data.contact[i].dist < -0.005:
                    severe_clipping = True
                    break
        if severe_clipping:
            continue

        scores = face_scores(model, data, cube_body_id, cam_id)
        best_face = max(scores, key=scores.get)

        if best_face == target_face:
            return True, quat

    return False, None


# ==========================================
# Unknown-class generators
# ==========================================

def gen_unknown_hand_or_empty(model, data, cube_jnt_id, finger_qpos, finger_ranges):
    """Cube hidden underground; sometimes an empty scene, sometimes a bare
    hand posed randomly (occludes nothing of a cube because there is none)."""
    qpos_adr = model.jnt_qposadr[cube_jnt_id]
    data.qpos[qpos_adr:qpos_adr+3] = [0, 0, -10.0]
    if np.random.rand() < 0.7:
        randomize_fingers(data, finger_qpos, finger_ranges, strength=np.random.uniform(0.3, 1.0))
    else:
        reset_fingers(data, finger_qpos)
    mujoco.mj_forward(model, data)
    return True


def gen_unknown_heavily_occluded(model, data, base_pos, qpos_adr, finger_qpos, finger_ranges):
    """A cube IS present (any orientation), but fingers are driven to
    near-closed / random-tight poses so most of it is covered."""
    sample_pos = base_pos + np.array([
        np.random.uniform(-POS_JITTER_XY, POS_JITTER_XY),
        np.random.uniform(-POS_JITTER_XY, POS_JITTER_XY),
        np.random.uniform(0.0, 0.03)
    ])
    quat = random_unit_quat()
    data.qpos[qpos_adr:qpos_adr+3] = sample_pos
    data.qpos[qpos_adr+3:qpos_adr+7] = quat
    randomize_fingers(data, finger_qpos, finger_ranges, strength=np.random.uniform(0.8, 1.0))
    mujoco.mj_forward(model, data)
    return True


def gen_unknown_edge_on_or_ambiguous(model, data, base_pos, qpos_adr, cube_body_id, cam_id,
                                      ambiguous=False):
    """Searches for a pose where either (a) the best face is nearly edge-on
    to the camera (low confidence for everyone), or (b) two faces are almost
    tied for 'most visible' -- both are genuinely ambiguous views a
    classifier should refuse rather than guess on."""
    for _ in range(1500):
        sample_pos = base_pos + np.array([
            np.random.uniform(-POS_JITTER_XY, POS_JITTER_XY),
            np.random.uniform(-POS_JITTER_XY, POS_JITTER_XY),
            np.random.uniform(0.0, POS_JITTER_Z_MAX)
        ])
        quat = random_unit_quat()
        data.qpos[qpos_adr:qpos_adr+3] = sample_pos
        data.qpos[qpos_adr+3:qpos_adr+7] = quat
        mujoco.mj_forward(model, data)

        severe_clipping = False
        for i in range(data.ncon):
            geom1 = data.contact[i].geom1
            geom2 = data.contact[i].geom2
            if cube_body_id in (model.geom_bodyid[geom1], model.geom_bodyid[geom2]):
                if data.contact[i].dist < -0.005:
                    severe_clipping = True
                    break
        if severe_clipping:
            continue

        scores = face_scores(model, data, cube_body_id, cam_id)
        vals = sorted(scores.values(), reverse=True)
        top1, top2 = vals[0], vals[1]

        if ambiguous:
            if (top1 - top2) < 0.05:
                return True
        else:
            if top1 < 0.20:
                return True
    return False


def gen_unknown_leaving_frame(model, data, base_pos, qpos_adr):
    """Pushes the cube toward/past the edge of the camera frustum."""
    direction = np.random.choice(["x", "y", "z"])
    offset = {
        "x": np.array([np.random.choice([-1, 1]) * np.random.uniform(0.06, 0.10), 0, 0]),
        "y": np.array([0, np.random.choice([-1, 1]) * np.random.uniform(0.06, 0.10), 0]),
        "z": np.array([0, 0, np.random.uniform(0.08, 0.14)]),
    }[direction]
    sample_pos = base_pos + offset
    quat = random_unit_quat()
    data.qpos[qpos_adr:qpos_adr+3] = sample_pos
    data.qpos[qpos_adr+3:qpos_adr+7] = quat
    mujoco.mj_forward(model, data)
    return True


# ==========================================
# Background plate (for the visibility check on numbered faces)
# ==========================================

def get_background_frame(model, data, renderer, cube_jnt_id, finger_qpos):
    qpos_adr = model.jnt_qposadr[cube_jnt_id]
    original_pos = data.qpos[qpos_adr:qpos_adr+3].copy()
    original_fingers = data.qpos[finger_qpos].copy() if len(finger_qpos) else None

    data.qpos[qpos_adr:qpos_adr+3] = [0, 0, -10.0]
    reset_fingers(data, finger_qpos)
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=CAMERA_NAME)
    bg_img = renderer.render()

    data.qpos[qpos_adr:qpos_adr+3] = original_pos
    if original_fingers is not None:
        data.qpos[finger_qpos] = original_fingers
    mujoco.mj_forward(model, data)
    return bg_img


# ==========================================
# Main
# ==========================================

def main():
    os.environ.pop('MUJOCO_GL', None)
    os.makedirs(IMAGE_DIR, exist_ok=True)
    if not os.path.exists(XML_PATH):
        return print(f"Error: XML not found at {XML_PATH}")

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=512, width=512)

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
    cube_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
    if cube_jnt_id == -1:
        cube_jnt_id = 0
    cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, CUBE_BODY_NAME)

    finger_qpos, finger_ranges = discover_finger_joints(model)
    base_cam_pos = model.cam_pos[cam_id].copy()
    base_cam_quat = model.cam_quat[cam_id].copy()

    bg_rgb = get_background_frame(model, data, renderer, cube_jnt_id, finger_qpos)

    csv_rows = []
    qpos_adr = model.jnt_qposadr[cube_jnt_id]

    target_per_class = NUM_SAMPLES // len(ALL_FACES)
    numbered_total = target_per_class * len(ALL_FACES)
    unknown_total = int(round(numbered_total * UNKNOWN_FRACTION / (1 - UNKNOWN_FRACTION)))
    grand_total = numbered_total + unknown_total

    class_counts = {f: 0 for f in ALL_FACES}
    class_counts["unknown"] = 0
    last_good_quats = {f: None for f in ALL_FACES}

    unknown_generators = ["hand_empty", "occluded", "edge_on", "ambiguous", "leaving_frame", "blurred"]
    unknown_target_each = max(1, unknown_total // len(unknown_generators))

    print(f"Numbered samples: {numbered_total} ({target_per_class}/face) | "
          f"Unknown samples: {unknown_total} (~{UNKNOWN_FRACTION*100:.1f}% of {grand_total} total)")

    with tqdm(total=grand_total, desc="Generating dataset") as pbar:
        # ---- Numbered faces (1-6), diversified ----
        while sum(class_counts[f] for f in ALL_FACES) < numbered_total:
            needed_faces = [f for f in ALL_FACES if class_counts[f] < target_per_class]
            target_face = np.random.choice(needed_faces)

            # Randomize the "anchor"/base hand position per-sample for positional diversity
            base_pos = BASE_POS_CENTER + np.array([
                np.random.uniform(-BASE_POS_JITTER_XY, BASE_POS_JITTER_XY),
                np.random.uniform(-BASE_POS_JITTER_XY, BASE_POS_JITTER_XY),
                np.random.uniform(-BASE_POS_JITTER_Z, BASE_POS_JITTER_Z),
            ])

            if np.random.rand() < FINGER_OCCLUSION_PROB:
                randomize_fingers(data, finger_qpos, finger_ranges, strength=np.random.uniform(0.15, 0.55))
            else:
                reset_fingers(data, finger_qpos)

            success, successful_quat = find_valid_pose_for_target_face(
                model, data, target_face, base_pos, qpos_adr, cube_body_id, cam_id,
                last_known_quat=last_good_quats[target_face]
            )
            if not success:
                continue

            # 1. Render a clean scene to accurately check if the cube is in the frame
            renderer.update_scene(data, camera=CAMERA_NAME)
            clean_rgb_img = renderer.render()
            
            rgb_diff = np.abs(clean_rgb_img.astype(np.int16) - bg_rgb.astype(np.int16))
            visible_mask = np.max(rgb_diff, axis=2) > 15
            
            if np.sum(visible_mask) < 150:
                continue

            # 2. Once verified, apply jitter and render the final randomized image
            jitter_camera(model, cam_id, base_cam_pos, base_cam_quat)
            jitter_lighting(model)
            renderer.update_scene(data, camera=CAMERA_NAME)
            rgb_img = renderer.render()
            bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
            
            # Reset for the next iteration
            reset_camera(model, cam_id, base_cam_pos, base_cam_quat)

            last_good_quats[target_face] = successful_quat
            bgr_img = apply_camera_artifacts(bgr_img)

            img_filename = f"{len(csv_rows):06d}.png"
            cv2.imwrite(os.path.join(IMAGE_DIR, img_filename), bgr_img)
            csv_rows.append([img_filename, target_face])
            class_counts[target_face] += 1
            pbar.update(1)

        # ---- Unknown class, six failure modes ----
        gen_idx = 0
        while class_counts["unknown"] < unknown_total:
            mode = unknown_generators[gen_idx % len(unknown_generators)]
            gen_idx += 1

            base_pos = BASE_POS_CENTER + np.array([
                np.random.uniform(-BASE_POS_JITTER_XY, BASE_POS_JITTER_XY),
                np.random.uniform(-BASE_POS_JITTER_XY, BASE_POS_JITTER_XY),
                np.random.uniform(-BASE_POS_JITTER_Z, BASE_POS_JITTER_Z),
            ])

            ok = False
            heavy_blur = False
            if mode == "hand_empty":
                ok = gen_unknown_hand_or_empty(model, data, cube_jnt_id, finger_qpos, finger_ranges)
            elif mode == "occluded":
                ok = gen_unknown_heavily_occluded(model, data, base_pos, qpos_adr, finger_qpos, finger_ranges)
            elif mode == "edge_on":
                ok = gen_unknown_edge_on_or_ambiguous(model, data, base_pos, qpos_adr, cube_body_id, cam_id,
                                                       ambiguous=False)
            elif mode == "ambiguous":
                ok = gen_unknown_edge_on_or_ambiguous(model, data, base_pos, qpos_adr, cube_body_id, cam_id,
                                                        ambiguous=True)
            elif mode == "leaving_frame":
                ok = gen_unknown_leaving_frame(model, data, base_pos, qpos_adr)
            elif mode == "blurred":
                # A normal-ish random pose, but rendered with heavy blur baked in.
                quat = random_unit_quat()
                data.qpos[qpos_adr:qpos_adr+3] = base_pos + np.array([0, 0, np.random.uniform(0.0, 0.03)])
                data.qpos[qpos_adr+3:qpos_adr+7] = quat
                reset_fingers(data, finger_qpos)
                mujoco.mj_forward(model, data)
                ok, heavy_blur = True, True

            if not ok:
                continue

            jitter_camera(model, cam_id, base_cam_pos, base_cam_quat)
            jitter_lighting(model)
            renderer.update_scene(data, camera=CAMERA_NAME)
            rgb_img = renderer.render()
            bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
            reset_camera(model, cam_id, base_cam_pos, base_cam_quat)

            bgr_img = apply_camera_artifacts(bgr_img, heavy_blur=heavy_blur)

            img_filename = f"{len(csv_rows):06d}.png"
            cv2.imwrite(os.path.join(IMAGE_DIR, img_filename), bgr_img)
            csv_rows.append([img_filename, "unknown"])
            class_counts["unknown"] += 1
            pbar.update(1)

            if class_counts["unknown"] >= unknown_target_each * len(unknown_generators):
                pass  # allow slight overshoot handling below via the while condition

    with open(OUTPUT_CSV, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "label"])
        writer.writerows(csv_rows)

    print(f"\nDataset complete! Saved {len(csv_rows)} samples to {IMAGE_DIR}")
    print(f"Per-class final counts: {class_counts}")


if __name__ == "__main__":
    main()