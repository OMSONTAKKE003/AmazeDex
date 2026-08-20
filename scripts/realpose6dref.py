"""
Real-camera 6-tag cube pose estimation with an external reference tag
defining the world origin.

Tag layout:
  IDs 0-5  : mounted on the cube faces (tag36h11, ~1cm)
  ID 6     : fixed reference tag mounted in the scene (tag36h11, ~5-8cm)
             This tag defines the world origin. All cube poses are expressed
             relative to it, making the estimate camera-position-independent.

*** BEFORE RUNNING ***
  1. Fill in CUBE_HALF_EDGE with your measured cube half-edge length.
  2. Fill in REFERENCE_TAG_SIZE with your measured reference tag size.
  3. Fill in FACE_QUATS_WXYZ if your physical tag orientations differ.
  4. Place calibration.npz (from cameracalibration.py) in the same folder.
  5. Mount the reference tag (ID 6) rigidly in the scene, in camera view at all times.

Controls:
  q / Q / ESC  - quit (ensure the display window has focus)
"""

import argparse
import os
import cv2
import numpy as np
from pupil_apriltags import Detector
from scipy.spatial.transform import Rotation as R

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="6-Tag Cube Pose Estimation with World Origin")
parser.add_argument("--camera", type=int, default=0,
                    help="Camera index (default: 9)")
args, _ = parser.parse_known_args()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CAMERA_INDEX       = 8
FAMILY             = "tag36h11"
CUBE_TAG_SIZE      = 0.01    # meters -- measured edge length of cube face tags (IDs 0-5)
REFERENCE_TAG_SIZE = 0.05    # meters -- measured edge length of reference tag (ID 6)
                              # make this bigger than cube tags for better stability

REFERENCE_TAG_ID   = 6       # the fixed scene tag that defines world origin

CUBE_HALF_EDGE     = 0.025   # meters -- HALF your cube's real edge length

# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CALIB_PATH = os.path.join(SCRIPT_DIR, "calibration.npz")

if not os.path.exists(CALIB_PATH):
    raise FileNotFoundError(
        f"calibration.npz not found in {SCRIPT_DIR}.\n"
        f"Run cameracalibration.py first and place the output here."
    )

calib     = np.load(CALIB_PATH)
camMatrix = calib["camMatrix"]
distCoeff = calib["distCoeff"]
fx, fy    = camMatrix[0, 0], camMatrix[1, 1]
cx, cy    = camMatrix[0, 2], camMatrix[1, 2]
print(f"Loaded calibration.  Reprojection error: {calib['repError']:.4f} px")

# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------
detector = Detector(
    families=FAMILY,
    nthreads=1,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0
)

# ---------------------------------------------------------------------------
# Cube geometry -- offsets and rotations of each face tag relative to cube center
# ---------------------------------------------------------------------------
h = CUBE_HALF_EDGE

# Translation from cube center to each face tag center, in cube body frame
FACE_OFFSET_FROM_CENTER = {
    0: np.array([ 0,  0, -h]),   # bottom  (-Z)
    1: np.array([ h,  0,  0]),   # right   (+X)
    2: np.array([-h,  0,  0]),   # left    (-X)
    3: np.array([ 0,  h,  0]),   # front   (+Y)
    4: np.array([ 0, -h,  0]),   # back    (-Y)
    5: np.array([ 0,  0,  h]),   # top     (+Z)
}

# Rotation of each face tag's local frame relative to cube body frame
# quaternion order: (w, x, y, z)
FACE_QUATS_WXYZ = {
    0: (1,          0,          0,          0),          # bottom  -- identity
    1: (0.7071068,  0,          0.7071068,  0),          # right   -- +90 deg about Y
    2: (0.7071068,  0,         -0.7071068,  0),          # left    -- -90 deg about Y
    3: (0.7071068, -0.7071068,  0,          0),          # front   -- -90 deg about X
    4: (0.7071068,  0.7071068,  0,          0),          # back    -- +90 deg about X
    5: (1,          0,          0,          0),          # top     -- identity
}

FACE_NAMES = {
    0: "-Z (bottom)",
    1: "+X (right)",
    2: "-X (left)",
    3: "+Y (front)",
    4: "-Y (back)",
    5: "+Z (top)",
}

# Pre-compute rotation matrices for each face
FACE_ROTMATS = {
    tid: R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    for tid, q in FACE_QUATS_WXYZ.items()
}

CUBE_TAG_IDS = set(FACE_ROTMATS.keys())   # {0, 1, 2, 3, 4, 5}

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def is_valid_rotation(rotmat, tol=1e-3):
    """Reject degenerate or reflected rotation matrices before feeding to scipy."""
    det = np.linalg.det(rotmat)
    orth_err = np.linalg.norm(rotmat @ rotmat.T - np.eye(3))
    return (det > 0.5) and (orth_err < tol)


def face_pose_to_cube_pose(face_pos_cam, face_rot_cam, tag_id):
    """
    Given a detected cube face tag's pose in the camera frame,
    back out the cube CENTER's pose in the camera frame.

    face_pos_cam : (3,)   translation of tag in camera frame
    face_rot_cam : (3,3)  rotation matrix of tag in camera frame
    tag_id       : int    which face tag (0-5)

    Returns cube_pos_cam (3,), cube_rot_cam (3,3)
    """
    face_rot_local = FACE_ROTMATS[tag_id]         # tag frame in cube body frame
    offset_local   = FACE_OFFSET_FROM_CENTER[tag_id]

    # cube orientation in camera frame
    cube_rot_cam = face_rot_cam @ face_rot_local.T

    # cube center position in camera frame
    # face_pos = cube_pos + cube_rot @ offset  =>  cube_pos = face_pos - cube_rot @ offset
    cube_pos_cam = face_pos_cam - cube_rot_cam @ offset_local

    return cube_pos_cam, cube_rot_cam


def to_world_frame(pos_cam, rot_cam, ref_pos_cam, ref_rot_cam):
    """
    Transform a pose expressed in the camera frame into the
    reference tag's local frame (= world frame).

    ref_rot_cam.T is the inverse rotation since R is orthogonal.

    Returns pos_world (3,), rot_world (3,3)
    """
    delta_cam = pos_cam - ref_pos_cam          # vector in camera frame
    pos_world  = ref_rot_cam.T @ delta_cam     # rotate into ref frame
    rot_world  = ref_rot_cam.T @ rot_cam       # express rotation in ref frame
    return pos_world, rot_world


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30.0)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open camera at index {CAMERA_INDEX}.\n"
        f"Run:  ls /dev/video*  to list available cameras, then pass --camera <index>."
    )

print(f"Camera opened at index {CAMERA_INDEX}.")
print(f"Reference tag ID : {REFERENCE_TAG_ID}  (size {REFERENCE_TAG_SIZE*100:.1f} cm)")
print(f"Cube tag IDs     : 0-5  (size {CUBE_TAG_SIZE*100:.1f} cm)")
print("Press 'q' or ESC to quit.\n")

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read from camera.")
        break

    # undistort before detection
    frame_undist = cv2.undistort(frame, camMatrix, distCoeff)
    gray         = cv2.cvtColor(frame_undist, cv2.COLOR_BGR2GRAY)

    # detect all tags in one pass
    # we pass CUBE_TAG_SIZE here; reference tag size handled separately below
    all_tags = detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=[fx, fy, cx, cy],
        tag_size=CUBE_TAG_SIZE      # used for IDs 0-5; ref tag re-estimated below
    )

    display = frame_undist.copy()

    # --- separate reference tag from cube tags ---
    ref_tag       = None
    ref_pos_cam   = None
    ref_rot_cam   = None
    cube_tags     = []

    for tag in all_tags:
        if tag.tag_id == REFERENCE_TAG_ID:
            ref_tag = tag
        elif tag.tag_id in CUBE_TAG_IDS:
            cube_tags.append(tag)

    # re-estimate reference tag pose with its own (larger) tag size
    if ref_tag is not None:
        # pupil_apriltags doesn't expose per-tag size in a single detect() call,
        # so we re-run detection_pose manually using the correct size for the ref tag
        import pupil_apriltags as _pa
        info = _pa.Detection(
            tag_family   = ref_tag.tag_family,
            tag_id       = ref_tag.tag_id,
            hamming      = ref_tag.hamming,
            decision_margin = ref_tag.decision_margin,
            homography   = ref_tag.homography,
            center       = ref_tag.center,
            corners      = ref_tag.corners,
            pose_R       = ref_tag.pose_R,
            pose_t       = ref_tag.pose_t,
            pose_err     = ref_tag.pose_err,
        )
        # re-solve pose with correct physical size
        pose_r, pose_t, pose_err = detector.detection_pose(
            ref_tag,
            camera_params=(fx, fy, cx, cy),
            tag_size=REFERENCE_TAG_SIZE
        )
        ref_pos_cam = pose_t.flatten()   if hasattr(pose_t, 'flatten') else np.array(pose_t).flatten()
        ref_rot_cam = np.array(pose_r)[:3,:3] if np.array(pose_r).shape == (4,4) else np.array(pose_r)

        # draw reference tag
        for c in ref_tag.corners:
            cv2.circle(display, tuple(c.astype(int)), 7, (0, 165, 255), -1)   # orange
        cv2.putText(display, f"REF ID{REFERENCE_TAG_ID}",
                    tuple(ref_tag.center.astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    # --- cube pose estimation ---
    cube_pos_estimates = []
    cube_rot_estimates = []
    detected_face_names = []

    for tag in cube_tags:
        face_pos_cam = tag.pose_t.flatten()
        face_rot_cam = tag.pose_R

        cube_pos_cam, cube_rot_cam = face_pose_to_cube_pose(
            face_pos_cam, face_rot_cam, tag.tag_id
        )

        if not is_valid_rotation(cube_rot_cam):
            print(f"  WARNING: degenerate rotation discarded from tag_id={tag.tag_id} "
                  f"({FACE_NAMES[tag.tag_id]})")
            continue

        cube_pos_estimates.append(cube_pos_cam)
        cube_rot_estimates.append(cube_rot_cam)
        detected_face_names.append(FACE_NAMES[tag.tag_id])

        # draw cube tag corners
        for c in tag.corners:
            cv2.circle(display, tuple(c.astype(int)), 5, (0, 255, 0), -1)    # green
        cv2.putText(display, f"ID{tag.tag_id}",
                    tuple(tag.center.astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # --- fuse cube pose estimates ---
    if cube_pos_estimates:
        fused_pos_cam = np.mean(cube_pos_estimates, axis=0)

        if len(cube_rot_estimates) == 1:
            fused_rot_cam = cube_rot_estimates[0]
        else:
            try:
                fused_rot_cam = R.from_matrix(cube_rot_estimates).mean().as_matrix()
            except ValueError:
                fused_rot_cam = cube_rot_estimates[0]

        # --- express in world frame if reference tag visible ---
        if ref_pos_cam is not None and ref_rot_cam is not None:
            cube_pos_world, cube_rot_world = to_world_frame(
                fused_pos_cam, fused_rot_cam, ref_pos_cam, ref_rot_cam
            )
            quat_world = R.from_matrix(cube_rot_world).as_quat()   # [x,y,z,w]

            print(f"[WORLD]  {len(cube_pos_estimates)} tag(s): {', '.join(detected_face_names)}")
            print(f"  pos (m) : x={cube_pos_world[0]:+.4f}  "
                  f"y={cube_pos_world[1]:+.4f}  "
                  f"z={cube_pos_world[2]:+.4f}")
            print(f"  quat(x,y,z,w): {quat_world[0]:+.4f}  {quat_world[1]:+.4f}  "
                  f"{quat_world[2]:+.4f}  {quat_world[3]:+.4f}")

            dist_cm = np.linalg.norm(cube_pos_world) * 100
            cv2.putText(display,
                        f"{len(cube_pos_estimates)} cube tag(s) | world dist: {dist_cm:.1f} cm",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.putText(display,
                        f"REF visible -- world frame active",
                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)

        else:
            # reference tag not visible -- fall back to camera frame
            quat_cam = R.from_matrix(fused_rot_cam).as_quat()
            print(f"[CAM FRAME - ref tag not visible]  "
                  f"{len(cube_pos_estimates)} tag(s): {', '.join(detected_face_names)}")
            print(f"  pos (m) : x={fused_pos_cam[0]:+.4f}  "
                  f"y={fused_pos_cam[1]:+.4f}  "
                  f"z={fused_pos_cam[2]:+.4f}")

            dist_cm = np.linalg.norm(fused_pos_cam) * 100
            cv2.putText(display,
                        f"{len(cube_pos_estimates)} cube tag(s) | cam dist: {dist_cm:.1f} cm",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.putText(display,
                        "WARNING: ref tag not visible -- camera frame only",
                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    else:
        cv2.putText(display, "No cube tags detected",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if ref_pos_cam is None:
        cv2.putText(display, f"Reference tag (ID {REFERENCE_TAG_ID}) NOT visible",
                    (20, display.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    cv2.imshow("Cube Pose Estimation -- World Origin from Ref Tag", display)
    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), ord('Q'), 27):
        break

cap.release()
cv2.destroyAllWindows()