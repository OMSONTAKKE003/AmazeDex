"""
Real-camera 6-tag cube pose estimation, with fusion when multiple tags are
visible simultaneously.

*** BEFORE RUNNING: you MUST fill in FACE_OFFSET_FROM_CENTER and
FACE_QUATS_WXYZ below with values measured from YOUR physical cube.
The placeholders here assume a simple symmetric cube with tags centered
on each face -- replace them with your actual measurements (see Step 2
of the real-world checklist: measure each tag's center position relative
to the cube's true geometric center, and record how each tag's printed
orientation maps to the cube's local axes). ***

Unlike the MuJoCo version, there is no separate "world frame" here --
pose is reported directly in the camera's own reference frame, since a
single fixed real camera has no independent ground-truth frame to convert
into. If you later add a second camera or a motion capture reference,
you'd introduce a proper world-frame transform at that point.
"""

import cv2
import numpy as np
import os
from pupil_apriltags import Detector
from scipy.spatial.transform import Rotation as R

# --- Config ---
CAMERA_INDEX = 0
FAMILY = "tag36h11"
TAG_SIZE = 0.01  # meters -- use your MEASURED tag edge length

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CALIB_PATH = os.path.join(SCRIPT_DIR, "calibration.npz")

if not os.path.exists(CALIB_PATH):
    raise FileNotFoundError(
        f"calibration.npz not found in {SCRIPT_DIR}. Run a calibration script first."
    )

calib = np.load(CALIB_PATH)
camMatrix = calib["camMatrix"]
distCoeff = calib["distCoeff"]
fx, fy = camMatrix[0, 0], camMatrix[1, 1]
cx, cy = camMatrix[0, 2], camMatrix[1, 2]

detector = Detector(
    families=FAMILY,
    nthreads=1,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0
)

# ============================================================
# *** FILL THESE IN WITH YOUR MEASURED CUBE GEOMETRY ***
# ============================================================

# Each tag's center position relative to the cube's true geometric center,
# measured in meters, in a local coordinate frame YOU define (e.g. aligned
# with the cube's edges). Placeholder below assumes a simple 5cm cube with
# tags centered on each face -- replace with your actual measurements.
CUBE_HALF_EDGE = 0.025  # meters -- HALF the real cube's edge length, measure this

FACE_OFFSET_FROM_CENTER = {
    0: np.array([0, 0, -CUBE_HALF_EDGE]),   # -Z (bottom)
    1: np.array([CUBE_HALF_EDGE, 0, 0]),    # +X (right)
    2: np.array([-CUBE_HALF_EDGE, 0, 0]),   # -X (left)
    3: np.array([0, CUBE_HALF_EDGE, 0]),    # +Y (front)
    4: np.array([0, -CUBE_HALF_EDGE, 0]),   # -Y (back)
    5: np.array([0, 0, CUBE_HALF_EDGE]),    # +Z (top)
}

# Each tag's fixed orientation relative to your chosen cube-local axes,
# as a quaternion (w, x, y, z). Placeholder assumes each tag's printed
# "up" direction aligns simply with the cube's edges -- VERIFY this
# empirically (see the "print detected vs ground truth matrices" method
# from earlier) rather than trusting it blindly, since real tag mounting
# is rarely perfectly aligned.
FACE_QUATS_WXYZ = {
    0: (1, 0, 0, 0),
    1: (0.7071068, 0, 0.7071068, 0),
    2: (0.7071068, 0, -0.7071068, 0),
    3: (0.7071068, -0.7071068, 0, 0),
    4: (0.7071068, 0.7071068, 0, 0),
    5: (1, 0, 0, 0),
}

FACE_NAMES = {
    0: "-Z (bottom)", 1: "+X (right)", 2: "-X (left)",
    3: "+Y (front)", 4: "-Y (back)", 5: "+Z (top)",
}

# ============================================================

FACE_ROTMATS = {tid: R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
                 for tid, q in FACE_QUATS_WXYZ.items()}


def is_valid_rotation(rotmat, tol=1e-3):
    """Reject degenerate/reflected pose estimates before they reach scipy."""
    det = np.linalg.det(rotmat)
    orthogonality_err = np.linalg.norm(rotmat @ rotmat.T - np.eye(3))
    return (det > 0.5) and (orthogonality_err < tol)


def face_pose_to_cube_pose(face_pos_cam, face_rot_cam, tag_id):
    """Back out the cube center's pose (in camera frame) from a detected face's pose."""
    face_rot_local = FACE_ROTMATS[tag_id]
    offset_local = FACE_OFFSET_FROM_CENTER[tag_id]

    cube_rot_cam = face_rot_cam @ face_rot_local.T
    cube_pos_cam = face_pos_cam - cube_rot_cam @ offset_local
    return cube_pos_cam, cube_rot_cam


cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}")

print("Loaded calibration. Reprojection error (px):", calib["repError"])
print("Press 'q' to quit.\n")
print("*** Remember: FACE_OFFSET_FROM_CENTER / FACE_QUATS_WXYZ must match YOUR")
print("*** physically measured cube -- placeholders will give wrong results")
print("*** on an uncalibrated real cube.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read from camera.")
        break

    frame_undist = cv2.undistort(frame, camMatrix, distCoeff)
    gray = cv2.cvtColor(frame_undist, cv2.COLOR_BGR2GRAY)

    tags = detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=[fx, fy, cx, cy],
        tag_size=TAG_SIZE
    )

    display = frame_undist.copy()

    cube_pos_estimates = []
    cube_rot_estimates = []
    detected_face_names = []

    for tag in tags:
        if tag.tag_id not in FACE_ROTMATS:
            continue  # unknown tag id -- skip

        face_pos_cam = tag.pose_t.flatten()
        face_rot_cam = tag.pose_R

        cube_pos, cube_rot = face_pose_to_cube_pose(face_pos_cam, face_rot_cam, tag.tag_id)

        if not is_valid_rotation(cube_rot):
            print(f"WARNING: discarded degenerate pose from tag_id={tag.tag_id} "
                  f"({FACE_NAMES[tag.tag_id]})")
            continue

        cube_pos_estimates.append(cube_pos)
        cube_rot_estimates.append(cube_rot)
        detected_face_names.append(FACE_NAMES[tag.tag_id])

        for c in tag.corners:
            cv2.circle(display, tuple(c.astype(int)), 5, (0, 255, 0), -1)
        cv2.putText(display, f"ID{tag.tag_id}", tuple(tag.center.astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if cube_pos_estimates:
        fused_pos = np.mean(cube_pos_estimates, axis=0)

        if len(cube_rot_estimates) == 1:
            fused_rot = cube_rot_estimates[0]
        else:
            try:
                fused_rot = R.from_matrix(cube_rot_estimates).mean().as_matrix()
            except ValueError:
                fused_rot = cube_rot_estimates[0]

        fused_quat = R.from_matrix(fused_rot).as_quat()

        print(f"{len(cube_pos_estimates)} tag(s): {', '.join(detected_face_names)}")
        print(f"  FUSED cube pos (m): x={fused_pos[0]:+.4f} y={fused_pos[1]:+.4f} z={fused_pos[2]:+.4f} "
              f"| quat (x,y,z,w): {fused_quat[0]:+.4f} {fused_quat[1]:+.4f} {fused_quat[2]:+.4f} {fused_quat[3]:+.4f}")

        dist_cm = np.linalg.norm(fused_pos) * 100
        cv2.putText(display, f"{len(cube_pos_estimates)} tag(s) | dist: {dist_cm:.1f} cm",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(display, "No tag detected",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("6-Tag Cube Pose Estimation (real camera)", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()