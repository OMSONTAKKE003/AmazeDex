"""
Real-camera 6-tag cube pose estimation using Intel RealSense camera.
Uses pyrealsense2 SDK instead of OpenCV VideoCapture to avoid V4L2 timeout issues.

Tag layout:
  IDs 0-5  : mounted on the cube faces (tag36h11, ~1cm)
  ID 6     : fixed reference tag mounted in the scene (tag36h11, ~5-8cm)
             Defines the world origin. All cube poses expressed relative to it.

*** BEFORE RUNNING ***
  1. Fill in CUBE_HALF_EDGE with your measured cube half-edge length.
  2. Fill in REFERENCE_TAG_SIZE with your measured reference tag size (calipers).
  3. Place calibration.npz in the same folder as this script.
  4. Mount reference tag (ID 6) rigidly in scene, always in camera view.

Controls:
  q / Q / ESC  - quit (ensure display window has focus)
"""

import os
import cv2
import numpy as np
import pyrealsense2 as rs
from pupil_apriltags import Detector
from scipy.spatial.transform import Rotation as R

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FAMILY             = "tag36h11"
CUBE_TAG_SIZE      = 0.01    # meters -- measured edge length of cube tags (IDs 0-5)
REFERENCE_TAG_SIZE = 0.05    # meters -- measured edge length of reference tag (ID 6)
REFERENCE_TAG_ID   = 6       # fixed scene tag that defines world origin
CUBE_HALF_EDGE     = 0.025   # meters -- HALF your cube's real edge length

WIDTH, HEIGHT, FPS = 1280, 720, 15

# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CALIB_PATH = os.path.join(SCRIPT_DIR, "calibration.npz")

if not os.path.exists(CALIB_PATH):
    raise FileNotFoundError(
        f"calibration.npz not found in {SCRIPT_DIR}.\n"
        f"Run cameracalibration.py first."
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
# Cube geometry
# ---------------------------------------------------------------------------
h = CUBE_HALF_EDGE

FACE_OFFSET_FROM_CENTER = {
    0: np.array([ 0,  0, -h]),   # bottom  (-Z)
    1: np.array([ h,  0,  0]),   # right   (+X)
    2: np.array([-h,  0,  0]),   # left    (-X)
    3: np.array([ 0,  h,  0]),   # front   (+Y)
    4: np.array([ 0, -h,  0]),   # back    (-Y)
    5: np.array([ 0,  0,  h]),   # top     (+Z)
}

# (w, x, y, z)
FACE_QUATS_WXYZ = {
    0: (1,          0,          0,          0),
    1: (0.7071068,  0,          0.7071068,  0),
    2: (0.7071068,  0,         -0.7071068,  0),
    3: (0.7071068, -0.7071068,  0,          0),
    4: (0.7071068,  0.7071068,  0,          0),
    5: (1,          0,          0,          0),
}

FACE_NAMES = {
    0: "-Z (bottom)",
    1: "+X (right)",
    2: "-X (left)",
    3: "+Y (front)",
    4: "-Y (back)",
    5: "+Z (top)",
}

FACE_ROTMATS = {
    tid: R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    for tid, q in FACE_QUATS_WXYZ.items()
}

CUBE_TAG_IDS = set(FACE_ROTMATS.keys())

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_valid_rotation(rotmat, tol=1e-3):
    det      = np.linalg.det(rotmat)
    orth_err = np.linalg.norm(rotmat @ rotmat.T - np.eye(3))
    return (det > 0.5) and (orth_err < tol)


def face_pose_to_cube_pose(face_pos_cam, face_rot_cam, tag_id):
    """Back out cube CENTER pose in camera frame from a detected face tag."""
    face_rot_local = FACE_ROTMATS[tag_id]
    offset_local   = FACE_OFFSET_FROM_CENTER[tag_id]
    cube_rot_cam   = face_rot_cam @ face_rot_local.T
    cube_pos_cam   = face_pos_cam - cube_rot_cam @ offset_local
    return cube_pos_cam, cube_rot_cam


def to_world_frame(pos_cam, rot_cam, ref_pos_cam, ref_rot_cam):
    """Transform a camera-frame pose into the reference tag's frame (world frame)."""
    delta_cam = pos_cam - ref_pos_cam
    pos_world  = ref_rot_cam.T @ delta_cam
    rot_world  = ref_rot_cam.T @ rot_cam
    return pos_world, rot_world


def reestimate_pose_with_size(tag, tag_size):
    """
    Re-run pose estimation for a single detection using a different tag_size.
    pupil_apriltags bakes tag_size into the main detect() call, so for the
    reference tag (different physical size) we call detection_pose() separately.
    Returns pos (3,), rot (3,3) or None, None on failure.
    """
    try:
        result = detector.detection_pose(
            tag,
            camera_params=(fx, fy, cx, cy),
            tag_size=tag_size
        )
        # detection_pose returns (pose, e0, e1) where pose is 4x4
        pose_mat = np.array(result[0])
        rot = pose_mat[:3, :3]
        pos = pose_mat[:3,  3]
        return pos, rot
    except Exception as e:
        print(f"WARNING: detection_pose failed for tag {tag.tag_id}: {e}")
        return None, None

# ---------------------------------------------------------------------------
# RealSense pipeline
# ---------------------------------------------------------------------------
pipeline = rs.pipeline()
config   = rs.config()
config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)

print("Starting RealSense pipeline...")
pipeline.start(config)

# warmup -- RealSense needs several frames before exposure stabilises
print("Warming up camera (30 frames)...")
for _ in range(30):
    pipeline.wait_for_frames()
print("Ready.\n")

print(f"Reference tag ID : {REFERENCE_TAG_ID}  "
      f"(size {REFERENCE_TAG_SIZE*100:.1f} cm  -- defines world origin)")
print(f"Cube tag IDs     : 0-5  (size {CUBE_TAG_SIZE*100:.1f} cm)")
print("Press 'q' or ESC to quit.\n")

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
try:
    while True:
        frames      = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # numpy BGR frame -- direct drop-in for what cap.read() used to give
        frame = np.asanyarray(color_frame.get_data())

        frame_undist = cv2.undistort(frame, camMatrix, distCoeff)
        gray         = cv2.cvtColor(frame_undist, cv2.COLOR_BGR2GRAY)

        # detect all tags using CUBE_TAG_SIZE (ref tag re-estimated below)
        all_tags = detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=[fx, fy, cx, cy],
            tag_size=CUBE_TAG_SIZE
        )

        display = frame_undist.copy()

        # --- separate reference tag from cube tags ---
        ref_pos_cam = None
        ref_rot_cam = None
        cube_tags   = []

        for tag in all_tags:
            if tag.tag_id == REFERENCE_TAG_ID:
                # re-solve with correct physical size for the reference tag
                pos, rot = reestimate_pose_with_size(tag, REFERENCE_TAG_SIZE)
                if pos is not None and is_valid_rotation(rot):
                    ref_pos_cam = pos
                    ref_rot_cam = rot
                    # draw reference tag in orange
                    for c in tag.corners:
                        cv2.circle(display, tuple(c.astype(int)), 7, (0, 165, 255), -1)
                    cv2.putText(display, f"REF ID{REFERENCE_TAG_ID}",
                                tuple(tag.center.astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            elif tag.tag_id in CUBE_TAG_IDS:
                cube_tags.append(tag)

        # --- cube pose from visible face tags ---
        cube_pos_estimates  = []
        cube_rot_estimates  = []
        detected_face_names = []

        for tag in cube_tags:
            face_pos_cam = tag.pose_t.flatten()
            face_rot_cam = tag.pose_R

            cube_pos_cam, cube_rot_cam = face_pose_to_cube_pose(
                face_pos_cam, face_rot_cam, tag.tag_id
            )

            if not is_valid_rotation(cube_rot_cam):
                print(f"  WARNING: degenerate pose discarded  "
                      f"tag_id={tag.tag_id} ({FACE_NAMES[tag.tag_id]})")
                continue

            cube_pos_estimates.append(cube_pos_cam)
            cube_rot_estimates.append(cube_rot_cam)
            detected_face_names.append(FACE_NAMES[tag.tag_id])

            # draw cube tag corners in green
            for c in tag.corners:
                cv2.circle(display, tuple(c.astype(int)), 5, (0, 255, 0), -1)
            cv2.putText(display, f"ID{tag.tag_id}",
                        tuple(tag.center.astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # --- fuse cube estimates ---
        if cube_pos_estimates:
            fused_pos_cam = np.mean(cube_pos_estimates, axis=0)

            if len(cube_rot_estimates) == 1:
                fused_rot_cam = cube_rot_estimates[0]
            else:
                try:
                    fused_rot_cam = R.from_matrix(cube_rot_estimates).mean().as_matrix()
                except ValueError:
                    fused_rot_cam = cube_rot_estimates[0]

            # --- express in world frame if reference visible ---
            if ref_pos_cam is not None:
                cube_pos_world, cube_rot_world = to_world_frame(
                    fused_pos_cam, fused_rot_cam, ref_pos_cam, ref_rot_cam
                )
                quat_world = R.from_matrix(cube_rot_world).as_quat()  # [x,y,z,w]

                print(f"[WORLD]  {len(cube_pos_estimates)} tag(s): "
                      f"{', '.join(detected_face_names)}")
                print(f"  pos (m) : x={cube_pos_world[0]:+.4f}  "
                      f"y={cube_pos_world[1]:+.4f}  "
                      f"z={cube_pos_world[2]:+.4f}")
                print(f"  quat(x,y,z,w): {quat_world[0]:+.4f}  "
                      f"{quat_world[1]:+.4f}  "
                      f"{quat_world[2]:+.4f}  "
                      f"{quat_world[3]:+.4f}")

                dist_cm = np.linalg.norm(cube_pos_world) * 100
                cv2.putText(display,
                            f"{len(cube_pos_estimates)} cube tag(s) | "
                            f"world dist: {dist_cm:.1f} cm",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                cv2.putText(display, "REF visible -- world frame active",
                            (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)

            else:
                # reference not visible -- fall back to camera frame
                quat_cam = R.from_matrix(fused_rot_cam).as_quat()
                print(f"[CAM FRAME -- ref not visible]  "
                      f"{len(cube_pos_estimates)} tag(s): "
                      f"{', '.join(detected_face_names)}")
                print(f"  pos (m) : x={fused_pos_cam[0]:+.4f}  "
                      f"y={fused_pos_cam[1]:+.4f}  "
                      f"z={fused_pos_cam[2]:+.4f}")

                dist_cm = np.linalg.norm(fused_pos_cam) * 100
                cv2.putText(display,
                            f"{len(cube_pos_estimates)} cube tag(s) | "
                            f"cam dist: {dist_cm:.1f} cm",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                cv2.putText(display,
                            "WARNING: ref tag not visible -- camera frame only",
                            (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        else:
            cv2.putText(display, "No cube tags detected",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # reference tag status at bottom of frame
        if ref_pos_cam is None:
            cv2.putText(display,
                        f"Reference tag (ID {REFERENCE_TAG_ID}) NOT visible",
                        (20, display.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        cv2.imshow("Cube Pose -- World Origin from Ref Tag", display)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("Pipeline stopped.")