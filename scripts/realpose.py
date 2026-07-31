"""
Real-camera single-tag AprilTag pose estimation -- use this to verify
detection and pose output are sane BEFORE moving to the multi-tag cube setup.

Requires calibration.npz (from either the checkerboard or AprilTag-based
calibration script) in the same folder.

Controls:
  q - quit
"""

import cv2
import numpy as np
from pupil_apriltags import Detector
from scipy.spatial.transform import Rotation as R
import os

# --- Config ---
CAMERA_INDEX = 0
TAG_SIZE = 0.01  # meters -- use your MEASURED tag edge length, not the nominal size
FAMILY = "tag36h11"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CALIB_PATH = os.path.join(SCRIPT_DIR, "calibration.npz")

if not os.path.exists(CALIB_PATH):
    raise FileNotFoundError(
        f"calibration.npz not found in {SCRIPT_DIR}. "
        "Run a calibration script first (checkerboard or AprilTag-based)."
    )

calib = np.load(CALIB_PATH)
camMatrix = calib["camMatrix"]
distCoeff = calib["distCoeff"]

fx = camMatrix[0, 0]
fy = camMatrix[1, 1]
cx = camMatrix[0, 2]
cy = camMatrix[1, 2]

print("Loaded calibration:")
print("  Camera matrix:\n", camMatrix)
print("  Reprojection error (px):", calib["repError"])

detector = Detector(
    families=FAMILY,
    nthreads=1,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0
)

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}")

print("\nPress 'q' to quit.")
print("Detected pose prints below (position in meters, orientation as quaternion x,y,z,w).")

# NOTE: distortion is corrected up front (undistort the frame), then detection
# and pose estimation run on the undistorted image using the SAME camMatrix.
# This is Option A from earlier -- undistort first, keep camMatrix consistent
# for both undistortion and detection. Don't mix undistorted frames with the
# raw (distorted) camera matrix, or vice versa -- that silently produces
# subtly wrong poses.

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

    if tags:
        tag = tags[0]

        pos = tag.pose_t.flatten()
        quat = R.from_matrix(tag.pose_R).as_quat()  # [x, y, z, w]

        print(f"Tag {tag.tag_id} | pos (m): x={pos[0]:+.4f} y={pos[1]:+.4f} z={pos[2]:+.4f} "
              f"| quat (x,y,z,w): {quat[0]:+.4f} {quat[1]:+.4f} {quat[2]:+.4f} {quat[3]:+.4f}")

        for c in tag.corners:
            cv2.circle(display, tuple(c.astype(int)), 5, (0, 255, 0), -1)
        cv2.putText(display, f"ID {tag.tag_id}", tuple(tag.center.astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        dist_cm = np.linalg.norm(pos) * 100
        cv2.putText(display, f"Distance: {dist_cm:.1f} cm",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        cv2.putText(display, "No tag detected",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("AprilTag Pose Estimation (real camera)", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()