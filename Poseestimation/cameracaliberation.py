"""
Camera calibration using a single printed AprilTag as the calibration target
-- an alternative to a checkerboard, useful if you already have printed tags
and don't want to print/display a separate checkerboard pattern.

How it works: cv2.calibrateCamera() just needs, for each captured image, a set
of known 3D object points and their corresponding 2D image points. A checkerboard
gives many such point pairs per image (one per internal corner). A single AprilTag
gives 4 (its four corners), since you know its exact physical size. Fewer points
per image means you need MORE images (and more varied angles/distances) than a
checkerboard calibration to get a comparably accurate result -- aim for at least
20-30 captures covering a wide range of tilts, distances, and screen positions.

Controls:
  SPACE - capture the current frame (only if a tag is currently detected)
  q     - finish capturing and run calibration
"""

import cv2
import numpy as np
from pupil_apriltags import Detector

TAG_SIZE = 0.01  # meters -- use your MEASURED value, not the nominal 1cm
CAMERA_INDEX = 0

detector = Detector(
    families="tag36h11",
    nthreads=1,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0
)

# 3D coordinates of the tag's 4 corners in the tag's own local frame
# (center at origin, tag lying flat in the XY plane, Z=0)
half = TAG_SIZE / 2
object_points_template = np.array([
    [-half,  half, 0],   # top-left
    [ half,  half, 0],   # top-right
    [ half, -half, 0],   # bottom-right
    [-half, -half, 0],   # bottom-left
], dtype=np.float32)

world_pts_list = []
img_pts_list = []

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}")

print("Move the tag through varied distances, tilts, and screen positions.")
print("Press SPACE to capture a frame when a tag is detected (green corners).")
print("Aim for at least 20-30 captures. Press 'q' when done to run calibration.")

frame_size = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read from camera.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_size = gray.shape[::-1]  # (width, height)

    # NOTE: estimate_tag_pose is off here -- we only need 2D corners for
    # calibration; camera intrinsics don't exist yet, so pose estimation
    # would be meaningless at this stage anyway.
    tags = detector.detect(gray, estimate_tag_pose=False)

    display = frame.copy()
    detected_this_frame = None

    if tags:
        tag = tags[0]
        detected_this_frame = tag
        for c in tag.corners:
            cv2.circle(display, tuple(c.astype(int)), 5, (0, 255, 0), -1)
        cv2.putText(display, f"Tag {tag.tag_id} detected - SPACE to capture",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(display, "No tag detected",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.putText(display, f"Captures so far: {len(world_pts_list)}",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("AprilTag Calibration Capture", display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord(' ') and detected_this_frame is not None:
        # pupil_apriltags corner order: matches our object_points_template order
        img_pts_list.append(detected_this_frame.corners.astype(np.float32))
        world_pts_list.append(object_points_template.copy())
        print(f"Captured frame #{len(world_pts_list)}")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if len(world_pts_list) < 10:
    print(f"\nOnly {len(world_pts_list)} captures -- too few for a reliable calibration.")
    print("Re-run and aim for at least 20-30 varied captures.")
else:
    print(f"\nRunning calibration with {len(world_pts_list)} captures...")
    rep_error, camMatrix, distCoeff, rvecs, tvecs = cv2.calibrateCamera(
        world_pts_list, img_pts_list, frame_size, None, None
    )

    print("Camera Matrix:\n", camMatrix)
    print("Distortion Coefficients:\n", distCoeff)
    print(f"Reprojection Error (pixels): {rep_error:.4f}")

    if rep_error > 1.0:
        print("\nWARNING: reprojection error is high (>1.0 px). This calibration")
        print("method is less accurate than a checkerboard due to fewer points")
        print("per image. Consider Option A (screen-displayed checkerboard) or")
        print("capturing more varied frames and re-running.")

    np.savez("calibration.npz",
              repError=rep_error,
              camMatrix=camMatrix,
              distCoeff=distCoeff,
              rvecs=rvecs,
              tvecs=tvecs)
    print("\nSaved: calibration.npz")