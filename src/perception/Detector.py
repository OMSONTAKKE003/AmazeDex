#!/usr/bin/env python3

"""
Real-camera 6-tag cube pose estimation with an external reference tag
defining the world origin.

Tag layout:
    IDs 0-5 : mounted on the cube faces
              (tag36h11, ~1 cm)

    ID 6    : fixed reference tag mounted in the scene
              (tag36h11, ~5 cm)

The reference tag defines the world origin.

All cube poses are expressed relative to the reference tag.

Controls:
    q / Q / ESC - quit
"""

import argparse
import os

import cv2
import numpy as np

from pupil_apriltags import Detector
from scipy.spatial.transform import Rotation as R


# ===========================================================================
# Arguments
# ===========================================================================

parser = argparse.ArgumentParser(
    description="6-Tag Cube Pose Estimation with World Origin"
)

parser.add_argument(
    "--camera",
    type=int,
    default=0,
    help="Camera index"
)

args, _ = parser.parse_known_args()


# ===========================================================================
# Configuration
# ===========================================================================

# Camera
CAMERA_INDEX = 2

# AprilTag family
FAMILY = "tag36h11"

# Physical size of cube face tags
# IDs 0-5
CUBE_TAG_SIZE = 0.025      # meters = 1 cm

# Physical size of reference tag
# ID 6
REFERENCE_TAG_SIZE = 0.05  # meters = 5 cm

# Reference tag
REFERENCE_TAG_ID = 6

# Half of the physical cube edge length
#
# Example:
# cube edge = 5 cm
# half edge = 2.5 cm
#
CUBE_HALF_EDGE = 0.025     # meters


# ===========================================================================
# Calibration
# ===========================================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

CALIB_PATH = os.path.join(
    SCRIPT_DIR,
    "calibration.npz"
)


if not os.path.exists(CALIB_PATH):

    raise FileNotFoundError(
        f"calibration.npz not found in {SCRIPT_DIR}.\n"
        f"Run cameracalibration.py first and place the output here."
    )


calib = np.load(CALIB_PATH)


camMatrix = calib["camMatrix"]

distCoeff = calib["distCoeff"]


fx = camMatrix[0, 0]

fy = camMatrix[1, 1]

cx = camMatrix[0, 2]

cy = camMatrix[1, 2]


print(
    f"Loaded calibration. "
    f"Reprojection error: {calib['repError']:.4f} px"
)


# ===========================================================================
# AprilTag Detector
# ===========================================================================

detector = Detector(
    families=FAMILY,
    nthreads=1,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0
)


# ===========================================================================
# Cube geometry
# ===========================================================================

h = CUBE_HALF_EDGE


# ---------------------------------------------------------------------------
# Position of each tag relative to the cube center
#
# These vectors are expressed in the cube body frame.
# ---------------------------------------------------------------------------

FACE_OFFSET_FROM_CENTER = {

    # ID 0
    0: np.array([
        0,
        0,
        -h
    ]),

    # ID 1
    1: np.array([
        h,
        0,
        0
    ]),

    # ID 2
    2: np.array([
        -h,
        0,
        0
    ]),

    # ID 3
    3: np.array([
        0,
        h,
        0
    ]),

    # ID 4
    4: np.array([
        0,
        -h,
        0
    ]),

    # ID 5
    5: np.array([
        0,
        0,
        h
    ])
}


# ===========================================================================
# Orientation of each cube face tag
#
# Quaternion format:
#
#     (w, x, y, z)
#
# scipy Rotation expects:
#
#     (x, y, z, w)
# ===========================================================================

FACE_QUATS_WXYZ = {

    # ID 0 - bottom (-Z)
    0: (
        1,
        0,
        0,
        0
    ),

    # ID 1 - right (+X)
    1: (
        0.7071068,
        0,
        0.7071068,
        0
    ),

    # ID 2 - left (-X)
    2: (
        0.7071068,
        0,
        -0.7071068,
        0
    ),

    # ID 3 - front (+Y)
    3: (
        0.7071068,
        -0.7071068,
        0,
        0
    ),

    # ID 4 - back (-Y)
    4: (
        0.7071068,
        0.7071068,
        0,
        0
    ),

    # ID 5 - top (+Z)
    5: (
        1,
        0,
        0,
        0
    )
}


# ===========================================================================
# Human-readable face names
# ===========================================================================

FACE_NAMES = {

    0: "-Z (bottom)",

    1: "+X (right)",

    2: "-X (left)",

    3: "+Y (front)",

    4: "-Y (back)",

    5: "+Z (top)"
}


# ===========================================================================
# Convert face quaternions to rotation matrices
# ===========================================================================

FACE_ROTMATS = {

    tag_id: R.from_quat(
        [
            q[1],
            q[2],
            q[3],
            q[0]
        ]
    ).as_matrix()

    for tag_id, q in FACE_QUATS_WXYZ.items()
}


# Cube IDs
CUBE_TAG_IDS = set(
    FACE_ROTMATS.keys()
)


# ===========================================================================
# Geometry helper
# ===========================================================================

def is_valid_rotation(
    rotmat,
    tol=1e-3
):

    """
    Check whether a matrix is a valid rotation matrix.
    """

    rotmat = np.asarray(rotmat)

    if rotmat.shape != (3, 3):

        return False


    determinant = np.linalg.det(
        rotmat
    )


    orthogonal_error = np.linalg.norm(
        rotmat @ rotmat.T
        - np.eye(3)
    )


    return (
        determinant > 0.5
        and orthogonal_error < tol
    )


# ===========================================================================
# Convert face tag pose -> cube center pose
# ===========================================================================

def face_pose_to_cube_pose(
    face_pos_cam,
    face_rot_cam,
    tag_id
):

    """
    Given the pose of a cube face tag in the camera frame,
    calculate the pose of the cube center in the camera frame.

    Parameters
    ----------
    face_pos_cam : ndarray shape (3,)
        Position of tag in camera coordinates.

    face_rot_cam : ndarray shape (3,3)
        Orientation of tag in camera coordinates.

    tag_id : int
        Cube face tag ID.

    Returns
    -------
    cube_pos_cam : ndarray shape (3,)
        Cube center position in camera coordinates.

    cube_rot_cam : ndarray shape (3,3)
        Cube orientation in camera coordinates.
    """

    # Rotation of tag frame relative to cube frame
    face_rot_local = FACE_ROTMATS[tag_id]


    # Position of tag relative to cube center
    offset_local = FACE_OFFSET_FROM_CENTER[tag_id]


    # -----------------------------------------------------------------------
    # Cube orientation
    #
    # face_rot_cam =
    #     cube_rot_cam @ face_rot_local
    #
    # Therefore:
    #
    # cube_rot_cam =
    #     face_rot_cam @ face_rot_local.T
    # -----------------------------------------------------------------------

    cube_rot_cam = (
        face_rot_cam
        @ face_rot_local.T
    )


    # -----------------------------------------------------------------------
    # Cube center
    #
    # face_pos_cam =
    #     cube_pos_cam
    #     + cube_rot_cam @ offset_local
    #
    # Therefore:
    #
    # cube_pos_cam =
    #     face_pos_cam
    #     - cube_rot_cam @ offset_local
    # -----------------------------------------------------------------------

    cube_pos_cam = (
        face_pos_cam
        - cube_rot_cam @ offset_local
    )


    return (
        cube_pos_cam,
        cube_rot_cam
    )


# ===========================================================================
# Convert camera frame -> reference/world frame
# ===========================================================================

def to_world_frame(
    pos_cam,
    rot_cam,
    ref_pos_cam,
    ref_rot_cam
):

    """
    Convert a pose from camera coordinates into
    the reference-tag coordinate system.

    The reference tag defines the world frame.
    """

    # Position relative to reference tag
    delta_cam = (
        pos_cam
        - ref_pos_cam
    )


    # Rotate position into reference frame
    pos_world = (
        ref_rot_cam.T
        @ delta_cam
    )


    # Rotate orientation into reference frame
    rot_world = (
        ref_rot_cam.T
        @ rot_cam
    )


    return (
        pos_world,
        rot_world
    )


# ===========================================================================
# Camera
# ===========================================================================

cap = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_V4L2
)


cap.set(
    cv2.CAP_PROP_FOURCC,
    cv2.VideoWriter_fourcc(
        *"YUYV"
    )
)

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    640
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    480
)

cap.set(
    cv2.CAP_PROP_FPS,
    30.0
)


if not cap.isOpened():

    raise RuntimeError(
        f"Could not open camera at index {CAMERA_INDEX}.\n"
        f"Run: ls /dev/video* to list available cameras, "
        f"then pass --camera <index>."
    )


print(
    f"Camera opened at index {CAMERA_INDEX}."
)

print(
    f"Reference tag ID : "
    f"{REFERENCE_TAG_ID} "
    f"(size {REFERENCE_TAG_SIZE * 100:.1f} cm)"
)

print(
    f"Cube tag IDs     : "
    f"0-5 "
    f"(size {CUBE_TAG_SIZE * 100:.1f} cm)"
)

print(
    "Press 'q' or ESC to quit.\n"
)


# ===========================================================================
# Main loop
# ===========================================================================

while True:

    # -----------------------------------------------------------------------
    # Read camera frame
    # -----------------------------------------------------------------------

    ret, frame = cap.read()


    if not ret:

        print(
            "Failed to read from camera."
        )

        break


    # -----------------------------------------------------------------------
    # Undistort image
    # -----------------------------------------------------------------------

    frame_undist = cv2.undistort(
        frame,
        camMatrix,
        distCoeff
    )


    # -----------------------------------------------------------------------
    # Convert to grayscale
    # -----------------------------------------------------------------------

    gray = cv2.cvtColor(
        frame_undist,
        cv2.COLOR_BGR2GRAY
    )


    # -----------------------------------------------------------------------
    # Display image
    # -----------------------------------------------------------------------

    display = frame_undist.copy()


    # =======================================================================
    # DETECTION 1
    #
    # Cube tags:
    #
    # IDs 0-5
    # physical size = 1 cm
    # =======================================================================

    cube_detections = detector.detect(

        gray,

        estimate_tag_pose=True,

        camera_params=[
            fx,
            fy,
            cx,
            cy
        ],

        tag_size=CUBE_TAG_SIZE
    )


    # =======================================================================
    # DETECTION 2
    #
    # Reference tag:
    #
    # ID 6
    # physical size = 5 cm
    #
    # We need a second detection because tag_size is passed to
    # detector.detect() and affects the calculated 3D pose.
    # =======================================================================

    reference_detections = detector.detect(

        gray,

        estimate_tag_pose=True,

        camera_params=[
            fx,
            fy,
            cx,
            cy
        ],

        tag_size=REFERENCE_TAG_SIZE
    )


    # =======================================================================
    # Separate cube tags and reference tag
    # =======================================================================

    cube_tags = []

    ref_tag = None


    # -----------------------------------------------------------------------
    # Cube detections
    # -----------------------------------------------------------------------

    for tag in cube_detections:

        if tag.tag_id in CUBE_TAG_IDS:

            cube_tags.append(
                tag
            )


    # -----------------------------------------------------------------------
    # Reference detection
    # -----------------------------------------------------------------------

    for tag in reference_detections:

        if tag.tag_id == REFERENCE_TAG_ID:

            ref_tag = tag

            break


    # =======================================================================
    # Reference tag pose
    # =======================================================================

    ref_pos_cam = None

    ref_rot_cam = None


    if ref_tag is not None:

        # pose_t is already calculated by detector.detect()
        ref_pos_cam = np.asarray(
            ref_tag.pose_t
        ).flatten()


        # pose_R is already calculated by detector.detect()
        ref_rot_cam = np.asarray(
            ref_tag.pose_R
        )


        # -------------------------------------------------------------------
        # Validate reference rotation
        # -------------------------------------------------------------------

        if not is_valid_rotation(
            ref_rot_cam
        ):

            print(
                "WARNING: invalid reference-tag rotation"
            )

            ref_pos_cam = None

            ref_rot_cam = None


        # -------------------------------------------------------------------
        # Draw reference tag
        # -------------------------------------------------------------------

        if ref_pos_cam is not None:

            for c in ref_tag.corners:

                cv2.circle(

                    display,

                    tuple(
                        c.astype(int)
                    ),

                    7,

                    (0, 165, 255),

                    -1
                )


            cv2.putText(

                display,

                f"REF ID{REFERENCE_TAG_ID}",

                tuple(
                    ref_tag.center.astype(int)
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.6,

                (0, 165, 255),

                2
            )


    # =======================================================================
    # Cube pose estimation
    # =======================================================================

    cube_pos_estimates = []

    cube_rot_estimates = []

    detected_face_names = []


    for tag in cube_tags:

        # -------------------------------------------------------------------
        # Tag pose
        # -------------------------------------------------------------------

        face_pos_cam = np.asarray(
            tag.pose_t
        ).flatten()


        face_rot_cam = np.asarray(
            tag.pose_R
        )


        # -------------------------------------------------------------------
        # Convert face pose -> cube center pose
        # -------------------------------------------------------------------

        cube_pos_cam, cube_rot_cam = (
            face_pose_to_cube_pose(

                face_pos_cam,

                face_rot_cam,

                tag.tag_id
            )
        )


        # -------------------------------------------------------------------
        # Check rotation
        # -------------------------------------------------------------------

        if not is_valid_rotation(
            cube_rot_cam
        ):

            print(

                f"WARNING: degenerate rotation "
                f"discarded from tag_id={tag.tag_id} "
                f"({FACE_NAMES[tag.tag_id]})"
            )

            continue


        # -------------------------------------------------------------------
        # Store estimate
        # -------------------------------------------------------------------

        cube_pos_estimates.append(
            cube_pos_cam
        )

        cube_rot_estimates.append(
            cube_rot_cam
        )

        detected_face_names.append(
            FACE_NAMES[tag.tag_id]
        )


        # -------------------------------------------------------------------
        # Draw cube tag
        # -------------------------------------------------------------------

        for c in tag.corners:

            cv2.circle(

                display,

                tuple(
                    c.astype(int)
                ),

                5,

                (0, 255, 0),

                -1
            )


        cv2.putText(

            display,

            f"ID{tag.tag_id}",

            tuple(
                tag.center.astype(int)
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            (0, 255, 0),

            2
        )


    # =======================================================================
    # Fuse cube pose estimates
    # =======================================================================

    if cube_pos_estimates:

        # -------------------------------------------------------------------
        # Average cube position
        # -------------------------------------------------------------------

        fused_pos_cam = np.mean(
            cube_pos_estimates,
            axis=0
        )


        # -------------------------------------------------------------------
        # Fuse cube orientation
        # -------------------------------------------------------------------

        if len(cube_rot_estimates) == 1:

            fused_rot_cam = (
                cube_rot_estimates[0]
            )

        else:

            try:

                fused_rot_cam = (

                    R.from_matrix(
                        cube_rot_estimates
                    )
                    .mean()
                    .as_matrix()
                )

            except ValueError:

                fused_rot_cam = (
                    cube_rot_estimates[0]
                )


        # ===================================================================
        # Reference tag visible
        # ===================================================================

        if (
            ref_pos_cam is not None
            and ref_rot_cam is not None
        ):

            cube_pos_world, cube_rot_world = (
                to_world_frame(

                    fused_pos_cam,

                    fused_rot_cam,

                    ref_pos_cam,

                    ref_rot_cam
                )
            )


            # ----------------------------------------------------------------
            # Convert rotation matrix -> quaternion
            #
            # scipy format:
            # [x, y, z, w]
            # ----------------------------------------------------------------

            quat_world = (
                R.from_matrix(
                    cube_rot_world
                ).as_quat()
            )


            # ----------------------------------------------------------------
            # Terminal output
            # ----------------------------------------------------------------

            print(

                f"[WORLD] "
                f"{len(cube_pos_estimates)} tag(s): "
                f"{', '.join(detected_face_names)}"
            )


            print(

                f"  pos (m) : "
                f"x={cube_pos_world[0]:+.4f}  "
                f"y={cube_pos_world[1]:+.4f}  "
                f"z={cube_pos_world[2]:+.4f}"
            )


            print(

                f"  quat(x,y,z,w): "
                f"{quat_world[0]:+.4f}  "
                f"{quat_world[1]:+.4f}  "
                f"{quat_world[2]:+.4f}  "
                f"{quat_world[3]:+.4f}"
            )


            # ----------------------------------------------------------------
            # Distance from world origin
            # ----------------------------------------------------------------

            dist_cm = (
                np.linalg.norm(
                    cube_pos_world
                ) * 100
            )


            cv2.putText(

                display,

                f"{len(cube_pos_estimates)} "
                f"cube tag(s) | "
                f"world dist: {dist_cm:.1f} cm",

                (20, 40),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.65,

                (0, 255, 0),

                2
            )


            cv2.putText(

                display,

                "REF visible -- world frame active",

                (20, 70),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.55,

                (0, 165, 255),

                2
            )


        # ===================================================================
        # Reference tag NOT visible
        # ===================================================================

        else:

            quat_cam = (
                R.from_matrix(
                    fused_rot_cam
                ).as_quat()
            )


            print(

                f"[CAM FRAME - ref tag not visible] "
                f"{len(cube_pos_estimates)} tag(s): "
                f"{', '.join(detected_face_names)}"
            )


            print(

                f"  pos (m) : "
                f"x={fused_pos_cam[0]:+.4f}  "
                f"y={fused_pos_cam[1]:+.4f}  "
                f"z={fused_pos_cam[2]:+.4f}"
            )


            dist_cm = (
                np.linalg.norm(
                    fused_pos_cam
                ) * 100
            )


            cv2.putText(

                display,

                f"{len(cube_pos_estimates)} "
                f"cube tag(s) | "
                f"cam dist: {dist_cm:.1f} cm",

                (20, 40),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.65,

                (0, 255, 0),

                2
            )


            cv2.putText(

                display,

                "WARNING: ref tag not visible -- "
                "camera frame only",

                (20, 70),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.55,

                (0, 0, 255),

                2
            )


    # =======================================================================
    # No cube tags
    # =======================================================================

    else:

        cv2.putText(

            display,

            "No cube tags detected",

            (20, 40),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (0, 0, 255),

            2
        )


    # =======================================================================
    # Reference visibility warning
    # =======================================================================

    if ref_pos_cam is None:

        cv2.putText(

            display,

            f"Reference tag "
            f"(ID {REFERENCE_TAG_ID}) "
            f"NOT visible",

            (
                20,
                display.shape[0] - 20
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            (0, 0, 255),

            2
        )


    # =======================================================================
    # Show camera
    # =======================================================================

    cv2.imshow(

        "Cube Pose Estimation -- "
        "World Origin from Ref Tag",

        display
    )


    # -----------------------------------------------------------------------
    # Keyboard
    # -----------------------------------------------------------------------

    key = cv2.waitKey(1) & 0xFF


    if key in (
        ord("q"),
        ord("Q"),
        27
    ):

        break


# ===========================================================================
# Cleanup
# ===========================================================================

cap.release()

cv2.destroyAllWindows()