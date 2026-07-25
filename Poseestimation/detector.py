import cv2
import numpy as np
import mujoco
import mujoco.viewer
from pupil_apriltags import Detector
from scipy.spatial.transform import Rotation as R

# --- MuJoCo model ---
model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)

cam_name = "tracking_camera"
cam_id = model.camera(cam_name).id
tag_geom_id = model.geom("tag_number_5").id

# Lower offscreen render resolution -- this (plus the removed cv2 window) is
# what was mainly causing the MuJoCo control panel to feel laggy. 1080x1080
# every single physics step is expensive; 480x480 is plenty for pose accuracy
# at these tag sizes and is much cheaper to render each frame.
WIDTH, HEIGHT = 480, 480
renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)

fovy = model.cam_fovy[cam_id]
fy = HEIGHT / (2 * np.tan(np.radians(fovy) / 2))
fx = fy
cx, cy = WIDTH / 2, HEIGHT / 2

detector = Detector(
    families="tag36h11",
    nthreads=1,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0
)

TAG_SIZE = 0.01  # meters

# Position transform: OpenCV (+Z forward) -> MuJoCo camera-local (-Z forward)
cv_to_mj = np.array([
    [1,  0,  0],
    [0, -1,  0],
    [0,  0, -1]
])

# Rotation flip to align AprilTag's tag-frame convention with the geom's local frame.
# Flips X and Z axes (180 deg about the Y axis)
tag_frame_flip = np.diag([-1, 1, -1])


def get_tag_ground_truth():
    pos = data.geom_xpos[tag_geom_id].copy()
    rotmat = data.geom_xmat[tag_geom_id].reshape(3, 3).copy()
    return pos, rotmat


def cam_pose_to_world(pos_cam, rot_cam):
    cam_pos_world = data.cam_xpos[cam_id]
    cam_rot_world = data.cam_xmat[cam_id].reshape(3, 3)

    pos_cam_mj = cv_to_mj @ pos_cam
    rot_cam_mj = cv_to_mj @ rot_cam

    pos_world = cam_pos_world + cam_rot_world @ pos_cam_mj
    rot_world = cam_rot_world @ rot_cam_mj
    return pos_world, rot_world


mujoco.mj_forward(model, data)

# Only render + run AprilTag detection every N physics steps.
# Rendering (even offscreen) is the expensive part of this loop -- doing it
# every single step is what's fighting the slider/control panel for CPU time.
# mj_step + viewer.sync() still run every step, so physics and slider response
# stay smooth; detection just updates less often.
DETECT_EVERY_N_STEPS = 5

print("Launching MuJoCo Viewer. Use the Control tab sliders to move the fingers.")
print(f"Pose values print every {DETECT_EVERY_N_STEPS} steps below (no separate camera window).")

with mujoco.viewer.launch_passive(model, data) as viewer:
    step = 0
    while viewer.is_running():
        # Step the physics simulation (no hardcoded data.ctrl overrides -- sliders drive control)
        mujoco.mj_step(model, data)

        if step % DETECT_EVERY_N_STEPS == 0:
            # --- offscreen render from tracking_camera for AprilTag (no window shown) ---
            renderer.update_scene(data, camera=cam_name)
            rgb = renderer.render()
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

            tags = detector.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=[fx, fy, cx, cy],
                tag_size=TAG_SIZE
            )

            if tags:
                tag = tags[0]
                pos_world, rot_world_raw = cam_pose_to_world(tag.pose_t.flatten(), tag.pose_R)
                rot_world = rot_world_raw @ tag_frame_flip

                gt_pos, gt_rot = get_tag_ground_truth()

                det_quat = R.from_matrix(rot_world).as_quat()  # [x, y, z, w]
                gt_quat = R.from_matrix(gt_rot).as_quat()

                print(f"Step {step:4d}")
                print(f"  Detected  pos (m): x={pos_world[0]:+.4f} y={pos_world[1]:+.4f} z={pos_world[2]:+.4f} "
                      f"| quat (x,y,z,w): {det_quat[0]:+.4f} {det_quat[1]:+.4f} {det_quat[2]:+.4f} {det_quat[3]:+.4f}")
                print(f"  GroundTr  pos (m): x={gt_pos[0]:+.4f} y={gt_pos[1]:+.4f} z={gt_pos[2]:+.4f} "
                      f"| quat (x,y,z,w): {gt_quat[0]:+.4f} {gt_quat[1]:+.4f} {gt_quat[2]:+.4f} {gt_quat[3]:+.4f}")
            else:
                print(f"Step {step:4d} | NO DETECTION")

        # Sync viewer state with physics state every step -- keeps sliders responsive
        # regardless of how often detection runs
        viewer.sync()
        step += 1