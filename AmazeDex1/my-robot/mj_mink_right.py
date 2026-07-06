import mujoco
import mujoco.viewer

ROOT_PATH = "C:\\Users\\luvja\\Desktop\\updatedwithstand\\AmazeDex1"

model = mujoco.MjModel.from_xml_path(
    f"{ROOT_PATH}\\my-robot\\scene.xml"
)
data = mujoco.MjData(model)

# Use launch_passive to gain control over the viewer object and your simulation loop
with mujoco.viewer.launch_passive(model, data) as viewer:
    
    # Simulation loop
    while viewer.is_running():
        mujoco.mj_step(model, data)
        with viewer.lock():
            # Set the camera to a fixed position and orientation
         viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
         viewer.cam.trackbodyid = 1
         viewer.cam.azimuth = 55
         viewer.cam.elevation = -50
         viewer.cam.distance = 0.4
         viewer.cam.lookat[:] = [0.3 , 0.4, 0.5]

        
        # Pick up changes and refresh the viewer frame
        viewer.sync()