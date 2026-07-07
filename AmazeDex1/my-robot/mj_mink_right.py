import mujoco
import mujoco.viewer
import time 
import numpy as np

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
        t = data.time
        data.ctrl[0] = np.sin(t+1 )  #thumb
        data.ctrl[1] = np.sin(t+3 )  #thumb 
        data.ctrl[2] = np.cos(t +2)   #finger 1
        data.ctrl[3] = np.sin(3*t+3)  #finger 1
        data.ctrl[4] = np.cos(t+5 ) #finger 2
        data.ctrl[5] = np.cos(5*t +2) #finger 2
        data.ctrl[6] = np.tan(2*t+1) #finger 3
        data.ctrl[7] = np.cos(t+0.5 ) #finger 3

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