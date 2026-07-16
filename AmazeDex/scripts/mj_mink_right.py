import mujoco
import mujoco.viewer
import time 
import numpy as np

ROOT_PATH = "C:\\Users\\luvja\\Desktop\\updatedwithstand\\AmazeDex\\resources\\"

model = mujoco.MjModel.from_xml_path(f"{ROOT_PATH}scene.xml")
data = mujoco.MjData(model)

camera_id = model.camera("tracking_camera").id

with mujoco.viewer.launch_passive(model, data) as viewer:
    
    with viewer.lock():
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = camera_id

    # Simulation loop
    
    while viewer.is_running():
        mujoco.mj_step(model, data)
        t = data.time
        data.ctrl[0] = np.sin(t+1)
        data.ctrl[1] = np.sin(t+3)
        data.ctrl[3] = np.sin(1*t+3)
        data.ctrl[4] = np.cos(t+5)
        data.ctrl[5] = np.cos(5*t+2)
        data.ctrl[6] = np.tan(2*t+1)
        data.ctrl[7] = np.cos(t+0.5)

        
        viewer.sync()