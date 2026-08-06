import mujoco
import mujoco.viewer
import time 
import numpy as np

ROOT_PATH = "C:\\Users\\luvja\\Desktop\\updatedwithstand\\AmazeDex\\resources\\"

model = mujoco.MjModel.from_xml_path(f"{ROOT_PATH}rock.xml")
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
    

        
        viewer.sync()