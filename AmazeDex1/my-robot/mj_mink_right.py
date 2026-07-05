import mujoco
import mujoco.viewer

ROOT_PATH = "C:\\Users\\luvja\\Desktop\\updatedwithstand\\AmazeDex1"

model = mujoco.MjModel.from_xml_path(
            f"{ROOT_PATH}\\my-robot\\scene.xml"
        )
#MODEL_PATH = "C:\\Users\\luvja\\Desktop\\amaze\\AmazingHand-main\\Demo\\AHSimulation\\AHSimulation\\Hand_Demo\\AH_Right\\mjcf\\scene.xml"
#model = mujoco.MjModel.from_xml_path(model)
data = mujoco.MjData(model)

with mujoco.viewer.launch(model, data):
    while True:
        mujoco.mj_step(model, data)