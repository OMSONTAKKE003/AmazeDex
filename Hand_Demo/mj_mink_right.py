import mujoco
import mujoco.viewer

MODEL_PATH = "C:\\Users\\luvja\\Desktop\\amaze\\AmazingHand-main\\Demo\\AHSimulation\\AHSimulation\\Hand_Demo\\AH_Right\\mjcf\\scene.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

with mujoco.viewer.launch(model, data):
    while True:
        mujoco.mj_step(model, data)