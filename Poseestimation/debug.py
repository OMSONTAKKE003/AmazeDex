import cv2 as cv
import mujoco

model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

WIDTH, HEIGHT = 1080, 1080
renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)

renderer.update_scene(data, camera="tracking_camera")
rgb = renderer.render()

cv.imwrite("debug_frame.png", cv.cvtColor(rgb, cv.COLOR_RGB2BGR))
print("Saved debug_frame.png")