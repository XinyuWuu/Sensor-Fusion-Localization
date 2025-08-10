import mujoco as mj
from mujoco import MjModel
from mujoco import MjData
from mujoco import MjvCamera
from mujoco import MjvOption
from mujoco import MjvScene
from mujoco import MjrContext
import glfw
import numpy as np
import matplotlib.pyplot as plt
from mapGenerater import MapGenerater
import h5py
import progressbar

MG = MapGenerater()

model = MjModel.from_xml_path("car.xml")
data = MjData(model)
camera = MjvCamera()
option = MjvOption()
scene = MjvScene()
context = MjrContext()

viewport_width = 300
viewport_height = 300
viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
rgb = np.zeros((viewport_width, viewport_height, 3), dtype=np.uint8)
rgb_vflip = np.zeros((viewport_width, viewport_height, 3), dtype=np.uint8)
depth = np.zeros((viewport_width, viewport_height, 1), dtype=np.float32)
glfw.init()
glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
glfw.window_hint(glfw.DOUBLEBUFFER, glfw.FALSE)
windows = glfw.create_window(
    viewport_width, viewport_height, "invisible", None, None)
glfw.make_context_current(windows)

mj.mjv_defaultCamera(camera)
mj.mjv_defaultOption(option)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

mj.mjr_setBuffer(mj.mjtFramebuffer.mjFB_OFFSCREEN.value, context)

total_sample = 1e6
db = h5py.File("data_1e6_s.hdf5", 'a')

tt_ratio = 0.999
train_data = db.create_dataset(
    "train_data", (int(total_sample * tt_ratio), viewport_width, viewport_height), dtype=np.uint8)
train_label = db.create_dataset(
    "train_label", (int(total_sample * tt_ratio), 3), dtype=np.float64)

test_data = db.create_dataset(
    "test_data", (int(total_sample * (1 - tt_ratio)), viewport_width, viewport_height), dtype=np.uint8)
test_label = db.create_dataset(
    "test_label", (int(total_sample * (1 - tt_ratio)), 3), dtype=np.float64)

widgets = ['progressing: ', progressbar.Percentage(), ' ', progressbar.Bar(marker='0', left='[', right=']'),
           ' ', progressbar.Timer(), ' ', progressbar.ETA()]

pbar = progressbar.ProgressBar(widgets=widgets, maxval=total_sample)
pbar.start()
flag = False
# TODO set controller callback

sampled = 0
while True:
    time_prev = data.time
    # pos = np.array([1.71111933,  0.71758118, -1.11925649])
    # angle=pos[2]
    # data.joint("car").qpos = np.array([
    #     pos[0],
    #     pos[1],
    #     0,
    #     np.cos(angle/2),
    #     0,
    #     0,
    #     np.sin(angle/2)
    # ])
    angle = (np.random.rand() - 0.5) * 2 * np.pi
    data.joint("car").qpos = np.array([
        np.random.rand() * 2.56,
        np.random.rand() * 2.56,
        0,
        np.cos(angle / 2),
        0,
        0,
        np.sin(angle / 2)
    ])
    mj.mj_step(model, data)
    changed = MG.draw_map(*(data.cam("cam1").xpos[0:2]))
    model.body("map_texture").pos = np.array(
        [MG.pos[0], MG.pos[1], model.body("map_texture").pos[2]])
    model.tex_rgb = np.stack([MG.map_array] * 3, axis=2).reshape((-1))
    mj.mjr_uploadTexture(model, context, 0)
    mj.mj_step(model, data)

    # Update scene and render
    camera.fixedcamid = 0
    camera.type = mj.mjtCamera.mjCAMERA_FIXED.value
    mj.mjv_updateScene(model, data, option, None, camera,
                       mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)
    mj.mjr_readPixels(rgb, depth, viewport, context)
    rgb_vflip = np.flipud(rgb)
    # plt.imshow(rgb_vflip)
    # plt.show()
    if sampled < int(tt_ratio * total_sample):
        train_data[sampled] = rgb_vflip[:, :, 0]
        train_label[sampled] = np.array([data.joint("car").qpos[0],
                                         data.joint("car").qpos[1],
                                         np.arctan2(2 * data.joint("car").qpos[3] * data.joint(
                                             "car").qpos[6], 1 - 2 * data.joint("car").qpos[6]**2)
                                         ])
    else:
        test_data[sampled - int(tt_ratio * total_sample)] = rgb_vflip[:, :, 0]
        test_label[sampled - int(tt_ratio * total_sample)] = np.array([data.joint("car").qpos[0],
                                                                       data.joint(
                                                                           "car").qpos[1],
                                                                       np.arctan2(2 * data.joint("car").qpos[3] * data.joint(
                                                                           "car").qpos[6], 1 - 2 * data.joint("car").qpos[6]**2)
                                                                       ])
    sampled += 1
    if sampled % 1000 == 0:
        pbar.update(sampled)
    if sampled >= total_sample:
        break


glfw.terminate()
db.close()
pbar.finish()
