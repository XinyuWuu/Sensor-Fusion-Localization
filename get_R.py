import cv2
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
import torch
from torch import nn
import progressbar

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'


class Tag2Location(nn.Module):
    def __init__(self, input_size: int, state_size: int, output_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.state_size = state_size
        self.output_size = output_size

        self.lstm3 = nn.LSTM(input_size=state_size, hidden_size=state_size,
                             num_layers=3, bidirectional=True, batch_first=True)

        self.FC0 = nn.Sequential(
            nn.Linear(input_size, 2 * state_size),
            nn.ReLU(),
            nn.Linear(2 * state_size, state_size),
            nn.ReLU(),
        )

        self.FC3 = nn.Sequential(
            nn.Linear(2 * state_size, 2 * state_size),
            nn.ReLU(),
            nn.Linear(2 * state_size, 2 * state_size),
            nn.ReLU(),
            nn.Linear(2 * state_size, output_size),
        )

    def forward(self, x):
        x0 = self.FC0(x.data)
        h3, (ht3, ct3) = self.lstm3(x0)
        y3 = self.FC3(torch.concat([ht3[-1], ht3[-2]], dim=1))
        return y3


# class Tag2Location(nn.Module):
#     def __init__(self, input_size: int, state_size: int, output_size: int) -> None:
#         super().__init__()
#         self.input_size = input_size
#         self.state_size = state_size
#         self.output_size = output_size

#         self.lstm3 = nn.LSTM(input_size=state_size, hidden_size=state_size,
#                              num_layers=3, batch_first=True)

#         self.FC0 = nn.Sequential(
#             nn.Linear(input_size, 2 * state_size),
#             nn.ReLU(),
#             nn.Linear(2 * state_size, state_size),
#             nn.ReLU(),
#         )

#         self.FC3 = nn.Sequential(
#             nn.Linear(state_size, 2 * state_size),
#             nn.ReLU(),
#             nn.Linear(2 * state_size, 2 * state_size),
#             nn.ReLU(),
#             nn.Linear(2 * state_size, output_size),
#         )

#     def forward(self, x):
#         # print(x.shape)
#         x0 = self.FC0(x)
#         # print(x0.shape)
#         h3, (ht3, ct3) = self.lstm3(x0)
#         # print(ht3.shape)
#         y3 = self.FC3(ht3[-1])
#         return y3


MG = MapGenerater()

img_temp_box = []
for i in range(16):
    key1 = MG.img_box_keys[i]
    img_temp_box.append(cv2.merge(
        [MG.img_box[key1][:, :, 2], MG.img_box[key1][:, :, 1], MG.img_box[key1][:, :, 0]]))

marker = np.full((80, 80), 255, dtype=np.uint8)
marker[20:60, 20:60] = np.zeros((40, 40), dtype=np.uint8)
marker = cv2.merge([marker] * 3)

img_temp_box.append(marker)

img_temp_contours = []
for i in range(img_temp_box.__len__()):
    img_to_detect = img_temp_box[i].copy()
    _, img_to_detect_binary = cv2.threshold(cv2.cvtColor(
        img_to_detect, cv2.COLOR_BGR2GRAY), 200, 255, cv2.THRESH_BINARY_INV)
    tem_contours, _ = cv2.findContours(
        img_to_detect_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    img_temp_contours.append(tem_contours[-1])


def map2tag(img_to_process: cv2.Mat, img_temp_box: list):
    _, img_to_process_binary = cv2.threshold(cv2.cvtColor(
        img_to_process, cv2.COLOR_BGR2GRAY), 128, 255, cv2.THRESH_BINARY_INV)
    img_contours, _ = cv2.findContours(
        img_to_process_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    x_seq = []
    for contour in img_contours:
        if cv2.contourArea(contour) < 100:
            continue
        M = cv2.moments(contour)
        xpos = M['m10'] / M['m00'] / 300
        ypos = M['m01'] / M['m00'] / 300
        if xpos < 0.1 or xpos > 0.9 or ypos < 0.1 or ypos > 0.9:
            continue
        min_value = 2
        min_index = -1
        for template_index in range(img_temp_box.__len__()):
            value = cv2.matchShapes(
                img_temp_contours[template_index], contour, 1, 0.0)
            if value < min_value:
                min_value = value
                min_index = template_index
        x_seq.append(
            np.array([min_index, min_value, xpos, ypos], dtype=np.float32))
    return x_seq


model = MjModel.from_xml_path("car.xml")
data = MjData(model)
camera = MjvCamera()
option = MjvOption()
scene = MjvScene()
context = MjrContext()

viewport_width = 300
viewport_height = 300
framerate = 60
viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
rgb = np.zeros((viewport_width, viewport_height, 3), dtype=np.uint8)
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

total_sample = int(1e4)
sampled = 0
flag = False
# TODO set controller callback

T2L = Tag2Location(4, 512, 3)
T2L.to(device)
# T2L.load_state_dict(torch.load("tag2location_res/epoch.ptd"))
T2L.load_state_dict(torch.load("epoch_n.ptd"))

for p in T2L.parameters():
    p.requires_grad = False

time_log = np.zeros(total_sample, np.float32)
xpos_log = np.zeros(total_sample, np.float32)
ypos_log = np.zeros(total_sample, np.float32)
theta_log = np.zeros(total_sample, np.float32)
xpos_hat_log = np.zeros(total_sample, np.float32)
ypos_hat_log = np.zeros(total_sample, np.float32)
theta_hat_log = np.zeros(total_sample, np.float32)

widgets = ['progressing: ', progressbar.Percentage(), ' ', progressbar.Bar(marker='0', left='[', right=']'),
           ' ', progressbar.Timer(), ' ', progressbar.ETA()]
pbar = progressbar.ProgressBar(widgets=widgets, maxval=total_sample)
pbar.start()

while True:
    time_prev = data.time
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
    if changed:
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
    rgb_ = np.flipud(rgb)
    img_to_process = cv2.merge(
        [rgb_[:, :, 0]] * 3)
    # plt.imshow(img_to_process)
    # plt.show()
    x_seq = map2tag(img_to_process, img_temp_box)
    with torch.no_grad():
        yhat = T2L(torch.from_numpy(
            np.array(x_seq).reshape(1, -1, 4)).to(device))
    yhat = yhat.detach().cpu().numpy()[0]
    y = np.array(
        [
            data.joint("car").qpos[0],
            data.joint("car").qpos[1],
            np.arctan2(
                2 * data.joint("car").qpos[3] * data.joint("car").qpos[6],
                1 - 2 * data.joint("car").qpos[6] ** 2,
            ),
        ]
    )
    time_log[sampled]=data.time
    xpos_log[sampled]=y[0]
    ypos_log[sampled]=y[1]
    theta_log[sampled]=y[2]
    xpos_hat_log[sampled]=yhat[0]
    ypos_hat_log[sampled]=yhat[1]
    theta_hat_log[sampled]=yhat[2]
    sampled += 1
    pbar.update(sampled)
    if (sampled >= total_sample):
        break
pbar.finish()
x_c = ((xpos_hat_log - xpos_log)**2).mean()
y_c = ((ypos_hat_log - ypos_log)**2).mean()
theta_c = ((theta_hat_log - theta_log)**2).mean()
xy_c = ((xpos_hat_log - xpos_log) * (ypos_hat_log - ypos_log)).mean()
xtheta_c = ((xpos_hat_log - xpos_log) * (theta_hat_log - theta_log)).mean()
ytheta_c = ((ypos_hat_log - ypos_log) * (theta_hat_log - theta_log)).mean()

R = np.array([[x_c, xy_c, xtheta_c],
              [xy_c, y_c, ytheta_c],
              [xtheta_c, ytheta_c, theta_c]])

print(R)

true_color = "green"
prediction_color = "blue"
marker_size = 1
fig = plt.figure(100, (10, 10), 300)
ax1 = fig.add_subplot(3, 1, 1)
ax2 = fig.add_subplot(3, 1, 2)
ax3 = fig.add_subplot(3, 1, 3)
ax1.scatter(time_log, xpos_log, color=true_color, s=marker_size)
ax1.scatter(time_log, xpos_hat_log, color=prediction_color, s=marker_size)
ax2.scatter(time_log, ypos_log, color=true_color, s=marker_size)
ax2.scatter(time_log, ypos_hat_log, color=prediction_color, s=marker_size)
ax3.scatter(time_log, theta_log, color=true_color, s=marker_size)
ax3.scatter(time_log, theta_hat_log, color=prediction_color, s=marker_size)

fig = plt.figure(200, (10, 10), 300)
ax = fig.add_subplot(1, 1, 1)
ax.scatter(xpos_log, ypos_log, color=true_color, s=marker_size)
ax.scatter(xpos_hat_log, ypos_hat_log, color=prediction_color, s=marker_size)

plt.show()

glfw.terminate()
