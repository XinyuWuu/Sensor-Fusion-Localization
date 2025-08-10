import os
import cv2
import mujoco as mj
from mujoco import MjModel
from mujoco import MjData
from mujoco import MjvCamera
from mujoco import MjvOption
from mujoco import MjvScene
from mujoco import MjrContext
import glfw
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from mapGenerater import MapGenerater
import torch
import torch.nn.utils.rnn as rnn_utils
from torch import nn
import progressbar
from random import gauss


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'


def dc(a, b):
    return ((np.array([np.cos(a), np.sin(a)]) - np.array([np.cos(b), np.sin(b)]))**2).sum()


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
depth = np.zeros((viewport_width, viewport_height, 1), dtype=np.float64)
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

rgb_raw_file = "rgbbuffer.out"
video_file = "video.mp4"
fp = open(rgb_raw_file, 'bw+')

total_sim_time = 600
flag = False
# TODO set controller callback
control_v = 4
data.ctrl = np.array([control_v, control_v])
angle = np.deg2rad(-45)
xpos = 0.1
ypos = 0.1
data.joint("car").qpos = np.array(
    [xpos, ypos, 0, np.cos(angle / 2), 0, 0, np.sin(angle / 2)])
mj.mj_step(model, data)
second_count = 0

T2L = Tag2Location(4, 512, 3)
T2L.to(device)
# T2L.load_state_dict(torch.load("tag2location_res/epoch.ptd"))
T2L.load_state_dict(torch.load("epoch_n.ptd"))

# R = np.array([[0.01274043, -0.00154429, 0.00061475],
#              [-0.00154429, 0.01068359, -0.00071598],
#               [0.00061475, -0.00071598, 0.0625117]])

R = np.array([[0.00774123, -0.00157088, 0.0007838],
              [-0.00157088, 0.00876199, -0.00087822],
              [0.0007838, -0.00087822, 0.03166099]])


class Kalmanf():
    def __init__(self, x, R, Q) -> None:
        self.K = np.zeros([3, 3], dtype=np.float64)
        self.P = np.zeros([3, 3], dtype=np.float64)
        self.R = R
        self.Q = Q
        self.x = x

    def t_up(self, deltat, u):
        self.x = self.x + deltat * u
        self.x[2] = self.x[2] - int(self.x[2] / np.pi) * np.pi
        self.P = self.P + deltat * self.Q

    def m_up(self, y):
        self.K = self.P.dot(np.linalg.inv(self.P + self.R))
        self.x = self.x + self.K.dot(y - self.x)
        self.x[2] = self.x[2] - int(self.x[2] / np.pi) * np.pi
        self.P = (np.eye(3) - self.K).dot(self.P)


Q = np.array([[0.01, 0.0, 0.0],
              [0.0, 0.01, 0.0],
              [0.0, 0.0, 0.03]]) * control_v
Q1 = np.sqrt(Q[0, 0])
Q2 = np.sqrt(Q[1, 1])
Q3 = np.sqrt(Q[2, 2])

x0 = np.array(
    [
        [data.joint("car").qpos[0]],
        [data.joint("car").qpos[1]],
        [
            np.arctan2(
                2 * data.joint("car").qpos[3] * data.joint("car").qpos[6],
                1 - 2 * data.joint("car").qpos[6] ** 2,
            )
        ],
    ]
)
KMf = Kalmanf(x0, R, Q)

x_imu_only = x0

for p in T2L.parameters():
    p.requires_grad = False

time_log = []
xpos_log = []
ypos_log = []
theta_log = []
xpos_hat_log = []
ypos_hat_log = []
theta_hat_log = []

time_k_log = []
xpos_k_log = []
ypos_k_log = []
theta_k_log = []
xpos_k_hat_log = []
ypos_k_hat_log = []
theta_k_hat_log = []

time_i_log = []
xpos_i_log = []
ypos_i_log = []
theta_i_log = []
xpos_i_hat_log = []
ypos_i_hat_log = []
theta_i_hat_log = []

widgets = ['progressing: ', progressbar.Percentage(), ' ', progressbar.Bar(marker='0', left='[', right=']'),
           ' ', progressbar.Timer(), ' ', progressbar.ETA()]
pbar = progressbar.ProgressBar(widgets=widgets, maxval=total_sim_time)
pbar.start()

Hz_imu = 120
imu_pre_time = 0.0
Hz_cam = 30
cam_pre_time = 0.0
time_prev = 0.0  # for visualization


while True:
    if data.time > second_count:
        second_count = int(data.time + 1)
        if second_count % 3 == 0:
            if abs(data.ctrl[0] - data.ctrl[1]) < 1e-3:
                data.ctrl = np.array([control_v * 1.5, control_v * 0.5]) if np.random.rand(
                ) < 0.5 else np.array([control_v * 0.5, control_v * 1.5])
            else:
                data.ctrl = np.array([control_v, control_v])
    mj.mj_step(model, data)
    changed = MG.draw_map(*(data.cam("cam1").xpos[0:2]))
    if changed:
        model.body("map_texture").pos = np.array(
            [MG.pos[0], MG.pos[1], model.body("map_texture").pos[2]])
        model.tex_rgb = np.stack([MG.map_array] * 3, axis=2).reshape((-1))
        mj.mjr_uploadTexture(model, context, 0)
        mj.mj_step(model, data)

    if data.time - imu_pre_time > 1 / Hz_imu:
        u = np.array([[data.joint("car").qvel[0] + gauss(0, Q1)],
                      [data.joint("car").qvel[1] + gauss(0, Q2)],
                      [data.joint("car").qvel[5] + gauss(0, Q3)]])
        x_imu_only = x_imu_only + (data.time - imu_pre_time) * u
        KMf.t_up(data.time - imu_pre_time, u)
        imu_pre_time = data.time

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

        time_i_log.append(data.time)
        xpos_i_hat_log.append(x_imu_only[0, 0])
        ypos_i_hat_log.append(x_imu_only[1, 0])
        theta_i_hat_log.append(x_imu_only[2, 0])
        xpos_i_log.append(y[0])
        ypos_i_log.append(y[1])
        theta_i_log.append(y[2])

        time_k_log.append(data.time)
        xpos_k_hat_log.append(KMf.x[0, 0])
        ypos_k_hat_log.append(KMf.x[1, 0])
        theta_k_hat_log.append(KMf.x[2, 0])
        xpos_k_log.append(y[0])
        ypos_k_log.append(y[1])
        theta_k_log.append(y[2])

    if data.time - cam_pre_time > 1 / Hz_cam:
        cam_pre_time = data.time
        camera.fixedcamid = 0
        camera.type = mj.mjtCamera.mjCAMERA_FIXED.value
        mj.mjv_updateScene(model, data, option, None, camera,
                           mj.mjtCatBit.mjCAT_ALL.value, scene)
        mj.mjr_render(viewport, scene, context)
        mj.mjr_readPixels(rgb, depth, viewport, context)
        rgb_ = np.flipud(rgb)
        rgb_.reshape((-1)).tofile(fp)
        img_to_process = cv2.merge(
            [rgb_[:, :, 0]] * 3)

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

        xmap = np.floor(y[0] / 2.56)
        xmap_plus = y[0] - xmap * 2.56
        xmap_plus_hat = yhat[0] if abs(
            yhat[0] - xmap_plus) < abs(yhat[0] - 2.56 - xmap_plus) else yhat[0] - 2.56
        xmap_plus_hat = xmap_plus_hat if abs(
            xmap_plus_hat - xmap_plus) < abs(yhat[0] + 2.56 - xmap_plus) else yhat[0] + 2.56
        yhat[0] = xmap * 2.56 + xmap_plus_hat

        ymap = np.floor(y[1] / 2.56)
        ymap_plus = y[1] - ymap * 2.56
        ymap_plus_hat = yhat[1] if abs(
            yhat[1] - ymap_plus) < abs(yhat[1] - 2.56 - ymap_plus) else yhat[1] - 2.56
        ymap_plus_hat = ymap_plus_hat if abs(
            ymap_plus_hat - ymap_plus) < abs(yhat[1] + 2.56 - ymap_plus) else yhat[1] + 2.56
        yhat[1] = ymap * 2.56 + ymap_plus_hat

        emap = np.floor(y[2] / (2 * np.pi))
        emap_plus = y[2] - emap * (2 * np.pi)
        # if emap_plus > np.pi:
        #     emap_plus = emap_plus - 2 * np.pi
        if yhat[2] < 0:
            yhat[2] = yhat[2] + 2 * np.pi
        # emap_plus_hat = yhat[2] if abs(
        #     yhat[2] - emap_plus) < abs(yhat[2] - (2 * np.pi) - emap_plus) else yhat[2] - (2 * np.pi)
        # emap_plus_hat = emap_plus_hat if abs(
        #     emap_plus_hat - emap_plus) < abs(yhat[2] + (2 * np.pi) - emap_plus) else yhat[2] + (2 * np.pi)
        # if emap_plus_hat < 0:
        #     emap_plus_hat = emap_plus_hat + 2 * np.pi
        # yhat[2] = emap * (2 * np.pi) + emap_plus_hat
        yhat[2] = emap * (2 * np.pi) + yhat[2]

        # yhat = y
        time_log.append(data.time)
        xpos_log.append(y[0])
        ypos_log.append(y[1])
        theta_log.append(y[2])
        xpos_hat_log.append(yhat[0])
        ypos_hat_log.append(yhat[1])
        theta_hat_log.append(yhat[2])
        yhat = np.array([[yhat[0]],
                        [yhat[1]],
                         [yhat[2]]])
        KMf.m_up(yhat)
        time_k_log.append(data.time)
        xpos_k_hat_log.append(KMf.x[0, 0])
        ypos_k_hat_log.append(KMf.x[1, 0])
        theta_k_hat_log.append(KMf.x[2, 0])
        xpos_k_log.append(y[0])
        ypos_k_log.append(y[1])
        theta_k_log.append(y[2])
    if data.time < total_sim_time:
        pbar.update(data.time)
    if (data.time > total_sim_time):
        break

pbar.finish()

time_log = np.array(time_log, dtype=np.float64)
xpos_log = np.array(xpos_log, dtype=np.float64)
ypos_log = np.array(ypos_log, dtype=np.float64)
theta_log = np.array(theta_log, dtype=np.float64)
xpos_hat_log = np.array(xpos_hat_log, dtype=np.float64)
ypos_hat_log = np.array(ypos_hat_log, dtype=np.float64)
theta_hat_log = np.array(theta_hat_log, dtype=np.float64)

time_i_log = np.array(time_i_log, dtype=np.float64)
xpos_i_log = np.array(xpos_i_log, dtype=np.float64)
ypos_i_log = np.array(ypos_i_log, dtype=np.float64)
theta_i_log = np.array(theta_i_log, dtype=np.float64)
xpos_i_hat_log = np.array(xpos_i_hat_log, dtype=np.float64)
ypos_i_hat_log = np.array(ypos_i_hat_log, dtype=np.float64)
theta_i_hat_log = np.array(theta_i_hat_log, dtype=np.float64)

time_k_log = np.array(time_k_log, dtype=np.float64)
xpos_k_log = np.array(xpos_k_log, dtype=np.float64)
ypos_k_log = np.array(ypos_k_log, dtype=np.float64)
theta_k_log = np.array(theta_k_log, dtype=np.float64)
xpos_k_hat_log = np.array(xpos_k_hat_log, dtype=np.float64)
ypos_k_hat_log = np.array(ypos_k_hat_log, dtype=np.float64)
theta_k_hat_log = np.array(theta_k_hat_log, dtype=np.float64)

x_c = ((xpos_hat_log - xpos_log)**2).mean()
y_c = ((ypos_hat_log - ypos_log)**2).mean()
theta_c = ((theta_hat_log - theta_log)**2).mean()
xy_c = ((xpos_hat_log - xpos_log) * (ypos_hat_log - ypos_log)).mean()
xtheta_c = ((xpos_hat_log - xpos_log) * (theta_hat_log - theta_log)).mean()
ytheta_c = ((ypos_hat_log - ypos_log) * (theta_hat_log - theta_log)).mean()

R_epi = np.array([[x_c, xy_c, xtheta_c],
                  [xy_c, y_c, ytheta_c],
                  [xtheta_c, ytheta_c, theta_c]])

x_i_c = ((xpos_i_hat_log - xpos_i_log)**2).mean()
y_i_c = ((ypos_i_hat_log - ypos_i_log)**2).mean()
theta_i_c = ((theta_i_hat_log - theta_i_log)**2).mean()
xy_i_c = ((xpos_i_hat_log - xpos_i_log) *
          (ypos_i_hat_log - ypos_i_log)).mean()
xtheta_i_c = ((xpos_i_hat_log - xpos_i_log) *
              (theta_i_hat_log - theta_i_log)).mean()
ytheta_i_c = ((ypos_i_hat_log - ypos_i_log) *
              (theta_i_hat_log - theta_i_log)).mean()

R_i_epi = np.array([[x_i_c, xy_i_c, xtheta_i_c],
                    [xy_i_c, y_i_c, ytheta_i_c],
                    [xtheta_i_c, ytheta_i_c, theta_i_c]])

x_k_c = ((xpos_k_hat_log - xpos_k_log)**2).mean()
y_k_c = ((ypos_k_hat_log - ypos_k_log)**2).mean()
theta_k_c = ((theta_k_hat_log - theta_k_log)**2).mean()
xy_k_c = ((xpos_k_hat_log - xpos_k_log) *
          (ypos_k_hat_log - ypos_k_log)).mean()
xtheta_k_c = ((xpos_k_hat_log - xpos_k_log) *
              (theta_k_hat_log - theta_k_log)).mean()
ytheta_k_c = ((ypos_k_hat_log - ypos_k_log) *
              (theta_k_hat_log - theta_k_log)).mean()

R_k_epi = np.array([[x_k_c, xy_k_c, xtheta_k_c],
                    [xy_k_c, y_k_c, ytheta_k_c],
                    [xtheta_k_c, ytheta_k_c, theta_k_c]])

print(R_epi)
# print(np.linalg.)
print(R_i_epi)
print(R_k_epi)

true_color = "green"
prediction_color = "blue"
kalman_color = "purple"
imu_color = "yellow"
marker_size = 1
fig = plt.figure(100, (10, 10), 300)
ax1 = fig.add_subplot(3, 1, 1)
ax2 = fig.add_subplot(3, 1, 2)
ax3 = fig.add_subplot(3, 1, 3)

ax1.scatter(time_log, xpos_hat_log, color=prediction_color, s=marker_size)
ax1.scatter(time_k_log, xpos_k_hat_log, color=kalman_color, s=marker_size)
ax1.scatter(time_i_log, xpos_i_hat_log, color=imu_color, s=marker_size)
ax1.scatter(time_k_log, xpos_k_log, color=true_color, s=marker_size)

ax2.scatter(time_log, ypos_hat_log, color=prediction_color, s=marker_size)
ax2.scatter(time_k_log, ypos_k_hat_log, color=kalman_color, s=marker_size)
ax2.scatter(time_i_log, ypos_i_hat_log, color=imu_color, s=marker_size)
ax2.scatter(time_k_log, ypos_k_log, color=true_color, s=marker_size)

ax3.scatter(time_log, theta_hat_log, color=prediction_color, s=marker_size)
ax3.scatter(time_k_log, theta_k_hat_log, color=kalman_color, s=marker_size)
ax3.scatter(time_i_log, theta_i_hat_log, color=imu_color, s=marker_size)
ax3.scatter(time_k_log, theta_k_log, color=true_color, s=marker_size)

plt.savefig("res_img/img_for_doc_15.png")

fig = plt.figure(200, (10, 10), 300)
ax = fig.add_subplot(1, 1, 1)
ax.scatter(xpos_hat_log, ypos_hat_log, color=prediction_color, s=marker_size)
ax.scatter(xpos_i_hat_log, ypos_i_hat_log, color=imu_color, s=marker_size)
ax.scatter(xpos_k_hat_log, ypos_k_hat_log, color=kalman_color, s=marker_size)
ax.scatter(xpos_k_log, ypos_k_log, color=true_color, s=marker_size)

plt.savefig("res_img/img_for_doc_16.png")

plt.show()
# plt.pause(0.1)

glfw.terminate()
fp.close()

subprocess.run(["ffmpeg",
                "-f", "rawvideo",
                "-pixel_format", "rgb24",
                "-video_size", f"{int(viewport_height)}x{int(viewport_width)}",
                "-framerate", f"{int(Hz_cam)}",
                "-i", rgb_raw_file,
                video_file,
                "-y", ])
# ffmpeg -f rawvideo -pixel_format rgb24 -video_size 600x600 -framerate 60 -i rgbbuffer.out video.mp4
# Estimating duration from bitrate, this may be inaccurate
