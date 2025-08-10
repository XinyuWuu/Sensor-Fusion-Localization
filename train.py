import cv2
import h5py
import numpy as np
import pytesseract
import tesserocr
import torch
import torch.nn.utils.rnn as rnn_utils
from torch import nn
from torch.utils.data import DataLoader, Dataset
from matplotlib import pyplot as plt
from PIL import Image
from tesserocr import RIL, PyTessBaseAPI, get_languages


from mapGenerater import MapGenerater

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class mydataset(Dataset):
    def __init__(self, data, label, img_h, img_w) -> None:
        super(mydataset, self).__init__()
        self.data = data
        self.label = label
        self.img_h = img_h
        self.img_w = img_w

    def __getitem__(self, index):
        return self.data[index].reshape(-1, self.img_h, self.img_w), self.label[index].astype(np.float32)

    def __len__(self):
        return self.label.shape[0]


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
        x0 = rnn_utils.PackedSequence(
            data=x0, batch_sizes=x.batch_sizes)
        h3, (ht3, ct3) = self.lstm3(x0)
        y3 = self.FC3(torch.concat([ht3[-1], ht3[-2]], dim=1))
        return y3


MG = MapGenerater()

# img_temp_box = []
# for i in range(16):
#     for j in range(16):
#         key1 = MG.img_box_keys[i]
#         key2 = MG.img_box_keys[j]
#         bigimg = np.hstack([MG.img_box[key1], MG.img_box[key2]])
#         img_temp_box.append(cv2.merge(
#             [bigimg[:, :, 2], bigimg[:, :, 1], bigimg[:, :, 0]]))

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
    # len(tem_contours)
    img_temp_contours.append(tem_contours[-1])
    # img = cv2.drawContours(img_to_detect, tem_contours, -1, [255, 0, 0], 1)
    # plt.imshow(img)
    # plt.show()

data_file = "data_1e6_s.hdf5"
img_h, img_w = 300, 300
db = h5py.File(data_file, 'r')
test_data = db["test_data"]
test_label = db["test_label"]
train_data = db["train_data"]
train_label = db["train_label"]

# index = 35
# template = 0
# img_to_process = cv2.merge([train_data[index]] * 3)
# plt.imshow(img_to_process)
# plt.show()
# _, img_to_process_binary = cv2.threshold(cv2.cvtColor(
#     img_to_process, cv2.COLOR_BGR2GRAY), 128, 255, cv2.THRESH_BINARY_INV)

# img_contours, _ = cv2.findContours(
#     img_to_process_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# contour_to_draw = []
# min_pos = -1
# min_value = 10000
# for i in range(len(img_contours)):
#     if cv2.contourArea(img_contours[i]) < 100:
#         continue
#     min_value = 2
#     min_index = -1
#     for template_index in range(img_temp_box.__len__()):
#         value = cv2.matchShapes(
#             img_temp_contours[template_index], img_contours[i], 1, 0.0)
#         if value < min_value:
#             min_value = value
#             min_index = template_index
#     M = cv2.moments(img_contours[i])
#     img_to_draw = img_to_process.copy()
#     cv2.drawContours(img_to_draw, img_contours, i,
#                      color=[255, 0, 0], thickness=3)
#     print(np.array([min_index, min_value, M['m10'] / M['m00'] /
#           img_h, M['m01'] / M['m00'] / img_w], dtype=np.float32))
#     plt.imshow(img_to_draw)
#     plt.show()

# img = cv2.drawContours(img_to_process, img_contours, -
#                        1, color=[255, 0, 0], thickness=3)
# print(len(img_contours))
# plt.imshow(img)
# plt.show()


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
        xpos = M['m10'] / M['m00'] / img_h
        ypos = M['m01'] / M['m00'] / img_w
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


traindataset = mydataset(db["train_data"],
                         db['train_label'], img_h, img_w)
testdataset = mydataset(db["test_data"],
                        db['test_label'], img_h, img_w)
batch_size = 64
train_data_loader = DataLoader(
    traindataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=6)
test_data_loader = DataLoader(
    testdataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=6)

T2L = Tag2Location(4, 512, 3)
T2L.to(device)

lossf = nn.MSELoss()

opti = torch.optim.Adam(T2L.parameters(), lr=1e-6)

loss_log = []
ave_loss = 2.5
ave_loss_log = []
ave_loss_rho = 0.9

plt.ion()
fig = plt.figure(1, figsize=(20, 20), dpi=300)
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)
total_batch = traindataset.__len__() / batch_size
save_every_batch = int(total_batch / 10)

for epoch in range(8,100):
    for batch, (x, y) in enumerate(train_data_loader):
        x_seqs = []
        for batch_index in range(x.shape[0]):
            img_to_process = cv2.merge(
                [x[batch_index][0].numpy()] * 3)
            # plt.imshow(img_to_process)
            # plt.show()
            x_seq = map2tag(img_to_process, img_temp_box)
            x_seqs.append(torch.from_numpy(np.array(x_seq)))
        indexs = list(range(x_seqs.__len__()))
        indexs = sorted(indexs, key=lambda x: x_seqs[x].size()[
                        0], reverse=True)
        y = y[indexs]
        x_pack = rnn_utils.pack_sequence([x_seqs[ind] for ind in indexs])
        yhat = T2L(x_pack.to(device))
        loss = lossf(y.to(device), yhat)
        opti.zero_grad()
        loss.backward()
        opti.step()
        loss = float(loss.detach().cpu().numpy())
        loss_log.append(loss)
        ave_loss = ave_loss * ave_loss_rho + loss * (1 - ave_loss_rho)
        ave_loss_log.append(ave_loss)
        if batch % 100 == 0:
            ax1.cla()
            ax1.plot(loss_log, color="blue", linewidth=1)
            ax2.cla()
            ax2.plot(ave_loss_log, color="green", linewidth=1)
            plt.gcf().canvas.draw_idle()
            plt.gcf().canvas.start_event_loop(0)
            # plt.pause(0.01) # keep stealing focus
            # break
        if (batch + 1) % save_every_batch == 0:
            torch.save(T2L.state_dict(),
                        f"bilstm/epoch{epoch}_{batch}.ptd")
    torch.save(T2L.state_dict(), f"bilstm/epoch{epoch}.ptd")

torch.save(T2L.state_dict(), f"bilstm/epoch_n.ptd")

loss_cum = 0
for batch, (x, y) in enumerate(test_data_loader):
    x_seqs = []
    for batch_index in range(x.shape[0]):
        img_to_process = cv2.merge(
            [x[batch_index][0].numpy()] * 3)
        # plt.imshow(img_to_process)
        # plt.show()
        x_seq = map2tag(img_to_process, img_temp_box)
        x_seqs.append(torch.from_numpy(np.array(x_seq)))
    indexs = list(range(x_seqs.__len__()))
    indexs = sorted(indexs, key=lambda x: x_seqs[x].size()[
                    0], reverse=True)
    y = y[indexs]
    x_pack = rnn_utils.pack_sequence([x_seqs[ind] for ind in indexs])
    yhat = T2L(x_pack.to(device))
    loss = lossf(y.to(device), yhat)
    loss = float(loss.detach().cpu().numpy())
    loss_cum += loss
    print(batch)
print(loss_cum / 16)
print(ave_loss)
