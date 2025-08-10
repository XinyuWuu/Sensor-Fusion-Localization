import numpy as np
from matplotlib import pyplot as plt
import PIL.Image as img


class MapGenerater():
    def __init__(self) -> None:
        self.center_cell_x = 100
        self.center_cell_y = 100
        self.map_array = np.full((700, 700), 255, dtype=np.uint8)
        self.pos = np.array([0, 0])
        self.img_box = {}
        self.img_box_keys = ['0', '1', '2', '3', '4', '5', '6',
                             '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']
        self.img_box_values = []
        for i in range(16):
            img_array = np.asarray(
                img.open("resource/c" + str(i) + ".png"))[:, :, 0:3].copy()
            img_array.setflags(write=True)
            for xi in range(img_array.shape[0]):
                for yi in range(img_array.shape[1]):
                    if img_array[xi, yi, :].mean() > 128:
                        img_array[xi, yi, :] = np.array([255, 255, 255])
                    else:
                        img_array[xi, yi, :] = np.array([0, 0, 0])
            self.img_box_values.append(img_array)
            self.img_box[self.img_box_keys[i]] = self.img_box_values[i]
        self.change = True

    def dec2hex(self, a: int):
        s = hex(a)[2:]
        if len(s) == 2:
            return (s)
        else:
            return "0" + s

    def draw_map(self, x: float, y: float):
        if np.floor(100 * x).astype(int) == self.center_cell_x and np.floor(100 * y).astype(int) == self.center_cell_y:
            self.change = False
            return self.change
        self.center_cell_x = np.floor(100 * x).astype(int)
        self.center_cell_y = np.floor(100 * y).astype(int)
        self.map_array = np.full((700, 700), 255, dtype=np.uint8)
        for xi in range(7):
            for yi in range(7):
                xc = self.center_cell_x + xi - 3
                yc = self.center_cell_y - yi + 3
                xc_hex = self.dec2hex(xc % 256)
                yc_hex = self.dec2hex(yc % 256)
                self.map_array[10 + 100 * yi:10 + 100 * yi + 40,
                               25 + 100 * xi:25 + 100 * xi + 25] = self.img_box[xc_hex[0]][:, :, 0]
                self.map_array[10 + 100 * yi:10 + 100 * yi + 40,
                               50 + 100 * xi:50 + 100 * xi + 25] = self.img_box[xc_hex[1]][:, :, 0]
                self.map_array[50 + 100 * yi:50 + 100 * yi + 40,
                               25 + 100 * xi:25 + 100 * xi + 25] = self.img_box[yc_hex[0]][:, :, 0]
                self.map_array[50 + 100 * yi:50 + 100 * yi + 40,
                               50 + 100 * xi:50 + 100 * xi + 25] = self.img_box[yc_hex[1]][:, :, 0]

        for xi in range(6):
            for yi in range(6):
                self.map_array[80 + xi * 100:80 + xi * 100 + 40,
                               80 + yi * 100:80 + yi * 100 + 40] = np.full((40, 40), 0, dtype=np.uint8)
        self.pos = (np.floor(np.array([x, y]) * 100) + 0.5) / 100
        self.change = True
        return self.change
        # return (self.map_array, (np.floor(np.array([x, y]) * 100) + 0.5) / 100)

    def show_map(self):
        rgb_array = np.stack([self.map_array] * 3, axis=2)
        plt.imshow(rgb_array)
        plt.show(block=False)
        return rgb_array


# MG = MapGenerater()
# MG.draw_map(0, 0)
# rgb_arr = MG.show_map()
# map_img = img.fromarray(rgb_arr)
# map_img.save("materials/textures/img3.png")
