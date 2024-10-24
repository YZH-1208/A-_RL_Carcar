import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from PIL import Image

# 圖片路徑
image_path = '/home/daniel/maps/my_map0924.png'

# 讀取圖片
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 檢查是否成功讀取圖片
if img is None:
    print(f"Failed to load image at {image_path}")
    exit(1)

# 初始化縮放比例
scale_factor = 1.0

# 畫圖
fig, ax = plt.subplots()
img_plot = ax.imshow(img, cmap='gray')

# 記錄點擊座標
clicked_coords = None

def on_click(event):
    """ 處理滑鼠點擊事件，顯示座標與像素值 """
    global clicked_coords
    if event.xdata is not None and event.ydata is not None:
        x, y = int(event.xdata), int(event.ydata)
        clicked_coords = (x, y)
        pixel_value = img[y, x]
        print(f"Clicked at: ({x}, {y}), Pixel value: {pixel_value}")

def update_image():
    """ 根據縮放比例更新圖片顯示 """
    ax.clear()
    resized_img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    ax.imshow(resized_img, cmap='gray')
    fig.canvas.draw()

def zoom_in(event):
    """ 放大圖片 """
    global scale_factor
    scale_factor *= 1.2
    update_image()

def zoom_out(event):
    """ 縮小圖片 """
    global scale_factor
    scale_factor /= 1.2
    update_image()

# 設置放大與縮小按鈕
ax_zoom_in = plt.axes([0.7, 0.01, 0.1, 0.075])
ax_zoom_out = plt.axes([0.81, 0.01, 0.1, 0.075])

btn_zoom_in = Button(ax_zoom_in, 'Zoom In')
btn_zoom_out = Button(ax_zoom_out, 'Zoom Out')

# 綁定放大縮小按鈕的事件處理
btn_zoom_in.on_clicked(zoom_in)
btn_zoom_out.on_clicked(zoom_out)

# 綁定滑鼠點擊事件
fig.canvas.mpl_connect('button_press_event', on_click)

plt.show()
