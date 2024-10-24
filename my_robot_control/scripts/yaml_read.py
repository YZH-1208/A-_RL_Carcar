import yaml
import numpy as np
from PIL import Image

yaml_path = '/home/daniel/maps/my_map0924.yaml'

with open(yaml_path, 'r') as file:
    map_metadata = yaml.safe_load(file)
    map_origin = map_metadata['origin']  # 地圖原點
    map_resolution = map_metadata['resolution']  # 地圖解析度
    pgm_path = map_metadata['image']  # 取得PGM檔案路徑

    # 使用 PIL 讀取PGM檔
    pgm_image = Image.open(pgm_path).convert('L')
    slam_map = np.array(pgm_image)  # 轉為NumPy陣列
    map_height,map_width = slam_map.shape  # 地圖大小

    print(slam_map)


    # 將佔據格資料轉換成0（空地）和100（障礙物）
    slam_map = np.where(slam_map == 255, 0, slam_map)
    slam_map = np.where(slam_map == 0, 100, slam_map)