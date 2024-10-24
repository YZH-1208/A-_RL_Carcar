import numpy as np
import csv
from PIL import Image

# 讀取圖片，並轉換成 NumPy 矩陣
def load_image_as_grid(image_path):
    image = Image.open(image_path).convert('L')
    image_np = np.array(image)
    return image_np

# 設定路徑點 (x, y)，並轉換為圖片中的像素位置
waypoints = [(0.2206, 0.1208),
            (1.2812, 0.0748),
            (2.3472, 0.129),
            (3.4053, 0.1631),
            (4.4468, 0.1421),
            (5.5032, 0.1996),
            (6.5372, 0.2315),
            (7.5948, 0.2499),
            (8.6607, 0.3331),
            (9.6811, 0.3973),
            (10.6847, 0.4349),
            (11.719, 0.4814),
            (12.7995, 0.5223),
            (13.8983, 0.515),
            (14.9534, 0.6193),
            (15.9899, 0.7217),
            (17.0138, 0.7653),
            (18.0751, 0.8058),
            (19.0799, 0.864),
            (20.1383, 0.936),
            (21.1929, 0.9923),
            (22.2351, 1.0279),
            (23.3374, 1.1122),
            (24.4096, 1.1694),
            (25.4817, 1.2437),
            (26.5643, 1.3221),
            (27.6337, 1.4294),
            (28.6643, 1.4471),
            (29.6839, 1.4987),
            (30.7, 1.58),
            (31.7796, 1.6339),
            (32.8068, 1.7283),
            (33.8596, 1.8004),
            (34.9469, 1.9665),
            (35.9883, 1.9812),
            (37.0816, 2.0237),
            (38.1077, 2.1291),
            (39.1405, 2.1418),
            (40.1536, 2.2273),
            (41.1599, 2.2473),
            (42.2476, 2.2927),
            (43.3042, 2.341),
            (44.4049, 2.39),
            (45.5091, 2.4284),
            (46.579, 2.5288),
            (47.651, 2.4926),
            (48.6688, 2.6072),
            (49.7786, 2.6338),
            (50.7942, 2.6644),
            (51.868, 2.7625),
            (52.9149, 2.8676),
            (54.0346, 2.9602),
            (55.0855, 2.9847),
            (56.1474, 3.1212),
            (57.2397, 3.2988),
            (58.2972, 3.5508),
            (59.1103, 4.1404),
            (59.6059, 5.1039),
            (59.6032, 6.2015),
            (59.4278, 7.212),
            (59.3781, 8.2782),
            (59.4323, 9.2866),
            (59.3985, 10.304),
            (59.3676, 11.3302),
            (59.3193, 12.3833),
            (59.359, 13.4472),
            (59.3432, 14.4652),
            (59.3123, 15.479),
            (59.1214, 16.4917),
            (58.7223, 17.4568),
            (57.8609, 18.1061),
            (56.8366, 18.3103),
            (55.7809, 18.0938),
            (54.7916, 17.707),
            (53.7144, 17.5087),
            (52.6274, 17.3683),
            (51.6087, 17.1364),
            (50.5924, 17.0295),
            (49.5263, 16.9058),
            (48.4514, 16.7769),
            (47.3883, 16.6701),
            (46.3186, 16.5403),
            (45.3093, 16.4615),
            (44.263, 16.299),
            (43.2137, 16.1486),
            (42.171, 16.0501),
            (41.1264, 16.0245),
            (40.171, 16.7172),
            (39.1264, 16.8428),
            (38.1122, 17.019),
            (37.2234, 16.5322),
            (36.6845, 15.6798),
            (36.3607, 14.7064),
            (35.5578, 13.9947),
            (34.5764, 13.7466),
            (33.5137, 13.6068),
            (32.4975, 13.5031),
            (31.5029, 13.3368),
            (30.4162, 13.1925),
            (29.3894, 13.067),
            (28.3181, 12.9541),
            (27.3195, 12.8721),
            (26.2852, 12.8035),
            (25.241, 12.6952),
            (24.1598, 12.6435),
            (23.0712, 12.5947),
            (21.9718, 12.5297),
            (20.9141, 12.4492),
            (19.8964, 12.3878),
            (18.7163, 12.32),
            (17.6221, 12.2928),
            (16.5457, 12.2855),
            (15.5503, 12.1534),
            (14.4794, 12.0462),
            (13.4643, 11.9637),
            (12.3466, 11.7943),
            (11.2276, 11.6071),
            (10.2529, 12.0711),
            (9.7942, 13.0066),
            (9.398, 13.9699),
            (8.6017, 14.7268),
            (7.4856, 14.8902),
            (6.5116, 14.4724),
            (5.4626, 14.1256),
            (4.3911, 13.9535),
            (3.3139, 13.8013),
            (2.2967, 13.7577),
            (1.2165, 13.7116),
            (0.1864, 13.6054),
            (-0.9592, 13.4747),
            (-2.0086, 13.352),
            (-3.0267, 13.3358),
            (-4.0117, 13.5304),
            (-5.0541, 13.8047),
            (-6.0953, 13.9034),
            (-7.1116, 13.8871),
            (-8.152, 13.8062),
            (-9.195, 13.7043),
            (-10.2548, 13.6152),
            (-11.234, 13.3289),
            (-11.9937, 12.6211),
            (-12.3488, 11.6585),
            (-12.4231, 10.6268),
            (-12.3353, 9.5915),
            (-12.2405, 8.5597),
            (-12.1454, 7.4974),
            (-12.0596, 6.4487),
            (-12.0537, 5.3613),
            (-12.0269, 4.2741),
            (-11.999, 3.2125),
            (-11.9454, 2.2009),
            (-11.7614, 1.1884),
            (-11.2675, 0.2385),
            (-10.5404, -0.58),
            (-9.4494, -0.8399),
            (-8.3965, -0.8367),
            (-7.3912, -0.6242),
            (-6.3592, -0.463)
            ]  

# 找到左右牆壁
def find_wall(grid, waypoint, prev_waypoint):
    # 將實際位置轉換為像素位置
    x = int(waypoint[0] * 20 + 2000)
    y = int(-waypoint[1] * 20 + 2000)

    if prev_waypoint is None:
        direction_vector = np.array([x - 2000, y - 2000])
    else:
        prev_x = int(prev_waypoint[0] * 20 + 2000)
        prev_y = int(-prev_waypoint[1] * 20 + 2000)
        direction_vector = np.array([x - prev_x, y - prev_y])
    # 將前進向量歸一化
    direction_vector = direction_vector / np.linalg.norm(direction_vector)

    # 計算垂直於前進方向的向量
    perp_vector_right = np.array([-direction_vector[1], direction_vector[0]])  # 右邊
    perp_vector_left = np.array([direction_vector[1], -direction_vector[0]])  # 左邊

    # 從當前點沿著垂直向量的方向找牆壁
    left_wall = find_wall_in_direction(grid, x, y, perp_vector_left, direction_vector)
    right_wall = find_wall_in_direction(grid, x, y, perp_vector_right, direction_vector)

    return left_wall, right_wall

# 在垂直方向上找牆壁，找第一個像素值小於 250 的點，並判定連續20個單位是否有牆
def find_wall_in_direction(grid, start_x, start_y, perp_vector,direction_vector, max_distance=60, wall_threshold=250):
    for i in range(max_distance):
        test_x = int(start_x + i * perp_vector[0])
        test_y = int(start_y + i * perp_vector[1])

        if test_x < 0 or test_y < 0 or test_x >= grid.shape[1] or test_y >= grid.shape[0]:
            break

        # 檢查牆壁 (像素值 < 250)
        if grid[test_y, test_x] < wall_threshold:
            if is_valid_wall(grid, test_x, test_y, direction_vector):
                # print(i,' : ',test_x , test_y)
                return test_x, test_y
    return None

# 驗證是否為有效牆壁（連續 20 像素的障礙物）
def is_valid_wall(grid, wall_x, wall_y, direction_vector, check_length=30, min_valid_count=1, wall_threshold=250):
    count = 0
    for i in range(check_length):
        # 使用前進向量來判斷牆壁的連續性
        test_x = int(wall_x + i * direction_vector[0])
        test_y = int(wall_y + i * direction_vector[1])

        # 檢查是否超出圖片邊界
        if test_x < 0 or test_y < 0 or test_x >= grid.shape[1] or test_y >= grid.shape[0]:
            break

        # 判斷牆壁像素值是否小於牆壁的閾值
        if grid[test_y, test_x] < wall_threshold:
            count += 1
    # 如果連續的牆壁像素數量超過最低要求，則認為是有效牆壁
    return True



# 計算牆壁間距、中心點
def calculate_wall_distance_and_center(left_wall, right_wall):
    if left_wall is None or right_wall is None:
        return None, None

    distance = np.sqrt((right_wall[0] - left_wall[0]) ** 2 + (right_wall[1] - left_wall[1]) ** 2)
    center_x = (right_wall[0] + left_wall[0]) / 2
    center_y = (right_wall[1] + left_wall[1]) / 2
    return distance, (center_x, center_y)

def save_to_csv(results, csv_path):
    # 確保結果非空
    if not results:
        print("Error: No results to save.")
        return

    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # 寫入表頭
        writer.writerow(['Waypoint X', 'Waypoint Y', 'Left Wall X', 'Left Wall Y', 'Right Wall X', 'Right Wall Y', 'Wall Distance', 'Center X', 'Center Y'])
        
        # 寫入每個結果
        for row in results:
            if row is not None and isinstance(row, (list, tuple)):
                newrow = [row[0][0],row[0][1],row[1][0],row[1][1],row[2][0],row[2][1],row[3],row[4][0],row[4][1]]
                writer.writerow(newrow)

# 主函數
def main():
    # 載入圖片為 NumPy 網格
    image_path = '/home/daniel/maps/my_map0924_2.png'
    grid = load_image_as_grid(image_path)

    # 保存結果的列表
    results = []

    prev_waypoint = None
    last_valid_result = None  # 儲存前一個有效的牆壁資訊

    for i, waypoint in enumerate(waypoints):
        # 對應每個路徑點，找到左右牆壁
        left_wall, right_wall = find_wall(grid, waypoint, prev_waypoint)

        # 如果左右牆壁都找到，計算牆壁間距和中心點
        if left_wall is not None and right_wall is not None:
            distance, center = calculate_wall_distance_and_center(left_wall, right_wall)
            result = [waypoint, left_wall, right_wall, distance, center]
            last_valid_result = result  # 更新最新的有效牆壁資訊
        else:
            # 如果無法找到牆壁，使用前一個路徑點的數據並進行等效位移
            if last_valid_result is not None:
                prev_waypoint = waypoints[i - 1] if i > 0 else (0, 0)
                prev_left_wall, prev_right_wall = last_valid_result[1], last_valid_result[2]

                print(prev_left_wall, prev_right_wall)
                # 計算位移向量
                delta_x = waypoint[0] * 20 - prev_waypoint[0] * 20
                delta_y = -waypoint[1] * 20 - (-prev_waypoint[1] * 20)

                print(delta_x, delta_y)
                # 進行牆壁等效位移
                new_left_wall = [prev_left_wall[0] + delta_x, prev_left_wall[1] + delta_y]
                new_right_wall = [prev_right_wall[0] + delta_x, prev_right_wall[1] + delta_y]

                # 計算新牆壁的間距和中心點
                distance, center = calculate_wall_distance_and_center(new_left_wall, new_right_wall)
                result = [waypoint, new_left_wall, new_right_wall, distance, center]

                # 即便使用遞補的牆壁資料，也將它設為最新的有效牆壁資料
                last_valid_result = result
            else:
                result = None  # 如果第一個點無牆壁資訊，則為 None

        results.append(result)
        prev_waypoint = waypoint  # 更新前一個路徑點

    # 保存結果到 CSV
    csv_path = '/home/daniel/maps/wall_data.csv'
    save_to_csv(results, csv_path)

if __name__ == "__main__":
    main()
