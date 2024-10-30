#!/usr/bin/env python3
import rospy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from geometry_msgs.msg import Twist, PointStamped
from sensor_msgs.msg import PointCloud2, Imu
from gazebo_msgs.srv import SetModelState, GetModelState
from gazebo_msgs.msg import ModelState, ContactsState
import sensor_msgs.point_cloud2 as pc2
from scipy.special import comb
from collections import namedtuple
import cv2
import open3d as o3d
import tf
from tf.transformations import quaternion_from_euler
import time
from torch.amp import GradScaler
import yaml
from PIL import Image

# 超參數
REFERENCE_DISTANCE_TOLERANCE = 0.65
MEMORY_SIZE = 10000
BATCH_SIZE = 256
GAMMA = 0.99
LEARNING_RATE = 0.0003
PPO_EPOCHS = 5
CLIP_PARAM = 0.2
PREDICTION_HORIZON = 400
CONTROL_HORIZON = 10

device = torch.device("cpu")
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class PrioritizedMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = [None] * capacity
        self.position = 0
        self.priorities = torch.zeros((capacity,), dtype=torch.float32).cuda()
        self.alpha = 0.6
        self.epsilon = 1e-5

    def add(self, state, action, reward, done, next_state):
        if state is None or action is None or reward is None or done is None or next_state is None:
            rospy.logwarn("Warning: Attempted to add None to memory, skipping entry.")
            return

        max_priority = self.priorities.max() if self.memory[self.position] is not None else torch.tensor(1.0, device=device)
        self.memory[self.position] = (
            torch.tensor(state, dtype=torch.float32, device=device),
            torch.tensor(action, dtype=torch.float32, device=device),
            torch.tensor(reward, dtype=torch.float32, device=device),
            torch.tensor(done, dtype=torch.float32, device=device),
            torch.tensor(next_state, dtype=torch.float32, device=device)
        )
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if self.position == 0:
            raise ValueError("No samples available in memory.")

        # Ensure all priorities are valid for sampling
        if len(self.memory) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]

        # Handle NaN in priorities
        if torch.isnan(priorities).any():
            priorities = torch.nan_to_num(priorities, nan=0.0)

        probabilities = priorities ** self.alpha
        total = probabilities.sum()

        if total > 0:
            probabilities /= total
        else:
            probabilities = torch.ones_like(probabilities) / len(probabilities)

        indices = torch.multinomial(probabilities, batch_size, replacement=False).cuda()
        samples = [self.memory[idx] for idx in indices if self.memory[idx] is not None]

        if len(samples) == 0 or any(sample is None for sample in samples):
            raise ValueError("Sampled None from memory.")

        weights = (len(self.memory) * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        batch = list(zip(*samples))
        states, actions, rewards, dones, next_states = batch

        # Ensure all states are 4D tensors
        states = [s if s.dim() == 4 else s.view(1, 3, 64, 64) for s in states]
        next_states = [ns if ns.dim() == 4 else ns.view(1, 3, 64, 64) for ns in next_states]

        return (
            torch.stack(states).to(device),
            torch.stack(actions).to(device),
            torch.stack(rewards).to(device),
            torch.stack(dones).to(device),
            torch.stack(next_states).to(device),
            indices,
            weights.to(device)
        )

    def update_priorities(self, batch_indices, batch_priorities):
        # 確保每個 priority 是單一標量
        for idx, priority in zip(batch_indices, batch_priorities):
            # 如果 priority 是 numpy 陣列，檢查其 size
            if priority.size > 1:
                priority = priority[0]
            self.priorities[idx] = priority.item() + self.epsilon

    def clear(self):
        self.position = 0
        self.memory = [None] * self.capacity
        self.priorities = torch.zeros((self.capacity,), dtype=torch.float32).cuda()

class GazeboEnv:
    def __init__(self, model):
        rospy.init_node('gazebo_rl_agent', anonymous=True)
        self.model = model
        self.pub_cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.pub_imu = rospy.Publisher('/imu/data', Imu, queue_size=10)
        self.sub_scan = rospy.Subscriber('/velodyne_points', PointCloud2, self.scan_callback)
        self.sub_collision_chassis = rospy.Subscriber('/my_robot/bumper_data', ContactsState, self.collision_callback)
        self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.listener = tf.TransformListener()
        self.action_space = 2
        self.observation_space = (3, 64, 64)
        self.state = np.zeros(self.observation_space)
        self.done = False
        self.target_x = -5.3334
        self.target_y = -0.3768
        self.waypoints = self.generate_waypoints()
        self.current_waypoint_index = 0
        self.last_twist = Twist()
        self.epsilon = 0.05
        self.collision_detected = False
        self.previous_robot_position = None  # 初始化 previous_robot_position 為 None
        self.previous_distance_to_goal = None  # 初始化 previous_distance_to_goal 為 None

        self.lidar_data = None  # 初始化 lidar_data，避免 AttributeError

        self.max_no_progress_steps = 10
        self.no_progress_steps = 0
        
        # 新增屬性，標記是否已計算過優化路徑
        self.optimized_waypoints_calculated = False
        self.optimized_waypoints = []  # 儲存優化後的路徑點

        self.waypoint_failures = {i: 0 for i in range(len(self.waypoints))}

        # 加载SLAM地圖
        self.load_slam_map('/home/daniel/maps/my_map0924.yaml')

        self.optimize_waypoints_with_a_star()
        
    def load_slam_map(self, yaml_path):
        # 讀取 YAML 檔案
        with open(yaml_path, 'r') as file:
            map_metadata = yaml.safe_load(file)
            self.map_origin = map_metadata['origin']  # 地圖原點
            self.map_resolution = map_metadata['resolution']  # 地圖解析度
            png_path = map_metadata['image'].replace(".pgm", ".png")  # 修改為png檔案路徑
            
            # 使用 PIL 讀取PNG檔
            png_image = Image.open(png_path).convert('L')
            self.slam_map = np.array(png_image)  # 轉為NumPy陣列


    def generate_waypoints(self):
        waypoints = [
            (0.2206, 0.1208),
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
            (-6.3592, -0.463),
            (self.target_x, self.target_y)
        ]
        return waypoints
    
    def gazebo_to_image_coords(self, gazebo_x, gazebo_y):
        img_x = 2000 + gazebo_x * 20
        img_y = 2000 - gazebo_y * 20
        return int(img_x), int(img_y)

    def image_to_gazebo_coords(self, img_x, img_y):
        gazebo_x = (img_x - 2000) / 20
        gazebo_y = (2000 - img_y) / 20
        return gazebo_x, gazebo_y

    def a_star_optimize_waypoint(self, png_image, start_point, goal_point, grid_size=50):
        """
        A* 算法對 50x50 的正方形內進行路徑優化
        """
        # 使用 self.gazebo_to_image_coords 而不是 gazebo_to_image_coords
        img_start_x, img_start_y = self.gazebo_to_image_coords(*start_point)

        img_goal_x, img_goal_y = self.gazebo_to_image_coords(*goal_point)

        best_f_score = float('inf')
        best_point = (img_start_x, img_start_y)

        for x in range(img_start_x - grid_size // 2, img_start_x + grid_size // 2):
            for y in range(img_start_y - grid_size // 2, img_start_y + grid_size // 2):
                if not (0 <= x < png_image.shape[1] and 0 <= y < png_image.shape[0]):
                    continue

                g = np.sqrt((x - img_start_x) ** 2 + (y - img_start_y) ** 2)
                h = np.sqrt((x - img_goal_x) ** 2 + (y - img_goal_y) ** 2)

                unwalkable_count = np.sum(png_image[max(0, y - grid_size // 2):min(y + grid_size // 2, png_image.shape[0]),
                                                    max(0, x - grid_size // 2):min(x + grid_size // 2, png_image.shape[1])] < 250)

                f = g + h + unwalkable_count * 2

                if f < best_f_score:
                    best_f_score = f
                    best_point = (x, y)

        # 使用 self.image_to_gazebo_coords 而不是 image_to_gazebo_coords
        optimized_gazebo_x, optimized_gazebo_y = self.image_to_gazebo_coords(*best_point)

        return optimized_gazebo_x, optimized_gazebo_y


    def optimize_waypoints_with_a_star(self):
        """
        使用 A* 算法來優化路徑點，但僅在尚未計算過時執行
        """
        if self.optimized_waypoints_calculated:
            rospy.loginfo("Using previously calculated optimized waypoints.")
            self.waypoints = self.optimized_waypoints  # 使用已計算的優化路徑
            return

        rospy.loginfo("Calculating optimized waypoints for the first time using A*.")
        optimized_waypoints = []
        for i in range(len(self.waypoints) - 1):
            start_point = (self.waypoints[i][0], self.waypoints[i][1])
            goal_point = (self.waypoints[i + 1][0], self.waypoints[i + 1][1])
            optimized_point = self.a_star_optimize_waypoint(self.slam_map, start_point, goal_point)
            optimized_waypoints.append(optimized_point)

        # 最後一個終點加入到優化後的路徑點列表中
        optimized_waypoints.append(self.waypoints[-1])
        
        self.optimized_waypoints = optimized_waypoints
        self.waypoints = optimized_waypoints
        self.optimized_waypoints_calculated = True  # 設定標記，表示已計算過

    def is_in_blind_spot(self, x, y, blind_spot_length=3.36):
        """
        檢查給定的點是否位於車輛的盲區內
        """
        blind_spot_edges = self.calculate_blind_spot_edge_distances()
        min_distance_to_edges = min(blind_spot_edges)

        # 如果距離小於盲區長度，則該點在盲區內
        return min_distance_to_edges < blind_spot_length

    # 用於計算當前點與最近障礙物之間的距離
    def calculate_distance_to_nearest_obstacle(self, x, y):
        min_distance = float('inf')
        for point in pc2.read_points(self.lidar_data, field_names=("x", "y", "z"), skip_nans=True):
            distance = np.sqrt((point[0] - x) ** 2 + (point[1] - y) ** 2)
            if distance < min_distance:
                min_distance = distance
        return min_distance
    
    def bezier_curve(self, waypoints, n_points=100):
        waypoints = np.array(waypoints)
        n = len(waypoints) - 1

        def bernstein_poly(i, n, t):
            return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))

        t = np.linspace(0.0, 1.0, n_points)
        curve = np.zeros((n_points, 2))

        for i in range(n + 1):
            curve += np.outer(bernstein_poly(i, n, t), waypoints[i])

        return curve

    def is_point_near_obstacle(self, x, y, threshold=0.25):
        if self.lidar_data is None:
            rospy.logwarn("LiDAR data is not available yet.")
            return False

        min_distance_to_obstacle = float('inf')
        for point in pc2.read_points(self.lidar_data, field_names=("x", "y", "z"), skip_nans=True):
            distance = np.sqrt((point[0] - x)**2 + (point[1] - y)**2)
            if distance < min_distance_to_obstacle:
                min_distance_to_obstacle = distance

        return min_distance_to_obstacle < threshold
    
    def calculate_blind_spot_edge_distances(self):
        # 计算车辆的边缘距离，同时考虑盲区
        blind_spot_length = 3.36  # 根据上面的计算结果，盲区长度为 3.36 米
        vehicle_edges = [
            (1.0, 0),    # 前方边缘
            (-1.0, 0),   # 后方边缘
            (0, 0.5),    # 左侧边缘
            (0, -0.5)    # 右侧边缘
        ]
        
        edge_distances = []
        for edge in vehicle_edges:
            edge_x, edge_y = edge
            distance = self.calculate_distance_to_nearest_obstacle(edge_x, edge_y)
            if distance < blind_spot_length:
                distance = blind_spot_length  # 如果障碍物在盲区内，设定为盲区距离
            edge_distances.append(distance)
        
        return edge_distances

    def collision_callback(self, data):
        if len(data.states) > 0:
            self.collision_detected = True
            rospy.loginfo("Collision detected!")
        else:
            self.collision_detected = False

    def is_collision_detected(self):
        return self.collision_detected

    def generate_imu_data(self):
        imu_data = Imu()
        imu_data.header.stamp = rospy.Time.now()
        imu_data.header.frame_id = 'chassis'

        imu_data.linear_acceleration.x = np.random.normal(0, 0.1)
        imu_data.linear_acceleration.y = np.random.normal(0, 0.1)
        imu_data.linear_acceleration.z = np.random.normal(9.81, 0.1)
        
        imu_data.angular_velocity.x = np.random.normal(0, 0.01)
        imu_data.angular_velocity.y = np.random.normal(0, 0.01)
        imu_data.angular_velocity.z = np.random.normal(0, 0.01)

        robot_x, robot_y, robot_yaw = self.get_robot_position()
        quaternion = quaternion_from_euler(0.0, 0.0, robot_yaw)
        imu_data.orientation.x = quaternion[0]
        imu_data.orientation.y = quaternion[1]
        imu_data.orientation.z = quaternion[2]
        imu_data.orientation.w = quaternion[3]

        return imu_data

    def scan_callback(self, data):
        # 确保数据有效
        if data is not None and self.is_valid_data(data):
            self.lidar_data = data
            combined_pcl_data = self.accumulate_lidar_data([data])

            if self.current_waypoint_index >= len(self.waypoints):
                rospy.logwarn("current_waypoint_index out of range, resetting to last valid index.")
                self.current_waypoint_index = len(self.waypoints) - 1

            current_waypoint_x, current_waypoint_y = self.waypoints[self.current_waypoint_index]

            self.state = self.generate_occupancy_grid(combined_pcl_data, current_waypoint_x, current_waypoint_y)

            self.future_waypoints = self.waypoints[self.current_waypoint_index+1 : min(self.current_waypoint_index+1+7, len(self.waypoints))]

        else:
            rospy.logwarn("Received invalid or empty LiDAR data")

    def is_valid_data(self, data):
        for point in pc2.read_points(data, field_names=("x", "y", "z"), skip_nans=True):
            if point[0] != 0.0 or point[1] != 0.0 or point[2] != 0.0:
                return True
        return False

    def accumulate_lidar_data(self, lidar_buffer):
        combined_points = []
        for data in lidar_buffer:
            points_list = []
            for point in pc2.read_points(data, field_names=("x", "y", "z"), skip_nans=True):
                if point[0] == 0.0 and point[1] == 0.0 and point[2] == 0.0:
                    continue

                point_transformed = self.transform_point(point, 'my_robot/velodyne', 'chassis')
                points_list.append(point_transformed)
            combined_points.extend(points_list)

        if len(combined_points) == 0:
            rospy.logwarn("No points collected after transformation")
            return PointCloud2()

        combined_pcl = o3d.geometry.PointCloud()
        combined_pcl.points = o3d.utility.Vector3dVector(np.array(combined_points, dtype=np.float32))

        return self.convert_open3d_to_ros(combined_pcl)

    def transform_point(self, point, from_frame, to_frame):
        try:
            now = rospy.Time.now()
            self.listener.waitForTransform(to_frame, from_frame, now, rospy.Duration(1.0))
            
            point_stamped = PointStamped()
            point_stamped.header.frame_id = from_frame
            point_stamped.header.stamp = now
            point_stamped.point.x = point[0]
            point_stamped.point.y = point[1]
            point_stamped.point.z = point[2]
            
            point_transformed = self.listener.transformPoint(to_frame, point_stamped)
            return [point_transformed.point.x, point_transformed.point.y, point_transformed.point.z]
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logerr(f"Transform failed: {e}")
            return [point[0], point[1], point[2]]

    def convert_open3d_to_ros(self, cloud):
        header = rospy.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'velodyne'
        points = np.asarray(cloud.points)
        return pc2.create_cloud_xyz32(header, points)

    def generate_occupancy_grid(self, lidar_data, waypoint_x, waypoint_y, grid_size=0.05, map_size=100):
        if lidar_data is None or not isinstance(lidar_data, PointCloud2):
            rospy.logwarn("No LiDAR data available, returning empty occupancy grid.")
            return np.zeros((4, 64, 64))

        grid_size = float(grid_size)
        map_size_in_grids = int(map_size / grid_size)
        grid = np.zeros((map_size_in_grids, map_size_in_grids))

        x_min, x_max = -map_size / 2, map_size / 2
        y_min, y_max = -map_size / 2, map_size / 2

        for point in pc2.read_points(lidar_data, field_names=("x", "y", "z"), skip_nans=True):
            x = point[0]
            y = point[1]

            if x_min <= x <= x_max and y_min <= y <= y_max:
                ix = int((x - x_min) / grid_size)
                iy = int((y - y_min) / grid_size)

                if 0 <= ix < grid.shape[1] and 0 <= iy < grid.shape[0]:
                    grid[iy, ix] = 1
                else:
                    rospy.logwarn(f"Index out of bounds after filtering: ix={ix}, iy={iy}")
            else:
                rospy.logwarn(f"LiDAR point out of bounds after filtering: x={x}, y={y}")

        grid = cv2.resize(grid, (64, 64), interpolation=cv2.INTER_LINEAR)

        waypoint_grid = np.zeros((2, 64, 64))
        waypoint_grid[0, :, :] = waypoint_x
        waypoint_grid[1, :, :] = waypoint_y

        occupancy_grid = np.vstack([grid[np.newaxis, :, :], waypoint_grid])

        return occupancy_grid

    def step(self, action):
        reward = 0
        robot_x, robot_y, robot_yaw = self.get_robot_position()

        # 計算當前機器人位置與所有 waypoints 的距離，並找到距離最近的 waypoint 的索引
        distances = [np.linalg.norm([robot_x - wp_x, robot_y - wp_y]) for wp_x, wp_y in self.waypoints]
        closest_index = np.argmin(distances)
        self.current_waypoint_index = closest_index

        # 確認最近的 waypoint 是否與目標位置相同
        current_waypoint_x, current_waypoint_y = self.waypoints[self.current_waypoint_index]
        distance_to_goal = np.linalg.norm([robot_x - self.target_x, robot_y - self.target_y])
        
        # 如果達到最終目標點
        if distance_to_goal < REFERENCE_DISTANCE_TOLERANCE:
            self.done = True
            reward += 10000
        else:
            # 根據距離的變化來給獎勵（具體獎勵設置可根據需求進行調整）
            reward += max(0, 10 - distance_to_goal * 2)

        print("Current waypoint index:", self.current_waypoint_index)

        if self.is_collision_detected():
            rospy.loginfo("Collision detected, resetting environment.")
            reward = -2000.0
            self.reset()
            return self.state, reward, True, {}

        distance_moved = np.linalg.norm([robot_x - self.previous_robot_position[0], robot_y - self.previous_robot_position[1]]) if self.previous_robot_position else 0
        
        self.previous_robot_position = (robot_x, robot_y)

        use_deep_rl_control = any(
            self.waypoint_failures.get(i, 0) > 1
            for i in range(max(0, self.current_waypoint_index - 3), min(len(self.waypoints), self.current_waypoint_index + 4))
        )

        if use_deep_rl_control:
            print('operate by RL')
            if isinstance(self.state, np.ndarray):
                self.state = torch.tensor(self.state, dtype=torch.float32).view(1, 3, 64, 64).to(device)
            elif self.state.dim() != 4:
                self.state = self.state.view(1, 3, 64, 64)

            action = self.model.act(self.state)
            print("RL Action output:", action)
            linear_speed = np.clip(action[0, 0].item(), -2.0, 2.0)
            steer_angle = np.clip(action[0, 1].item(), -0.6, 0.6)
        else:
            # 計算當前位置與 current_waypoint_index 的距離
            robot_x, robot_y, _ = self.get_robot_position()
            current_waypoint_x, current_waypoint_y = self.waypoints[self.current_waypoint_index]
            distance_to_waypoint = np.linalg.norm([robot_x - current_waypoint_x, robot_y - current_waypoint_y])

            if distance_moved < 0.05:
                self.no_progress_steps += 1
                if self.no_progress_steps >= self.max_no_progress_steps:
                    self.waypoint_failures[self.current_waypoint_index] += 1
                    print('failure at point', self.current_waypoint_index)
                    rospy.loginfo("No progress detected, resetting environment.")
                    reward = -2000.0
                    self.reset()
                    return self.state, reward, True, {}
            else:
                self.no_progress_steps = 0


            action = self.calculate_action_pure_pursuit()
            print("A* Action output:", action)
            linear_speed = np.clip(action[0], -2.0, 2.0)
            steer_angle = np.clip(action[1], -0.6, 0.6)

        print(linear_speed)
        print(steer_angle)

        twist = Twist()
        twist.linear.x = linear_speed
        twist.angular.z = steer_angle
        self.pub_cmd_vel.publish(twist)

        imu_data = self.generate_imu_data()
        self.pub_imu.publish(imu_data)

        rospy.sleep(0.1)

        reward, _ = self.calculate_reward(current_waypoint_x, current_waypoint_y)

        return self.state, reward, self.done, {}


    def reset(self):
        # 設置初始機器人位置和姿態
        yaw = -0.0053
        quaternion = quaternion_from_euler(0.0, 0.0, yaw)
        state_msg = ModelState()
        state_msg.model_name = 'my_robot'
        state_msg.pose.position.x = 0.2206
        state_msg.pose.position.y = 0.1208
        state_msg.pose.position.z = 2.2
        state_msg.pose.orientation.x = quaternion[0]
        state_msg.pose.orientation.y = quaternion[1]
        state_msg.pose.orientation.z = quaternion[2]
        state_msg.pose.orientation.w = quaternion[3]

        state_msg.twist.linear.x = 0.0
        state_msg.twist.linear.y = 0.0
        state_msg.twist.linear.z = 0.0
        state_msg.twist.angular.x = 0.0
        state_msg.twist.angular.y = 0.0
        state_msg.twist.angular.z = 0.0

        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            self.set_model_state(state_msg)
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")

        rospy.sleep(0.5)

        # 確保使用優化過的路徑點
        if hasattr(self, 'waypoints') and self.waypoints:
            rospy.loginfo("Using optimized waypoints for reset.")
        else:
            rospy.loginfo("No optimized waypoints found, generating new waypoints.")
            self.waypoints = self.generate_waypoints()

        self.current_waypoint_index = 0
        self.done = False

        empty_lidar_data = PointCloud2()
        current_waypoint_x, current_waypoint_y = self.waypoints[self.current_waypoint_index]
        self.state = self.generate_occupancy_grid(empty_lidar_data, current_waypoint_x, current_waypoint_y)

        self.last_twist = Twist()
        self.pub_cmd_vel.publish(self.last_twist)

        imu_data = self.generate_imu_data()
        self.pub_imu.publish(imu_data)

        self.previous_yaw_error = 0
        self.no_progress_steps = 0
        self.previous_distance_to_goal = None
        self.collision_detected = False

        # Ensure the state is 4D tensor
        if isinstance(self.state, np.ndarray):
            self.state = torch.tensor(self.state, dtype=torch.float32).view(1, 3, 64, 64).to(device)
        elif self.state.dim() != 4:
            self.state = self.state.view(1, 3, 64, 64)

        return self.state


    def calculate_reward(self, target_x, target_y):
        robot_x, robot_y, robot_yaw = self.get_robot_position()
        reward = 0
        done = False

        # # 將機器人的座標轉換為地圖上的坐標
        # map_x = int((robot_x - self.map_origin[0]) / self.map_resolution)
        # map_y = int((robot_y - self.map_origin[1]) / self.map_resolution)

        # map_y = 4000 - map_y

        # # 檢查機器人座標是否在地圖範圍內
        # if 0 <= map_x < 4000 and 0 <= map_y < 4000:
        #     # 根據機器人在地圖上的位置給予不同的獎勵或懲罰
        #     if self.slam_map[map_y, map_x] >= 250:
        #         reward += 10  # 白色區域 (可走區域)，給予較大獎勵
        #         print("On the road +10")
        #     elif self.slam_map[map_y, map_x] <= 190:
        #         reward -= 100  # 黑色區域 (障礙物)，給予懲罰
        #         print("Hit obstacle -100")
        #         done = True  # 碰到障礙物時，直接結束
        #     elif self.slam_map[map_y, map_x] >190 and self.slam_map[map_y, map_x]<250:
        #         reward += 5  # 灰色區域 (未知區域)，給予適當獎勵
        #         print("In unknown area +5")
        #     else:
        #         reward -= 10  # 偏離地圖或無法識別的區域，給予懲罰
        #         print("Not on the road -10")
        # else:
        #     reward -= 20  # 機器人在地圖範圍外，給予懲罰
        #     print("Out of map bounds -20")

        # 計算方向誤差的獎勵
        direction_to_target = np.arctan2(target_y - robot_y, target_x - robot_x)
        yaw_diff = np.abs(direction_to_target - robot_yaw)
        yaw_diff = np.arctan2(np.sin(yaw_diff), np.cos(yaw_diff))  # 確保角度在[-pi, pi]範圍內
        reward += 10*max(0, 5 - yaw_diff * 5)  # 誤差越小，獎勵越大

        # 計算距離目標點的獎勵
        distance_to_goal = np.sqrt((target_x - robot_x) ** 2 + (target_y - robot_y) ** 2)
        reward += max(0, 10 - distance_to_goal * 2)  # 距離越近，獎勵越高

        # 增加遠離障礙物的安全性獎勵
        if not self.is_point_near_obstacle(robot_x, robot_y, threshold=0.3):
            reward += 50  # 距離障礙物較遠時，給予安全性獎勵
        else:
            reward -= 500  # 距離障礙物較近時，給予懲罰

        return reward, done





    def get_robot_position(self):
        try:
            rospy.wait_for_service('/gazebo/get_model_state')
            model_state = self.get_model_state('my_robot', '')
            robot_x = model_state.pose.position.x
            robot_y = model_state.pose.position.y

            orientation_q = model_state.pose.orientation
            yaw = self.quaternion_to_yaw(orientation_q)
            return robot_x, robot_y, yaw
        except rospy.ServiceException as e:
            rospy.logerr(f"Get model state service call failed: %s", e)
            return 0, 0, 0

    def quaternion_to_yaw(self, orientation_q):
        siny_cosp = 2 * (orientation_q.w * orientation_q.z + orientation_q.x * orientation_q.y)
        cosy_cosp = 1 - 2 * (orientation_q.y * orientation_q.y + orientation_q.z * orientation_q.z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return yaw

    def calculate_action_pure_pursuit(self):
        robot_x, robot_y, robot_yaw = self.get_robot_position()

        # 動態調整前視距離（lookahead distance）
        linear_speed = np.linalg.norm([self.last_twist.linear.x, self.last_twist.linear.y])
        lookahead_distance = 2.0 + 0.5 * linear_speed  # 根據速度調整前視距離

        # 定義角度範圍，以當前車輛的yaw為中心
        angle_range = np.deg2rad(40)  # ±40度的範圍
        closest_index = None
        min_distance = float('inf')

        # 尋找該範圍內的最近路徑點
        for i in range(self.current_waypoint_index, len(self.waypoints)):
            wp_x, wp_y = self.waypoints[i]
            dist_to_wp = np.linalg.norm([wp_x - robot_x, wp_y - robot_y])
            direction_to_wp = np.arctan2(wp_y - robot_y, wp_x - robot_x)

            # 計算該點相對於當前車輛朝向的角度
            yaw_diff = direction_to_wp - robot_yaw
            yaw_diff = np.arctan2(np.sin(yaw_diff), np.cos(yaw_diff))  # 確保角度在[-pi, pi]範圍內

            # 如果點位於yaw ± 35度範圍內，並且距離更近
            if np.abs(yaw_diff) < angle_range and dist_to_wp < min_distance:
                min_distance = dist_to_wp
                closest_index = i

        # 如果沒有找到符合條件的點，則繼續使用原始最近點
        if closest_index is None:
            closest_index = self.find_closest_waypoint(robot_x, robot_y)

        target_index = closest_index

        # 根據前視距離選擇參考的路徑點
        cumulative_distance = 0.0
        for i in range(closest_index, len(self.waypoints)):
            wp_x, wp_y = self.waypoints[i]
            dist_to_wp = np.linalg.norm([wp_x - robot_x, wp_y - robot_y])
            cumulative_distance += dist_to_wp
            if cumulative_distance >= lookahead_distance:
                target_index = i
                break
        # 獲取前視點座標
        target_x, target_y = self.waypoints[target_index]

        # 計算前視點的方向
        direction_to_target = np.arctan2(target_y - robot_y, target_x - robot_x)
        yaw_error = direction_to_target - robot_yaw
        yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))  # 確保角度在[-pi, pi]範圍內

        # 根據角度誤差調整速度
        if np.abs(yaw_error) > 0.3:
            linear_speed = 0.5
        elif np.abs(yaw_error) > 0.1:
            linear_speed = 1.0
        else:
            linear_speed = 1.5

        # 使用PD控制器調整轉向角度
        kp, kd = self.adjust_control_params(linear_speed)
        previous_yaw_error = getattr(self, 'previous_yaw_error', 0)
        current_yaw_error_rate = yaw_error - previous_yaw_error
        steer_angle = kp * yaw_error + kd * current_yaw_error_rate
        steer_angle = np.clip(steer_angle, -0.6, 0.6)

        self.previous_yaw_error = yaw_error

        return np.array([linear_speed, steer_angle])


    def find_closest_waypoint(self, x, y):
        # 找到與當前位置最接近的路徑點
        min_distance = float('inf')
        closest_index = 0
        for i, (wp_x, wp_y) in enumerate(self.waypoints):
            dist = np.linalg.norm([wp_x - x, wp_y - y])
            if dist < min_distance:
                min_distance = dist
                closest_index = i
        return closest_index
    
    def adjust_control_params(self, linear_speed):
        if linear_speed <= 0.5:
            kp = 0.5
            kd = 0.2
        elif linear_speed <= 1.0:
            kp = 0.4
            kd = 0.3
        else:
            kp = 0.3
            kd = 0.4
        return kp, kd

class ActorCritic(nn.Module):
    def __init__(self, observation_space, action_space):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(128)

        self._to_linear = self._get_conv_output_size(observation_space)

        self.fc1 = nn.Linear(self._to_linear, 256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, 128)

        self.actor = nn.Linear(128, action_space)
        self.critic = nn.Linear(128, 1)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_space))
        

    def _get_conv_output_size(self, shape):
        x = torch.zeros(1, *shape)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = x.view(1, -1)
        return x.size(1)

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)

        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)

        action_mean = self.actor(x)
        action_log_std = self.actor_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        value = self.critic(x)

        return action_mean, action_std, value

    def act(self, state):
        # 確保 state 是 tensor，如果是 numpy，轉換為 tensor
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32).to(device)

        # 去除多餘維度直到 <= 4
        while state.dim() > 4:
            state = state.squeeze(0)

        # 添加缺少的維度直到 = 4
        while state.dim() < 4:
            state = state.unsqueeze(0)

        # 最終確認 state 是 4D
        if state.dim() != 4:
            raise ValueError(f"Expected state to be 4D, but got {state.dim()}D")

        action_mean, action_std, _ = self(state)
        action = action_mean + action_std * torch.randn_like(action_std)
        action = torch.tanh(action)
        max_action = torch.tensor([2.0, 0.6], device=action.device)
        min_action = torch.tensor([-2.0, -0.6], device=action.device)
        action = min_action + (action + 1) * (max_action - min_action) / 2
        return action.detach()

    def evaluate(self, state, action):
        action_mean, action_std, value = self(state)
        dist = torch.distributions.Normal(action_mean, action_std)
        action_log_probs = dist.log_prob(action).sum(dim=-1, keepdim=True)
        dist_entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return action_log_probs, value, dist_entropy

def ppo_update(ppo_epochs, env, model, optimizer, memory, scaler):
    for _ in range(ppo_epochs):
        state_batch, action_batch, reward_batch, done_batch, next_state_batch, indices, weights = memory.sample(BATCH_SIZE)
        
        adjusted_lr = LEARNING_RATE * (weights.mean().item() + 1e-3)
        for param_group in optimizer.param_groups:
            param_group['lr'] = adjusted_lr

        with torch.no_grad():
            old_log_probs, _, _ = model.evaluate(state_batch, action_batch)
        old_log_probs = old_log_probs.detach()

        for _ in range(PPO_EPOCHS):
            with torch.amp.autocast('cuda'):
                log_probs, state_values, dist_entropy = model.evaluate(state_batch, action_batch)
                advantages = reward_batch + (1 - done_batch) * GAMMA * model(next_state_batch)[2].detach() - state_values

                ratio = (log_probs - old_log_probs).exp()
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - CLIP_PARAM, 1 + CLIP_PARAM) * advantages

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(state_values, reward_batch + (1 - done_batch) * GAMMA * model(next_state_batch)[2].detach())
                entropy_loss = -0.01 * dist_entropy.mean()  # 添加熵正则项
                loss = actor_loss + 0.5 * critic_loss + entropy_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            priorities = (advantages + 1e-5).abs().detach().cpu().numpy()
            memory.update_priorities(indices, priorities)

def main():
    env = GazeboEnv(None)
    model = ActorCritic(env.observation_space, env.action_space).to(device)
    env.model = model
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scaler = GradScaler('cuda')
    memory = PrioritizedMemory(MEMORY_SIZE)

    model_path = "/home/daniel/catkin_ws/src/my_robot_control/scripts/saved_model_ppo.pth"
    best_model_path = "/home/daniel/catkin_ws/src/my_robot_control/scripts/best_model.pth"
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded existing model.")
    else:
        print("Created new model.")

    num_episodes = 1000000
    best_test_reward = -np.inf

    for e in range(num_episodes):
        if not env.optimized_waypoints_calculated:
            env.optimize_waypoints_with_a_star()

        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        total_reward = 0
        start_time = time.time()

        for time_step in range(1500):
            action = model.act(state)
            action_np = action.detach().cpu().numpy()
            next_state, reward, done, _ = env.step(action_np)
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)
            memory.add(state.cpu().numpy(), action_np, reward, done, next_state.cpu().numpy())
            state = next_state
            total_reward += reward

            elapsed_time = time.time() - start_time
            if done or elapsed_time > 240:
                if elapsed_time > 240:
                    reward -= 1000.0
                    print(f"Episode {e} failed at time step {time_step}: time exceeded 240 sec.")
                break

        ppo_update(PPO_EPOCHS, env, model, optimizer, memory, scaler)
        memory.clear()

        print(f"Episode {e}, Total Reward: {total_reward}")

        if total_reward > best_test_reward:
            best_test_reward = total_reward
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with reward: {best_test_reward}")

        if e % 20 == 0:
            torch.save(model.state_dict(), model_path)
            print(f"Model saved after {e} episodes.")

        rospy.sleep(1.0)

    torch.save(model.state_dict(), model_path)
    print("Final model saved.")

if __name__ == '__main__':
    main()