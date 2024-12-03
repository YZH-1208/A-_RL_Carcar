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
import random
from sklearn.cluster import KMeans
import csv
import datetime
from torch.optim.lr_scheduler import StepLR
import wandb
import torch

# 超參數
REFERENCE_DISTANCE_TOLERANCE = 0.65
MEMORY_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.99
LEARNING_RATE = 0.0003
PPO_EPOCHS = 5
CLIP_PARAM = 0.2
PREDICTION_HORIZON = 400
CONTROL_HORIZON = 10

device = torch.device("cpu")
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

def grid_filter(obstacles, grid_size=0.5):
    obstacles = np.array(obstacles)
    # 按照 grid_size 取整
    grid_indices = (obstacles // grid_size).astype(int)
    # 找到唯一的网格
    unique_indices = np.unique(grid_indices, axis=0)
    # 返回网格中心点
    filtered_points = unique_indices * grid_size + grid_size / 2
    return filtered_points

class PrioritizedMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = [None] * capacity
        self.position = 0
        self.priorities = torch.zeros((capacity,), dtype=torch.float32).cuda()
        self.alpha = 0.6
        self.beta = 0.4
        self.beta_increment = 0.001
        self.epsilon = 1e-6
        self.max_priority = 1.0

    def add(self, state, action, reward, done, next_state):
        print(f"Memory size: {len([x for x in self.memory if x is not None])}")
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32)
        if isinstance(next_state, np.ndarray):
            next_state = torch.tensor(next_state, dtype=torch.float32)
        
        # 调整维度
        if state.dim() == 5:
            state = state.squeeze(1)
        if next_state.dim() == 5:
            next_state = next_state.squeeze(1)
        
        # 确保状态有 4 个维度
        if state.dim() == 3:
            state = state.unsqueeze(0)
        if next_state.dim() == 3:
            next_state = next_state.unsqueeze(0)

        self.memory[self.position] = (
            state.to(device),
            torch.tensor(action, dtype=torch.float32, device=device),
            torch.tensor(reward, dtype=torch.float32, device=device),
            torch.tensor(done, dtype=torch.float32, device=device),
            next_state.to(device)
        )
        self.position = (self.position + 1) % self.capacity
        self.priorities[self.position] = self.max_priority


    def sample(self, batch_size):
        valid_samples = [i for i, x in enumerate(self.memory) if x is not None]
        if len(valid_samples) < batch_size:
            raise ValueError(f"Not enough valid samples. Have {len(valid_samples)}, need {batch_size}")

        # 更新 beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # 计算采样概率
        priorities = self.priorities[:len(valid_samples)]
        probs = priorities ** self.alpha
        probs = probs / probs.sum()

        # 采样索引
        indices = torch.multinomial(probs, batch_size, replacement=False)
        
        # 计算重要性采样权重
        weights = (len(valid_samples) * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()

        # 获取样本
        batch = [self.memory[idx] for idx in indices]
        states, actions, rewards, dones, next_states = zip(*batch)

        # 确保状态维度正确
        states = torch.stack(states).to(device)
        next_states = torch.stack(next_states).to(device)
        actions = torch.stack(actions).to(device)
        rewards = torch.stack(rewards).to(device)
        dones = torch.stack(dones).to(device)

        print(f"Sampled {batch_size} samples from memory.")
        return states, actions, rewards, dones, next_states, indices, weights.to(device)



    def update_priorities(self, indices, priorities):
        priorities = torch.tensor(priorities, dtype=self.priorities.dtype, device=self.priorities.device)
        assert len(indices) == len(priorities), \
            f"Shape mismatch: indices {indices.shape}, priorities {priorities.shape}"
        self.priorities[indices] = priorities




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
        self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.listener = tf.TransformListener()
        self.action_space = 2
        self.observation_space = (3, 64, 64)
        self.state = np.zeros(self.observation_space)
        self.done = False
        self.target_x = 59.7436
        self.target_y = 3.6285  
        self.waypoints = self.generate_waypoints()
        self.waypoint_distances = self.calculate_waypoint_distances()   # 計算一整圈機器任要奏的大致距離
        self.current_waypoint_index = 0
        self.last_twist = Twist()
        self.epsilon = 0.05
        self.previous_robot_position = None  # 初始化 previous_robot_position 為 None
        self.previous_distance_to_goal = None  # 初始化 previous_distance_to_goal 為 None

        self.max_no_progress_steps = 10
        self.no_progress_steps = 0
        
        # 新增屬性，標記是否已計算過優化路徑
        self.optimized_waypoints_calculated = False
        self.optimized_waypoints = []  # 儲存優化後的路徑點

        self.waypoint_failures = {i: 0 for i in range(len(self.waypoints))}

        self.total_distance_to_goal = np.linalg.norm([self.target_x + 6.4981, self.target_y +1.0627])  # 起點到目標的總距離
        self.last_progress_ratio = 0  # 上一次的進度比例

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
            png_image = Image.open('/home/daniel/maps/my_map0924_2.png').convert('L')
            self.slam_map = np.array(png_image)  # 轉為NumPy陣列


    def generate_waypoints(self):
        waypoints = [(-6.4981, -1.0627),
            (-5.4541, -1.0117),
            (-4.4041, -0.862),
            (-3.3692, -1.0294),
            (-2.295, -1.114),
            (-1.2472, -1.0318),
            (-0.1614, -0.6948),
            (0.8931, -0.8804),
            (1.9412, -0.8604),
            (2.9804, -0.7229),
            (3.874, -0.2681),
            (4.9283, -0.1644),
            (5.9876, -0.345),
            (7.019, -0.5218),
            (7.9967, -0.2338),
            (9.0833, -0.1096),
            (10.1187, -0.3335),
            (11.1745, -0.6322),
            (12.1693, -0.8619),
            (13.1291, -0.4148),
            (14.1217, -0.0282),
            (15.1261, 0.123),
            (16.1313, 0.4439),
            (17.1389, 0.696),
            (18.1388, 0.6685),
            (19.2632, 0.5127),
            (20.2774, 0.2655),
            (21.2968, 0.0303),
            (22.3133, -0.0192),
            (23.2468, 0.446),
            (24.1412, 0.9065),
            (25.1178, 0.5027),
            (26.1279, 0.4794),
            (27.0867, 0.8266),
            (28.0713, 1.4229),
            (29.1537, 1.3866),
            (30.2492, 1.1549),
            (31.385, 1.0995),
            (32.4137, 1.243),
            (33.4134, 1.5432),
            (34.4137, 1.5904),
            (35.4936, 1.5904),
            (36.5067, 1.5607),
            (37.5432, 1.5505),
            (38.584, 1.7008),
            (39.6134, 1.9053),
            (40.5979, 2.0912),
            (41.6557, 2.3779),
            (42.5711, 2.8643),
            (43.5911, 2.9725),
            (44.5929, 3.0637),
            (45.5919, 2.9841),
            (46.6219, 2.9569),
            (47.6314, 3.0027),
            (48.7359, 2.832),
            (49.5462, 2.1761),
            (50.5982, 2.1709),
            (51.616, 2.3573),
            (52.6663, 2.5593),
            (53.7532, 2.5325),
            (54.7851, 2.5474),
            (55.8182, 2.5174),
            (56.8358, 2.6713),
            (57.8557, 2.8815),
            (58.8912, 3.0949),
            (59.7436, 3.6285),
            (60.5865, 4.2367),
            (60.6504, 5.2876),
            (60.7991, 6.3874),
            (60.322, 7.3094),
            (59.8004, 8.1976),
            (59.4093, 9.195),
            (59.1417, 10.1994),
            (59.1449, 11.2274),
            (59.5323, 12.2182),
            (59.8637, 13.2405),
            (60.5688, 14.0568),
            (60.6266, 15.1571),
            (60.007, 15.9558),
            (59.0539, 17.0128),
            (57.9671, 17.326),
            (56.9161, 16.7399),
            (55.9553, 17.0346),
            (54.9404, 17.0596),
            (53.9559, 16.8278),
            (52.9408, 16.8697),
            (51.9147, 16.7642),
            (50.9449, 16.4902),
            (49.9175, 16.3029),
            (48.8903, 16.1165),
            (47.7762, 16.0994),
            (46.7442, 16.0733),
            (45.7566, 15.8195),
            (44.756, 15.7218),
            (43.7254, 15.9309),
            (42.6292, 15.8439),
            (41.6163, 15.8177),
            (40.5832, 15.7881),
            (39.5617, 15.773),
            (38.5099, 15.5648),
            (37.692, 14.9481),
            (36.8538, 14.3078),
            (35.8906, 13.8384),
            (34.8551, 13.6316),
            (33.8205, 13.5495),
            (32.7391, 13.4423),
            (31.7035, 13.1056),
            (30.6971, 12.7802),
            (29.6914, 12.5216),
            (28.7072, 12.3238),
            (27.6442, 12.0953),
            (26.5991, 11.9873),
            (25.5713, 11.9867),
            (24.488, 12.0679),
            (23.4441, 12.0246),
            (22.3169, 11.7745),
            (21.3221, 11.538),
            (20.3265, 11.4243),
            (19.2855, 11.5028),
            (18.2164, 11.5491),
            (17.1238, 11.6235),
            (16.0574, 11.4029),
            (14.982, 11.2479),
            (13.9491, 11.0487),
            (12.9017, 11.1455),
            (11.8915, 11.4186),
            (10.8461, 11.6079),
            (9.9029, 12.0097),
            (9.0549, 12.5765),
            (8.4289, 13.4238),
            (7.4035, 13.6627),
            (6.3785, 13.5659),
            (5.3735, 13.4815),
            (4.3971, 13.1044),
            (3.3853, 13.2918),
            (2.3331, 13.0208),
            (1.2304, 12.9829),
            (0.2242, 13.094),
            (-0.807, 12.9358),
            (-1.8081, 12.8495),
            (-2.7738, 13.3168),
            (-3.4822, 14.0699),
            (-4.5285, 14.2483),
            (-5.5965, 13.9753),
            (-6.5324, 13.6016),
            (-7.3092, 12.8632),
            (-8.3255, 12.9916),
            (-9.1914, 13.7593),
            (-10.2374, 14.069),
            (-11.2162, 13.7566),
            (-11.653, 12.8061),
            (-11.6989, 11.7238),
            (-11.8899, 10.7353),
            (-12.6174, 10.0373),
            (-12.7701, 8.9551),
            (-12.4859, 7.9523),
            (-12.153, 6.8903),
            (-12.4712, 5.819),
            (-13.0498, 4.8729),
            (-13.1676, 3.8605),
            (-12.4328, 3.1822),
            (-12.1159, 2.1018),
            (-12.8436, 1.2659),
            (-13.3701, 0.2175),
            (-13.0514, -0.8866),
            (-12.3046, -1.619),
            (-11.2799, -1.472),
            (-10.1229, -1.3051),
            (-9.1283, -1.4767),
            (-8.1332, -1.2563),
            (self.target_x, self.target_y)
]
        return waypoints
    
    def calculate_waypoint_distances(self):
        """
        計算每對相鄰 waypoint 之間的距離，並返回一個距離列表。
        """
        distances = []
        for i in range(len(self.waypoints) - 1):
            start_wp = self.waypoints[i]
            next_wp = self.waypoints[i + 1]
            distance = np.linalg.norm([next_wp[0] - start_wp[0], next_wp[1] - start_wp[1]])
            distances.append(distance)
        return distances


    def gazebo_to_image_coords(self, gazebo_x, gazebo_y):
        img_x = 2000 + gazebo_x * 20
        img_y = 2000 - gazebo_y * 20
        return int(img_x), int(img_y)

    def image_to_gazebo_coords(self, img_x, img_y):
        gazebo_x = (img_x - 2000) / 20.0
        gazebo_y = (2000 - img_y) / 20.0
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
                                                    max(0, x - grid_size // 2):min(x + grid_size // 2, png_image.shape[1])] < 180)

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
        print(self.waypoints)
        self.optimized_waypoints_calculated = True  # 設定標記，表示已計算過


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

    def generate_occupancy_grid(self, robot_x, robot_y, linear_speed, steer_angle, grid_size=0.05, map_size=100):
        # 将机器人的坐标转换为地图上的像素坐标

        linear_speed = np.clip(linear_speed, -2.0, 2.0)
        steer_angle = np.clip(steer_angle, -0.5, 0.5)

        img_x, img_y = self.gazebo_to_image_coords(robot_x, robot_y)

        # 计算64x64网格在图片上的起始和结束索引
        half_grid = 32
        start_x = max(0, img_x - half_grid)
        start_y = max(0, img_y - half_grid)
        end_x = min(self.slam_map.shape[1], img_x + half_grid)
        end_y = min(self.slam_map.shape[0], img_y + half_grid)

        # 提取图片中的64x64区域
        grid = np.zeros((64, 64), dtype=np.float32)
        grid_slice = self.slam_map[start_y:end_y, start_x:end_x]

        # 填充 grid，将超出地图范围的部分填充为0
        grid[:grid_slice.shape[0], :grid_slice.shape[1]] = grid_slice

        # 将当前机器人位置信息添加到占据栅格
        occupancy_grid = np.zeros((3, 64, 64), dtype=np.float32)
        
        # 第一层：归一化图片数据到 [0, 1]
        occupancy_grid[0, :, :] = grid

        # 第二层：归一化速度到 [0, 1]
        occupancy_grid[1, :, :] = robot_x

        # 第三层：归一化角度到 [0, 1]
        occupancy_grid[2, :, :] = robot_y

        if np.isnan(occupancy_grid).any() or np.isinf(occupancy_grid).any():
            raise ValueError("NaN or Inf detected in occupancy_grid!")
        return occupancy_grid


    def step(self, action, obstacles):
        reward = 0
        robot_x, robot_y, robot_yaw = self.get_robot_position()

        # 确保 action 是一维数组
        action = np.squeeze(action)
        linear_speed = np.clip(action[0], -2.0, 2.0)
        steer_angle = np.clip(action[1], -0.5, 0.5)
        print("linear speed = ", linear_speed, " steer angle = ", steer_angle)

        # 更新状态
        self.state = self.generate_occupancy_grid(robot_x, robot_y, linear_speed, steer_angle)

        distances = [np.linalg.norm([robot_x - wp_x, robot_y - wp_y]) for wp_x, wp_y in self.waypoints]
        closest_index = np.argmin(distances)

        if closest_index > self.current_waypoint_index:
            self.current_waypoint_index = closest_index

        distance_to_goal = np.linalg.norm([robot_x - self.target_x, robot_y - self.target_y])

        # 奖励逻辑：每靠近目标 1/12 的总距离，增加奖励
        progress_ratio = 1 - distance_to_goal / self.total_distance_to_goal  # 计算当前进度比例
        if progress_ratio - self.last_progress_ratio >= 1 / 12.0:  # 每超过 1/12 的距离进度
            reward += 3.0
            self.last_progress_ratio = progress_ratio  # 更新进度

        if distance_to_goal < 0.5:  # 设定阈值为0.5米，可根据需要调整
            print('Robot has reached the goal!')
            reward += 20.0 # 给一个大的正向奖励
            self.reset()
            return self.state, reward, True, {}  # 重置环境
        
        if self.current_waypoint_index < len(self.waypoints):
            current_wp = self.waypoints[self.current_waypoint_index]
            distance_to_wp = np.linalg.norm([robot_x - current_wp[0], robot_y - current_wp[1]])
            if distance_to_wp < 0.5:  # 假設通過 waypoint 的距離閾值為 0.5
                reward += 1.0  # 通過 waypoint 獎勵

        # 更新机器人位置
        if self.previous_robot_position is not None:
            distance_moved = np.linalg.norm([
                robot_x - self.previous_robot_position[0],
                robot_y - self.previous_robot_position[1]
            ])
            reward += distance_moved*5  # 根据移动距离奖励
            # print("reward by distance_moved +", distance_moved)
        else:
            distance_moved = 0

        self.previous_robot_position = (robot_x, robot_y)

        # 检查是否需要使用 RL 控制
        failure_range = range(
            max(0, self.current_waypoint_index - 6),
            min(len(self.waypoints), self.current_waypoint_index + 2)
        )
        use_deep_rl_control = any(
            self.waypoint_failures.get(i, 0) > 1 for i in failure_range
        )

        use_deep_rl_control = True  # test  RL mdoel

        collision = detect_collision(robot_x, robot_y, robot_yaw, obstacles)
        if use_deep_rl_control:
            if collision:
                self.waypoint_failures[self.current_waypoint_index] += 1
                print('touch the obstacles')
                reward -= 10.0
                self.reset()
                return self.state, reward, True, {}
        
        

        # # 处理无进展的情况
        # if distance_moved < 0.05:
        #     self.no_progress_steps += 1
        #     # reward -= 0.3
        #     if self.no_progress_steps >= 20:
        #         if use_deep_rl_control:
        #             print('failure at point', self.current_waypoint_index)
        #             rospy.loginfo("No progress detected, resetting environment.")
        #             reward -= 5.0
        #             self.reset()
        #             return self.state, reward, True, {}
        #         else:
        #             self.waypoint_failures[self.current_waypoint_index] += 1
        #             print('failure at point', self.current_waypoint_index)
        #             rospy.loginfo("No progress detected, resetting environment.")
        #             reward -= 10.0
        #             self.reset()
        #             return self.state, reward, True, {}
        # else:
        #     self.no_progress_steps = 0

        # 发布控制命令
        twist = Twist()
        twist.linear.x = linear_speed
        twist.angular.z = steer_angle
        self.pub_cmd_vel.publish(twist)
        self.last_twist = twist

        imu_data = self.generate_imu_data()
        self.pub_imu.publish(imu_data)

        rospy.sleep(0.1)

        if isinstance(self.state, np.ndarray):
            self.state = torch.tensor(self.state, dtype=torch.float32).unsqueeze(0).to(device)  # 增加 batch 维度
        elif self.state.dim() != 4:
            self.state = self.state.unsqueeze(0)  # 增加 batch 维度
        reward, _ = self.calculate_reward(robot_x, robot_y, reward, self.state)
        print('reward = ',reward)
        return self.state, reward, self.done, {}

    def reset(self):

        robot_x, robot_y,_ = self.get_robot_position()
        self.state = self.generate_occupancy_grid(robot_x, robot_y, linear_speed=0, steer_angle=0)

        # 設置初始機器人位置和姿態
        yaw = -0.0053
        quaternion = quaternion_from_euler(0.0, 0.0, yaw)
        state_msg = ModelState()
        state_msg.model_name = 'my_robot'
        state_msg.pose.position.x = -6.4981
        state_msg.pose.position.y = -1.0627
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

        self.last_twist = Twist()
        self.pub_cmd_vel.publish(self.last_twist)

        imu_data = self.generate_imu_data()
        self.pub_imu.publish(imu_data)

        self.previous_yaw_error = 0
        self.no_progress_steps = 0
        self.previous_distance_to_goal = None

        # Ensure the state is 4D tensor
        if isinstance(self.state, np.ndarray):
            self.state = torch.tensor(self.state, dtype=torch.float32).unsqueeze(0).to(device)
        elif self.state.dim() != 4:
            self.state = self.state.unsqueeze(0)
        return self.state


    def calculate_reward(self, robot_x, robot_y, reward, state):
        done = False
        # 將機器人的座標轉換為地圖上的坐標
        
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        if state.ndim == 4:
            # 对于 4 维情况，取第一个批次数据中的第一层
            occupancy_grid = state[0, 0]
        elif state.ndim == 3:
            # 对于 3 维情况，直接取第一层
            occupancy_grid = state[0]

        img_x, img_y = self.gazebo_to_image_coords(robot_x, robot_y)
        obstacle_count = np.sum(occupancy_grid <= 190/255.0)  # 假設state[0]為佔據網格通道
        reward += 1 - obstacle_count*3/100.0

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
        lookahead_distance = 1.2 + 0.5 * linear_speed  # 根據速度調整前視距離

        # 定義角度範圍，以當前車輛的yaw為中心
        angle_range = np.deg2rad(30)  # ±40度的範圍
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
            linear_speed = 3

        # 使用PD控制器調整轉向角度
        kp, kd = self.adjust_control_params(linear_speed)
        previous_yaw_error = getattr(self, 'previous_yaw_error', 0)
        current_yaw_error_rate = yaw_error - previous_yaw_error
        steer_angle = kp * yaw_error + kd * current_yaw_error_rate
        steer_angle = np.clip(steer_angle, -0.5, 0.5)

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
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2)
        self.bn4 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(self._get_conv_output_size(observation_space), 256)
        self.lstm = nn.LSTM(input_size=256, hidden_size=128, batch_first=True)

        self.actor_mean = nn.Linear(128, action_space)
        self.actor_std = nn.Linear(128, action_space)  # 动态方差
        self.critic = nn.Linear(128, 1)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        

    def _get_conv_output_size(self, shape):
        x = torch.zeros(1, *shape)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = x.view(1, -1)
        return x.size(1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))

        x, _ = self.lstm(x.unsqueeze(0))  # 时间序列处理
        x = x.squeeze(0)

        action_mean = self.actor_mean(x)
        action_std = torch.clamp(torch.exp(self.actor_std(x)), min=1e-3, max=2.0)  # 限制标准差范围
        value = self.critic(x)

        return action_mean, action_std, value


    def act(self, state):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32).to(device)
        
        # Ensure proper dimensions
        while state.dim() > 4:
            state = state.squeeze(0)
        while state.dim() < 4:
            state = state.unsqueeze(0)
        
        action_mean, action_std, _ = self(state)
        
        # Define action bounds
        action_space = torch.tensor([
            [-2.0, -0.5],  # min values
            [2.0, 0.5]    # max values
        ], device=device)
        
        # Scale actions to proper range
        scaled_mean = action_space[0] + (torch.tanh(action_mean) + 1.0) * (action_space[1] - action_space[0]) / 2.0
        
        if self.training:
            # Add bounded noise during training
            noise = torch.clamp(torch.randn_like(action_std) * action_std, -0.5, 0.5)
            action = torch.clamp(scaled_mean + noise, action_space[0], action_space[1])

        else:
            action = scaled_mean
                
        return action.detach()

    def evaluate(self, state, action):
        try:
            action_mean, action_std, value = self(state)
            action_std = torch.clamp(action_std, min=1e-6, max=1.0)  # 避免无效值
            dist = torch.distributions.Normal(action_mean, action_std)
            action_log_probs = dist.log_prob(action).sum(dim=-1, keepdim=True)
            dist_entropy = dist.entropy().sum(dim=-1, keepdim=True)
            return action_log_probs, value, dist_entropy
        except Exception as e:
            print(f"Error in evaluate: {e}")
            raise


class DWA:
    def __init__(self, goal):
        self.max_speed = 2
        self.max_yaw_rate = 0.5
        self.dt = 0.2
        self.predict_time = 3.0
        self.goal = goal
        self.robot_radius = 0.3

    def calc_dynamic_window(self, state):
        # 當前速度限制
        vs = [0, self.max_speed, -self.max_yaw_rate, self.max_yaw_rate]
        dw = vs
        return dw

    def motion(self, state, control):
        # 運動模型計算下一步
        x, y, theta, v, omega = state
        next_x = x + v * np.cos(theta) * self.dt
        next_y = y + v * np.sin(theta) * self.dt
        next_theta = theta + omega * self.dt
        next_v = control[0]
        next_omega = control[1]
        return [next_x, next_y, next_theta, next_v, next_omega]

    def calc_trajectory(self, state, control):
        # 預測軌跡
        trajectory = [state]
        for _ in range(int(self.predict_time / self.dt)):
            state = self.motion(state, control)
            trajectory.append(state)
        return np.array(trajectory)

    def calc_score(self, trajectory, obstacles):
        # 目标距离分数
        x, y = trajectory[-1, 0], trajectory[-1, 1]
        goal_dist = np.sqrt((self.goal[0] - x) ** 2 + (self.goal[1] - y) ** 2)
        goal_score = -goal_dist

        # 安全分数：检测轨迹中是否发生碰撞
        clearance_score = float('inf')
        for tx, ty, _, _, _ in trajectory:
            for ox, oy in obstacles:
                dist = np.sqrt((ox - tx) ** 2 + (oy - ty) ** 2)
                if dist < self.robot_radius:
                    return goal_score, -100.0, 0.0  # 如果发生碰撞，直接返回最低分
                clearance_score = min(clearance_score, dist)

        # 速度分数
        speed_score = trajectory[-1, 3]  # 最终速度
        return goal_score, clearance_score, speed_score

    def plan(self, state, obstacles):
        print("dwa goal: ", self.goal)
        # 獲取動態窗口
        obstacles = [(ox, oy) for ox, oy in obstacles]
        dw = self.calc_dynamic_window(state)  # 速度 角度限制
        # 遍歷動態窗口中的所有控制
        best_trajectory = None
        best_score = -100.0
        best_control = [0.0, 0.0]
        # print("Dynamic Window", dw)
        for v in np.arange(dw[0], dw[1], 0.2):  # 線速度範圍
            for omega in np.arange(dw[2], dw[3], 0.1):  # 角速度範圍

                # 模擬軌跡
                control = [v, omega]
                trajectory = self.calc_trajectory(state, control)
                # 計算評分函數
                goal_score, clearance_score, speed_score = self.calc_score(trajectory, obstacles)
                total_score = goal_score * 0.5 + clearance_score * 0.45  + speed_score * 0.05

                # 找到最佳控制
                if total_score > best_score:
                    best_score = total_score
                    best_trajectory = trajectory
                    best_control = control
        print(f"v: {v}, omega: {omega}, goal_score: {goal_score}, clearance_score: {clearance_score}, speed_score: {speed_score}")
        return best_control, best_trajectory

def ppo_update(ppo_epochs, env, model, optimizer, memory, scaler, batch_size, scheduler):
    valid_samples = len([x for x in memory.memory if x is not None])
    print(f"Valid samples in memory: {valid_samples}")
    if valid_samples < batch_size:
        print("No sufficient samples to update")
        return

    accumulation_steps = 4
    warmup_epochs = 10
    optimizer.zero_grad()

    for epoch in range(ppo_epochs):
        # Adjust learning rate
        progress = min(1.0, epoch / warmup_epochs)
        adjusted_lr = LEARNING_RATE * progress * (1 / (1 + epoch * 0.001))
        for param_group in optimizer.param_groups:
            param_group['lr'] = adjusted_lr

        # Sample batch
        state_batch, action_batch, reward_batch, done_batch, next_state_batch, indices, weights = memory.sample(batch_size)
        state_batch, next_state_batch = _adjust_dimensions(state_batch, next_state_batch)
        print(f"Sampled batch: State batch shape {state_batch.shape}, Reward batch {reward_batch.shape}")

        # Normalize rewards
        reward_batch = (reward_batch - reward_batch.mean()) / (reward_batch.std() + 1e-8)
        reward_batch = torch.clamp(reward_batch, -10.0, 10.0)

        # Compute advantages
        with torch.no_grad():
            old_log_probs, _, _ = model.evaluate(state_batch, action_batch)
            _, _, next_values = model(next_state_batch)
            _, _, values = model(state_batch)

        values = values.squeeze()
        target_values = reward_batch + (1 - done_batch) * GAMMA * next_values.squeeze()
        advantages = (target_values - values).detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = torch.clamp(advantages, -10.0, 10.0)

        print(f"Target values shape: {target_values.shape}, Advantages shape: {advantages.shape}")

        # PPO update steps
        for step in range(ppo_epochs):
            log_probs, values, dist_entropy = model.evaluate(state_batch, action_batch)
            log_probs = log_probs.view(-1)
            weights = weights.view(-1)
            ratio = (log_probs - old_log_probs).exp()
            ratio = torch.clamp(ratio, 0.0, 10.0)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - CLIP_PARAM, 1 + CLIP_PARAM) * advantages

            actor_loss = -(weights * torch.min(surr1, surr2)).mean()
            critic_loss = 0.5 * (weights * (values - target_values).pow(2)).mean()
            entropy_loss = -0.01 * (weights * dist_entropy).mean()

            loss = (actor_loss + critic_loss + entropy_loss) / accumulation_steps

            # Gradient accumulation
            scaler.scale(loss).backward()

            if (step + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            wandb.log({
                "Actor Loss": actor_loss.item(),
                "Critic Loss": critic_loss.item(),
                "Entropy Loss": entropy_loss.item(),
            })

        # Update priorities
        priorities = advantages.abs().detach().cpu().numpy()
        memory.update_priorities(indices, priorities)

    # Clear memory after update
    memory.clear()



def calculate_bounding_box(robot_x, robot_y, robot_yaw):

    # 机器人中心到边界的相对距离
    half_length = 0.355
    half_width = 0.303

    # 矩形的局部坐标系下的 4 个角点
    corners = np.array([
        [half_length, half_width],
        [half_length, -half_width],
        [-half_length, -half_width],
        [-half_length, half_width]
    ])

    # 旋转矩阵
    rotation_matrix = np.array([
        [np.cos(robot_yaw), -np.sin(robot_yaw)],
        [np.sin(robot_yaw), np.cos(robot_yaw)]
    ])

    # 全局坐标系下的角点
    global_corners = np.dot(corners, rotation_matrix.T) + np.array([robot_x, robot_y])
    return global_corners

def is_point_in_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    inside = False

    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]

        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
            inside = not inside
    return inside

def detect_collision(robot_x, robot_y, robot_yaw, obstacles):
    # 计算边界框
    bounding_box = calculate_bounding_box(robot_x, robot_y, robot_yaw,)

    # 遍历障碍物
    for obstacle in obstacles:
        # 检查是否在边界内
        if is_point_in_polygon(obstacle, bounding_box):
            return True
    return False


def _adjust_dimensions(state_batch, next_state_batch):
    # Remove extra dimension from state_batch if present
    if state_batch.dim() == 5:
        state_batch = state_batch.squeeze(1)
    
    # Remove extra dimension from next_state_batch if present
    if next_state_batch.dim() == 5:
        next_state_batch = next_state_batch.squeeze(1)
        
    # Verify final shapes
    if state_batch.dim() != 4:
        raise ValueError(f"State batch should be 4D, got shape: {state_batch.shape}")
    if next_state_batch.dim() != 4:
        raise ValueError(f"Next state batch should be 4D, got shape: {next_state_batch.shape}")
        
    return state_batch, next_state_batch

def _check_for_nan(tensors, error_message):
    for tensor in tensors:
        if tensor is not None and torch.isnan(tensor).any():
            raise ValueError(error_message)
        
def _check_for_invalid_values(tensor, name):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        raise ValueError(f"[PPO Update] {name} contains invalid values (NaN or Inf).")

def select_action_with_exploration(env, state, model, epsilon, dwa=None, obstacles=None):
    if random.random() < epsilon:
        if dwa is None or obstacles is None:
            raise ValueError("DWA controller or obstacles is not provided")
        print("[Exploration] Using DWA for action generation.")

        robot_x, robot_y, robot_yaw = env.get_robot_position()
        current_speed = env.last_twist.linear.x
        current_omega = env.last_twist.angular.z

        print('robot x = ', robot_x, 'robot_y = ', robot_y)
        state = [robot_x, robot_y, robot_yaw, current_speed, current_omega]
        action, _ = dwa.plan(state, obstacles)  
        action = torch.tensor(action, dtype=torch.float32).to(device)
    else:
        print('action by RL')
        action = model.act(state)
    return action


def save_movement_log_to_csv(movement_log, filename= f"/home/daniel/catkin_ws/src/my_robot_control/new_waypoint/move_log{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"):
    with open(filename,mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['x', 'y', 'yaw'])
        for log in movement_log:
            writer.writerow(log)
    print(f"Movement log saved to {filename}")

def main():
    wandb.init(
        project="gazebo-rl",
        config={
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "gamma": GAMMA,
            "clip_param": CLIP_PARAM,
            "ppo_epochs": PPO_EPOCHS,
            "memory_size": MEMORY_SIZE
        }
    )
    env = GazeboEnv(None)
    dwa = DWA(goal=env.waypoints[env.current_waypoint_index + 3])
    model = ActorCritic(env.observation_space, env.action_space).to(device)
    env.model = model

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.95)  # 每1000次調整學習率
    scaler = GradScaler('cuda')
    memory = PrioritizedMemory(MEMORY_SIZE)

    model_path = "/home/daniel/catkin_ws/src/my_robot_control/scripts/saved_model_ppo.pth"
    best_model_path = "/home/daniel/catkin_ws/src/my_robot_control/scripts/best_model.pth"
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print("Loaded existing model.")
    else:
        print("Created new model.")

    num_episodes = 1000000
    best_test_reward = -np.inf
    
    last_recorded_position = None

    # Initialize obstacles
    static_obstacles = []
    for y in range(env.slam_map.shape[0]):
        for x in range(env.slam_map.shape[1]):
            if env.slam_map[y, x] < 190:
                ox, oy = env.image_to_gazebo_coords(x, y)
                static_obstacles.append((ox, oy))
 
    for e in range(num_episodes):
        movement_log = []
        if not env.optimized_waypoints_calculated:
            env.optimize_waypoints_with_a_star()

        state = env.reset()
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        state = state.clone().detach().unsqueeze(0).to(device)

        total_reward = 0
        start_time = time.time()
        use_rl = False  # Flag to determine if RL is used in this episode

        for time_step in range(1500):
            robot_x, robot_y, robot_yaw = env.get_robot_position()

            if last_recorded_position is None or np.linalg.norm(
                [robot_x - last_recorded_position[0], robot_y - last_recorded_position[1]]
            ) >= 1.0386:
                movement_log.append((robot_x, robot_y, robot_yaw))
                last_recorded_position = (robot_x, robot_y)

            obstacles = [
                (ox, oy) for ox, oy in static_obstacles
                if np.sqrt((ox-robot_x)**2 + (oy - robot_y)**2) < 4.0
            ]
            obstacles = grid_filter(obstacles, grid_size=0.7)

            lookahead_index = min(env.current_waypoint_index + 3, len(env.waypoint_distances)-1)
            dwa.goal = env.waypoints[lookahead_index]

            failure_range = range(
                max(0, env.current_waypoint_index - 6),
                min(len(env.waypoints), env.current_waypoint_index + 2)
            )
            failure_counts = {i: env.waypoint_failures.get(i, 0) for i in failure_range}

            use_deep_rl_control = any(
                env.waypoint_failures.get(i, 0) > 1 for i in failure_range
            )

            use_deep_rl_control = True  # test RL model 

            if use_deep_rl_control:
                use_rl = True  # Mark this episode as using RL
                action = select_action_with_exploration(env, state, model, epsilon=0.0 , dwa=dwa, obstacles=obstacles)
                action_np = action.detach().cpu().numpy().flatten()
                print(f"RL Action at waypoint {env.current_waypoint_index}: {action_np}")
            else:
                action_np = env.calculate_action_pure_pursuit()
                print(f"A* Action at waypoint {env.current_waypoint_index}: {action_np}")

            next_state, reward, done, _ = env.step(action_np, obstacles=obstacles)

            if isinstance(next_state, np.ndarray):
                next_state = torch.tensor(next_state, dtype=torch.float32)

                        # 确保维度正确
            if next_state.dim() == 5:
                next_state = next_state.squeeze(1)  # 去掉多余的维度
            if next_state.dim() == 3:
                next_state = next_state.unsqueeze(0)  # 增加 batch 维度

            
            memory.add(state.cpu().numpy(), action_np, reward, done, next_state.cpu().numpy())
            total_reward += reward

            state = next_state

            state = (state - state.min()) / (state.max() - state.min() + 1e-5)

            elapsed_time = time.time() - start_time
            if done or elapsed_time > 150:
                if elapsed_time > 150:
                    # reward -= 10.0
                    print(f"Episode {e} failed at time step {time_step}: time exceeded 240 sec.")
                break

        if use_rl and len(memory.memory) > BATCH_SIZE:
            print(123456)
            current_batch_size = min(BATCH_SIZE, len(memory.memory))
            ppo_update(PPO_EPOCHS, env, model, optimizer, memory, scaler, batch_size=current_batch_size,scheduler=scheduler)
            
        
        if use_rl:
            wandb.log({
                "Episode": e,
                "Total Reward": total_reward
            })



        print(f"Episode {e}, Total Reward: {total_reward}, LR: {scheduler.get_last_lr()[0]}")

        if total_reward > best_test_reward:
            best_test_reward = total_reward
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with reward: {best_test_reward}")
            # save_movement_log_to_csv(movement_log)

        if e % 5 == 0:
            torch.save(model.state_dict(), model_path)
            print(f"Model saved after {e} episodes.")
        rospy.sleep(1.0)

    torch.save(model.state_dict(), model_path)
    print("Final model saved.")


if __name__ == '__main__':
    main()