#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import math
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import tf
from geometry_msgs.msg import Pose, Twist

def rtab_publisher():
    rospy.init_node('rtab_publisher')
    
    # 创建发布者
    odom_pub = rospy.Publisher('/odom', Odometry, queue_size=50)
    scan_pub = rospy.Publisher('/scan', LaserScan, queue_size=50)
    odom_broadcaster = tf.TransformBroadcaster()
    
    rate = rospy.Rate(5.0)  # 发布频率10Hz
    
    x = y = th = 0.0
    vx = 0.1  # 线速度
    vth = 0.1  # 角速度
    
    while not rospy.is_shutdown():
        current_time = rospy.Time.now()
        
        # 计算里程计位移
        dt = 1.0 / 10.0
        delta_x = vx * dt
        delta_th = vth * dt

        x += delta_x * math.cos(th)
        y += delta_x * math.sin(th)
        th += delta_th

        # 发布odom到base_link的TF变换
        odom_quat = tf.transformations.quaternion_from_euler(0, 0, th)
        odom_broadcaster.sendTransform(
            (x, y, 0.0),
            odom_quat,
            current_time,
            "base_link",
            "odom"
        )

        # 发布base_link到laser_frame的TF变换
        laser_quat = tf.transformations.quaternion_from_euler(0, 0, 0)
        odom_broadcaster.sendTransform(
            (0.2, 0.0, 0.0),  # 激光雷达相对于base_link的位移
            laser_quat,
            current_time,
            "laser_frame",
            "base_link"
        )

        # 创建并发布里程计消息
        odom = Odometry()
        odom.header.stamp = current_time
        odom.header.frame_id = "odom"

        # 设置位置
        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation.x = odom_quat[0]
        odom.pose.pose.orientation.y = odom_quat[1]
        odom.pose.pose.orientation.z = odom_quat[2]
        odom.pose.pose.orientation.w = odom_quat[3]

        # 设置速度
        odom.child_frame_id = "base_link"
        odom.twist.twist.linear.x = vx
        odom.twist.twist.linear.y = 0.0
        odom.twist.twist.angular.z = vth

        # 发布里程计
        odom_pub.publish(odom)
        
        # 创建并发布激光扫描消息
        scan = LaserScan()
        scan.header.stamp = current_time
        scan.header.frame_id = "laser_frame"
        scan.angle_min = -math.pi / 2
        scan.angle_max = math.pi / 2
        scan.angle_increment = math.pi / 180  # 每度一次测量
        scan.time_increment = (1.0 / 10) / 360
        scan.scan_time = 1.0 / 10
        scan.range_min = 0.12
        scan.range_max = 3.5

        # 模拟数据
        num_readings = int((scan.angle_max - scan.angle_min) / scan.angle_increment)
        scan.ranges = [2.0] * num_readings  # 所有距离都设置为2.0米
        scan.intensities = [1.0] * num_readings

        # 发布激光扫描
        scan_pub.publish(scan)
        
        rate.sleep()

if __name__ == '__main__':
    try:
        rtab_publisher()
    except rospy.ROSInterruptException:
        pass
