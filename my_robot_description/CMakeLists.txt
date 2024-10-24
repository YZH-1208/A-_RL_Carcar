cmake_minimum_required(VERSION 3.0.2)
project(my_robot_description)

find_package(catkin REQUIRED COMPONENTS
  rospy
  gazebo_ros
)

catkin_package()

catkin_install_python(PROGRAMS scripts/rtab_node_launch.py
                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})


include_directories(
  ${catkin_INCLUDE_DIRS}
)

install(DIRECTORY launch worlds models
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
