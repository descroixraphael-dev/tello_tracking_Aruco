from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(package='aruco_folder', executable='driver_node', output='screen'),
        Node(package='aruco_folder', executable='aruco_detector', output='screen'),
        Node(package='aruco_folder', executable='orbit_nav', output='screen'),
        Node(package='aruco_folder', executable='keyboard_land', output='screen'),
    ])
