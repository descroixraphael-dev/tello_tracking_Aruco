from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node

def generate_launch_description():
    # Define the nodes
    driver_node = Node(
        package='aruco_folder',
        executable='driver_node',
        output='screen'
    )

    aruco_detector_node = Node(
        package='aruco_folder',
        executable='aruco_detector',
        output='screen'
    )

    single_axis_node = Node(
        package='aruco_folder',
        executable='single_axis',
        output='screen'
    )

    keyboard_land_node = Node(
        package='aruco_folder',
        executable='keyboard_land',
        output='screen'
    )

    # Add delays using TimerAction (adjust the period in seconds as needed)
    delayed_aruco_detector = TimerAction(
        period=5.0,  # 5-second delay after launch starts
        actions=[aruco_detector_node]
    )

    delayed_single_axis = TimerAction(
        period=8.0,  # 8-second delay after launch starts (3 seconds after detector)
        actions=[single_axis_node]
    )

    return LaunchDescription([
        driver_node,
        delayed_aruco_detector,
        delayed_single_axis,
        keyboard_land_node,  # Launches immediately alongside driver_node
    ])
