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
    
    ukf_navigation_node = Node(
        package='aruco_folder', 
        executable='ukf_navigation', 
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
    
    delayed_ukf_navigation = TimerAction(
        period=8.0, # 10-second delay after launch starts (5 seconds after detector)
        actions=[ukf_navigation_node]
    )

    return LaunchDescription([
        driver_node,
        delayed_aruco_detector,
        delayed_ukf_navigation,
        keyboard_land_node,  # Launches immediately alongside driver_node
    ])
