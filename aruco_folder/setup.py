from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'aruco_folder'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), ['launch/cvm_launch.py']),
        (os.path.join('share', package_name, 'launch'), ['launch/ctrm_launch.py']),
       
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='student',
    maintainer_email='student@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        "aruco_detector=aruco_folder.aruco_detector:main",
        "driver_node=aruco_folder.driver_node:main",
        "navigation=aruco_folder.navigation:main",
        "ufk_navigation=aruco_folder.ukf_navigation:main",
        "yaw_control=aruco_folder.yaw_control:main",
        "keyboard_land=aruco_folder.keyboard_land:main",
        "tracker=aruco_folder.tracker:main",
        "tuning_solver_node=aruco_folder.tuning_solver_node:main",
        "single_axis=aruco_folder.single_axis:main",
        "orbit_nav=aruco_folder.orbit_nav:main",
        ],
    },
)
