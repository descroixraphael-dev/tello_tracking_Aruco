# tello_tracking_Aruco


Autonomous aerial-following module for the cooperative ground–aerial multi-robot
system. A DJI Tello tracks an ArUco marker mounted on a ground robot,
using a CTRV UKF for filtering and per-axis PID/FSM controllers for flight
control. This repo also supports a marker-based, drift-free odometry bridge
and a combined RViz2 visualization across the Tello and ASTRO ROS domains.



---

## 1. Directory Structure
Create the workspace and package folders:

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
```

---

## 2. Cloning the Repository

```bash
cd ~/ros2_ws/src
git clone https://github.com/descroixraphael-dev/tello_tracking_Aruco.git
```

Install Python/ROS dependencies:

```bash
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
pip install -r src/<repo-name>/requirements.txt   # djitellopy, opencv-python, filterpy, etc.
```
If pip is not installed install it:
```bash
sudo apt update
sudo python3-pip
```

Build:

```bash
cd ~/ros2_ws
colcon build --packages-select <repo-name> --symlink-install
source install/setup.bash
```

Add `source ~/ros2_ws/install/setup.bash` to `~/.bashrc` for persistence.

---

## 3. Network Setup (Dual Wi-Fi: Tello + ASTRO)

This stack talks to two robots on two separate subnets from one desktop (Tello on wlp0s20f3 / CycloneDDS / domain 10, ASTRO on wlx2887ba786c3c / Zenoh / domain 4).

📄 See aruco_folder/docstring/wifi_cofiguration.md for the full setup, environment exports, and DHCP-lease caveats — steps below assume that guide has already been followed and both interfaces are configured.


---

## 4. Node-Launch Protocols



### Protocol A — ArUco Detection Only
In one terminal write these line (don't forget to source your workspace)
```bash
ros-drone-a
ros2 run aruco_folder driver_node
```
in an other terminal:
```bash
ros-drone-a
ros2 run aruco_folder aruco_detector
```

### Protocol B — Single-Axis Test (structural template / calibration)

Use this if you want to tweak gains or debug some behavior, first do protocol A or the drone won't detect the marker

```bash
ros2 run aruco_folder single_axis
```

### Protocol C — Full Autonomous Tracking 

```bash
ros2 launch aruco_folder ctrm_launch
```

---


---

## 5. Requirements

- ROS 2 (Humble or later recommended)
- `rmw_cyclonedds_cpp`, `rmw_zenoh_cpp`
- Python: `djitellopy`, `opencv-python`, `opencv-contrib-python` (ArUco), `filterpy` or custom UKF
- `domain_bridge`
- ASTRO stack (CRTA-Lab) or any robot stack, `slam_toolbox`, `sllidar_ros2`, Nav2
