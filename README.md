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

📄 See docs/ros2_dual_wifi_guide.txt for the full setup, environment exports, and DHCP-lease caveats — steps below assume that guide has already been followed and both interfaces are configured.
```

---

## 4. Node-Launch Protocols

Each protocol assumes the workspace is sourced and the correct
`ROS_DOMAIN_ID` / `RMW_IMPLEMENTATION` are exported in that terminal.

### Protocol A — Tello Driver Only (bring-up / telemetry check)

```bash
export ROS_DOMAIN_ID=10 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ros2 run <repo-name> driver_node
```
Verify topics:
```bash
ros2 topic echo /tello/battery
ros2 topic echo /tello/velocity
ros2 topic echo /tello/attitude
```

### Protocol B — ArUco Detection Only

```bash
ros2 run <repo-name> aruco_detector --ros-args -p marker_size:=0.10 -p camera_frame:=tello_camera
```
Verify:
```bash
ros2 topic echo /aruco/pose
```

### Protocol C — Single-Axis Test (structural template / calibration)

Use this before full navigation to validate the two-phase flight FSM
(`ALTITUDE_CALIBRATE` → `AXIS_TEST`) and PID gains on one axis at a time.

```bash
ros2 run <repo-name> single_axis --ros-args -p axis:=x -p target_distance:=1.0
```

### Protocol D — Full Autonomous Tracking (CTRV UKF + PID FSM)

```bash
ros2 launch <repo-name> tello_bringup.launch.py
```
This brings up, in order: `driver_node` → `aruco_detector` → `ukf_navigation`.
The tracking-loss FSM engages automatically:
- **HOLD** at 2 s of lost marker
- **SEARCHING** (slow yaw scan) at 5 s
- **ABORTED → land** if not recovered

### Protocol E — Orbit Mode

```bash
ros2 run <repo-name> orbit_nav --ros-args -p radius:=1.5 -p angular_speed:=0.3
```

### Protocol F — Marker-Based Drift-Free Odometry

Chains ASTRO's map-frame localization with ArUco detections to compute Tello
position without integration drift. Requires ASTRO's localization stack
already running on domain 4.

```bash
ros2 run <repo-name> tello_odom_publisher
```

> Before running: confirm ASTRO's actual `base_frame` name from the live TF
> tree (`base_link` vs `base_footprint` inconsistency has been seen in
> `nav2_params.yaml`):
> ```bash
> ros2 run tf2_tools view_frames    # (on ASTRO domain, ROS_DOMAIN_ID=4)
> ```

---

## 5. Launching the Full Cooperative Stack

The combined stack spans **three ROS graphs** (Tello domain 10, ASTRO domain 4,
and a bridged visualization domain) connected via `domain_bridge`. Full
protocol is in `docs/tello_astro_rviz_protocol.md`.

**Terminal 1 — ASTRO (domain 4, Zenoh):**
```bash
export ROS_DOMAIN_ID=4 RMW_IMPLEMENTATION=rmw_zenoh_cpp
ros2 launch astro_bringup slam.launch.py
```
> Confirm `enable_interactive_mode: false` in `mapper_params_online_async.yaml`
> — if `true`, `slam_toolbox` starts paused and never publishes `/map`.
> Also confirm `sllidar_ros2` is launched (not started by default in the
> ASTRO repo) and that `robot_base_frame` matches across `nav2_params.yaml`.

**Terminal 2 — Tello (domain 10, CycloneDDS):**
```bash
export ROS_DOMAIN_ID=10 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ros2 launch <repo-name> tello_bringup.launch.py
```

**Terminal 3 — Domain Bridge:**
```bash
ros2 run domain_bridge domain_bridge config/domain_bridge.yaml
```

**Terminal 4 — Combined RViz2 visualization:**
```bash
export ROS_DOMAIN_ID=<bridge_domain>
ros2 launch <repo-name> tello_astro_rviz.launch.py
```

### Five-Phase Integration Protocol (cooperative exploration + aerial following)

1. ASTRO SLAM + frontier exploration bring-up, verify `/map` publishing
2. Tello bring-up + ArUco lock-on over ASTRO's marker
3. Dual-domain bridge + combined RViz verification
4. Joint run: ASTRO explores autonomously, Tello tracks via UKF nav
5. Fault-injection pass: marker occlusion, Wi-Fi drop, in-place rotation
   during frontier selection (**highest-risk case for the CTRV UKF** — test
   explicitly before calling integration done)

---

## 6. Quick Troubleshooting Reference

| Symptom | Likely Cause |
|---|---|
| `/map` never publishes | `enable_interactive_mode: true` in slam_toolbox params |
| Nav2 can't find robot pose | `robot_base_frame` mismatch (`base_link` vs `base_footprint`) |
| No lidar data | `sllidar_ros2` driver not launched independently |
| Tello nodes lose discovery mid-session | DHCP lease expired, IP changed — recheck CycloneDDS peer config |
| UKF yaw jumps/oscillates during rotation | In-place rotation edge case — known high-risk CTRV failure mode |
| Domain bridge shows no topics | Domain IDs / RMW implementation mismatch between terminals |

---

## 7. Requirements

- ROS 2 (Humble or later recommended)
- `rmw_cyclonedds_cpp`, `rmw_zenoh_cpp`
- Python: `djitellopy`, `opencv-python`, `opencv-contrib-python` (ArUco), `filterpy` or custom UKF
- `domain_bridge`
- ASTRO stack (CRTA-Lab), `slam_toolbox`, `sllidar_ros2`, Nav2
