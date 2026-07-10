"""
orbit_nav.py — Tello Drone Navigation Controller
Tracks a visual ArUco marker via a UKF-CTRV estimator and drives
PID-based velocity commands. Position errors are rotated into the
marker's own heading frame so that a rotating marker does not cause
lateral drift — the drone always recenters relative to where the
marker is facing.
"""

import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.node import Node
from scipy.linalg import cholesky, LinAlgError
from std_msgs.msg import Empty

# ─────────────────────────────────────────────────────────────────────────────
#  Stand-off offset (metres), expressed in the MARKER's own local frame.
#  Marker flat on floor facing up:
#    +Z  →  pointing straight up into the ceiling.
#    -Y  →  pointing backward along the ground towards the drone's parking spot.
#  Applying this through the marker's measured rotation matrix means the
#  drone holds 1m behind / 45cm above the marker regardless of how the
#  marker itself is rotated on the floor.
# ─────────────────────────────────────────────────────────────────────────────
OFFSET_TOWARD_M = 1.0   # desired stand-off distance behind the marker (m)
OFFSET_NORMAL_M = 0.45  # desired height above the marker (m)


def quaternion_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert a quaternion [qx, qy, qz, qw] to a 3x3 rotation matrix."""
    return np.array([
        [1 - 2 * (qy**2 + qz**2),     2 * (qx*qy - qz*qw),     2 * (qx*qz + qy*qw)],
        [    2 * (qx*qy + qz*qw), 1 - 2 * (qx**2 + qz**2),     2 * (qy*qz - qx*qw)],
        [    2 * (qx*qz - qy*qw),     2 * (qy*qz + qx*qw), 1 - 2 * (qx**2 + qy**2)],
    ])


def compute_stand_off_goal(marker_pos: np.ndarray, R_marker: np.ndarray) -> np.ndarray:
    """
    Given the marker's measured position and rotation matrix (camera frame),
    return the desired drone hold position: OFFSET_TOWARD_M behind and
    OFFSET_NORMAL_M above the marker, along the marker's own local axes.
    """
    local_offset = np.array([0.0, -OFFSET_TOWARD_M, OFFSET_NORMAL_M])
    return marker_pos + R_marker @ local_offset


def marker_heading_from_rotation(R_marker: np.ndarray) -> float:
    """
    Extract the marker's facing direction (its local -Y axis -- the same
    direction the stand-off offset is measured along) as a heading angle in
    the camera's horizontal (X-Z) plane. At heading 0 this points along
    camera +Z, matching the convention the tangential/radial split and yaw
    controller already assume.
    """
    facing_dir = -R_marker[:, 1]
    return float(np.arctan2(facing_dir[0], facing_dir[2]))


# ──────────────────────────────────────────────────────────────────────────────
# UKF — Constant Turn Rate and Velocity (CTRV) Motion Model
# ──────────────────────────────────────────────────────────────────────────────

class UKF_CTRV:
    """
    Unscented Kalman Filter with a CTRV motion model.

    State vector:  [pos_x, pos_y, pos_z, velocity, heading_θ, turn_rate_ω]
    Measurement:   [pos_x, pos_y, pos_z]  (direct position from marker detection)
    """

    # ── Tuning constants ──────────────────────────────────────────────────────
    # Process noise — tweak to trade off smoothness vs. responsiveness
    _Q_DIAG = [0.05, 0.05, 0.05, 0.1, 0.05, 0.05]
    # Measurement noise — reflects camera/detection uncertainty (metres)
    _R_DIAG = [0.002, 0.002, 0.002]

    # UKF spreading parameters (van der Merwe defaults)
    _ALPHA = 1e-3
    _KAPPA = 0.0
    _BETA  = 2.0

    def __init__(self, dt: float = 0.05):
        self.dt = dt
        n = 6  # number of states
        self.num_states = n

        # Filter state and covariance
        self.x = np.zeros(n)
        self.P = np.eye(n)

        # Noise matrices
        self.Q = np.diag(self._Q_DIAG)
        self.R = np.diag(self._R_DIAG)

        # Pre-compute sigma-point weights
        lam = (self._ALPHA ** 2) * (n + self._KAPPA) - n
        self.lambda_ = lam
        self.weights_m = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)))
        self.weights_c = self.weights_m.copy()
        self.weights_m[0] = lam / (n + lam)
        self.weights_c[0] = self.weights_m[0] + (1 - self._ALPHA ** 2 + self._BETA)

        # Placeholder for the most recent predicted sigma points (set during predict)
        self.sigma_pred: np.ndarray | None = None

    # ── Private helpers ───────────────────────────────────────────────────────

    def _generate_sigma_points(self) -> np.ndarray:
        """Return the (n × 2n+1) matrix of sigma points centred on self.x."""
        n     = self.num_states
        scale = n + self.lambda_
        try:
            L = cholesky(scale * self.P, lower=True)
        except LinAlgError:
            # Fallback: regularise P if it has lost positive-definiteness
            L = cholesky(scale * (self.P + np.eye(n) * 1e-6), lower=True)

        sigma = np.zeros((n, 2 * n + 1))
        sigma[:, 0] = self.x
        for i in range(n):
            sigma[:, i + 1]     = self.x + L[:, i]
            sigma[:, i + 1 + n] = self.x - L[:, i]
        return sigma

    def _apply_ctrv(self, sigma_pts):
        dt = self.dt
        out = np.zeros_like(sigma_pts)
        for i in range(sigma_pts.shape[1]):
            px, py, pz, v, theta, omega = sigma_pts[:, i]

            if abs(omega) > 1e-4:
                px_next = px + (v / omega) * (-np.cos(theta + omega * dt) + np.cos(theta))
                pz_next = pz + (v / omega) * ( np.sin(theta + omega * dt) - np.sin(theta))
            else:  # near-zero turn rate → straight line
                px_next = px + v * np.sin(theta) * dt
                pz_next = pz + v * np.cos(theta) * dt

            py_next = py  # altitude: not part of horizontal heading/turn-rate dynamics

            out[:, i] = [px_next, py_next, pz_next, v, theta + omega * dt, omega]
        return out

    # ── Public interface ──────────────────────────────────────────────────────

    def reinitialise(self, position: np.ndarray) -> None:
        """
        Hard-reset the filter to a known position.
        Called on target re-acquisition to prevent stale drift from causing
        a position 'snap' when the first fresh measurement arrives.
        """
        self.x[:3] = position
        self.x[3:] = 0.0                        # zero velocity, heading, turn-rate
        self.P     = np.eye(self.num_states) * 0.05  # tight initial uncertainty

    def predict(self, dt: float) -> None:
        """Time-update step: propagate state and covariance forward by dt."""
        self.dt      = dt
        sigma_pts    = self._generate_sigma_points()
        self.sigma_pred = self._apply_ctrv(sigma_pts)
        self.x       = np.sum(self.weights_m * self.sigma_pred, axis=1)

        self.P = self.Q.copy()
        for i in range(2 * self.num_states + 1):
            d       = (self.sigma_pred[:, i] - self.x).reshape(-1, 1)
            self.P += self.weights_c[i] * (d @ d.T)

    def update(self, z: np.ndarray) -> None:
        """Measurement-update step: correct state with a 3-D position measurement."""
        # Project sigma points into measurement space (first 3 states = position)
        z_sigma = self.sigma_pred[0:3, :]
        z_pred  = np.sum(self.weights_m * z_sigma, axis=1)

        S  = self.R.copy()
        Tc = np.zeros((self.num_states, 3))
        for i in range(2 * self.num_states + 1):
            zd  = (z_sigma[:, i] - z_pred).reshape(-1, 1)
            xd  = (self.sigma_pred[:, i] - self.x).reshape(-1, 1)
            S  += self.weights_c[i] * (zd @ zd.T)
            Tc += self.weights_c[i] * (xd @ zd.T)

        K       = Tc @ np.linalg.inv(S)
        self.x += K @ (z - z_pred)
        self.P -= K @ S @ K.T
        self.P  = 0.5 * (self.P + self.P.T)  # enforce symmetry


# ──────────────────────────────────────────────────────────────────────────────
# PID Controller
# ──────────────────────────────────────────────────────────────────────────────

class PIDController:
    """
    Discrete PID with:
    - Integral anti-windup (clamp to ±max_out / ki)
    - Sign-change reset to prevent integral wind-up after overshoot
    """

    def __init__(self, kp: float, ki: float, kd: float, max_out: float):
        self.kp      = kp
        self.ki      = ki
        self.kd      = kd
        self.max_out = max_out

        self.integral   = 0.0
        self.last_error = 0.0

    def reset(self) -> None:
        self.integral   = 0.0
        self.last_error = 0.0

    def compute(self, error: float, dt: float) -> float:
        # Reset integral when error crosses zero (reduces overshoot)
        if self.last_error != 0.0 and np.sign(error) != np.sign(self.last_error):
            self.integral = 0.0

        self.integral += error * dt
        if self.ki:
            limit         = self.max_out / self.ki
            self.integral = np.clip(self.integral, -limit, limit)

        derivative      = (error - self.last_error) / dt if dt > 0 else 0.0
        self.last_error = error

        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return float(np.clip(output, -self.max_out, self.max_out))


# ──────────────────────────────────────────────────────────────────────────────
# ROS 2 Navigation Node
# ──────────────────────────────────────────────────────────────────────────────

class TelloNavigationController(Node):
    """
    Subscribes to marker poses, runs UKF estimation, and publishes
    Twist velocity commands to track the detected marker.

    Coordinate convention (drone body frame):
        +x → forward   +y → left   +z → up

    The marker pose is expressed as a position error relative to the
    desired hold position. Errors are rotated into the marker's own
    heading frame (from UKF state θ) before being fed to the PIDs,
    so the drone recenters correctly even when the marker rotates.
    """

    # ── Tuning constants ──────────────────────────────────────────────────────
    CONTROL_HZ          = 20     # control loop rate (Hz)
    CONTROL_DT          = 1.0 / CONTROL_HZ

    TRACKING_LOST_WARN  = 1.0   # seconds before UKF free-runs
    TRACKING_LOST_HALT  = 5.0   # seconds before safety stop

    SOFT_START_DURATION = 0.5   # ramp-up window after re-acquisition (s)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def __init__(self):
        super().__init__('tello_nav_controller')

        # ROS interfaces
        self.pose_sub    = self.create_subscription(
            PoseStamped, '/tello/marker_pose', self._pose_callback, 10)
        self.cmd_pub     = self.create_publisher(Twist, '/cmd_vel', 10)
        self.takeoff_pub = self.create_publisher(Empty, '/tello/takeoff', 10)

        # Estimator
        self.ukf = UKF_CTRV()

        # PID controllers  (axis names match body-frame axes above)
        self.pid_x = PIDController(kp=0.0490, ki=0.0048, kd=0.0146, max_out=0.90, min_effective_out=0.0)  # strafe
        self.pid_y = PIDController(kp=0.0450, ki=0.0041, kd=0.0146, max_out=0.90, min_effective_out=0.0)  # altitude — slightly lower
        self.pid_z = PIDController(kp=0.0490, ki=0.0048, kd=0.0146, max_out=0.90, min_effective_out=0.0)  # forward — same as strafe
        self.pid_yaw = PIDController(kp=0.55, ki=0.00, kd=0.01, max_out=0.40)
        self._all_pids = [self.pid_x, self.pid_y, self.pid_z, self.pid_yaw]

        # Tracking state
        self._locked_marker_id     = None   # ID of the marker we are following
        self.latest_goal           = None   # most recent UKF position estimate
        self.is_tracking_lost      = False
        self.last_time             = self.get_clock().now()
        self.last_measurement_time = self.get_clock().now()

        # Angle velocity tracking state
        self.last_yaw_error = 0.0

        # Marker's measured facing direction (camera-frame heading, radians).
        # Replaces the old reliance on self.ukf.x[4], which only ever modeled
        # the travel direction of a mostly-static point, not the marker's
        # actual orientation.
        self.latest_marker_heading = 0.0

        # Re-acquisition soft-start
        self._reacquire_time = None         # timestamp of last re-acquisition

        self.control_timer = self.create_timer(self.CONTROL_DT, self._control_loop)

        self._trigger_takeoff()

    # ── ROS callbacks ─────────────────────────────────────────────────────────

    def _trigger_takeoff(self) -> None:
        self.get_logger().info("Waiting before takeoff…")
        time.sleep(1.2)
        self.takeoff_pub.publish(Empty())
        self.get_logger().info("Takeoff command sent.")

    def _pose_callback(self, msg: PoseStamped) -> None:
        """
        Receive a marker-pose measurement, filter it through the UKF,
        and update self.latest_goal with the estimated position.

        frame_id format:  "<marker_id>"  (only the integer marker ID is used)
        """
        current_time = self.get_clock().now()

        # ── Parse frame_id ────────────────────────────────────────────────────
        parts       = msg.header.frame_id.split(":")
        incoming_id = int(parts[0]) if parts[0].isdigit() else -1

        # ── Marker ID lock ────────────────────────────────────────────────────
        if self._locked_marker_id is None:
            self._locked_marker_id = incoming_id
            self.get_logger().info(f"Locked onto marker ID {self._locked_marker_id}.")
        elif incoming_id != self._locked_marker_id:
            return  # ignore measurements from other markers

        # ── Time delta ────────────────────────────────────────────────────────
        dt = (current_time - self.last_time).nanoseconds / 1e9
        if dt <= 0:
            dt = self.CONTROL_DT
        self.last_time             = current_time
        self.last_measurement_time = current_time

        # ── Raw marker pose → stand-off goal ──────────────────────────────────
        marker_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ])
        q = msg.pose.orientation
        R_marker = quaternion_to_rotation_matrix(q.x, q.y, q.z, q.w)

        # The point the drone should actually hold at: 1m behind / 45cm above
        # the marker, along the marker's own axes -- however it's rotated.
        z_measured = compute_stand_off_goal(marker_pos, R_marker)

        # The marker's real measured facing direction, used for the
        # tangential/radial frame rotation and for yaw (replaces self.ukf.x[4]).
        self.latest_marker_heading = marker_heading_from_rotation(R_marker)

        # ── Re-acquisition reset ──────────────────────────────────────────────
        if self.is_tracking_lost:
            # Reinitialise UKF directly from the fresh measurement so that
            # free-running drift (during lost period) does not cause a
            # position 'snap' when the first fresh measurement arrives.
            self.ukf.reinitialise(z_measured)
            for pid in self._all_pids:
                pid.reset()

            # Synchronize baseline yaw angle on re-acquisition to prevent derivative spikes
            cos_h, sin_h    = np.cos(self.latest_marker_heading), np.sin(self.latest_marker_heading)
            init_tangential = cos_h * z_measured[0] - sin_h * z_measured[2]
            init_radial     = sin_h * z_measured[0] + cos_h * z_measured[2]
            self.last_yaw_error = np.arctan2(init_tangential, init_radial)

            self._reacquire_time = current_time
            self.get_logger().info("Target re-acquired — UKF reset, soft-start active.")

        self.is_tracking_lost = False

        # ── UKF predict → update ─────────────────────────────────────────────
        # Note: the UKF now filters the *stand-off goal point*, same as before
        # -- just computed from the marker's real pose instead of received
        # pre-baked from the detector.
        self.ukf.predict(dt)
        self.ukf.update(z_measured)
        self.latest_goal = self.ukf.x[0:3]

    def _control_loop(self) -> None:
        """
        20 Hz control loop. Rotates the UKF position error into the marker's
        heading frame and publishes a Twist command for each axis.
        """
        now = self.get_clock().now()
        dt  = self.CONTROL_DT

        # ── No measurement yet ────────────────────────────────────────────────
        if self.latest_goal is None:
            self._reset_and_stop()
            self.is_tracking_lost = True
            return

        # ── Measurement staleness check ───────────────────────────────────────
        time_since_meas = (now - self.last_measurement_time).nanoseconds / 1e9

        if time_since_meas > self.TRACKING_LOST_WARN:
            self.is_tracking_lost = True

        if time_since_meas > self.TRACKING_LOST_HALT:
            self.get_logger().error("Target lost > 5 s — safety stop.")
            self._reset_and_stop()
            self.is_tracking_lost = True
            return

        # ── Position errors in world/camera frame ─────────────────────────────
        err_x, err_y, err_z = self.latest_goal

        # ── Rotate errors into marker's own frame ─────────────────────────────
        # Uses the marker's actual measured heading (from its detected
        # orientation). Projecting the position error onto the marker's
        # tangential and radial axes means the drone corrects relative to
        # where the marker is *facing*, not a fixed world direction — so a
        # rotating marker doesn't cause lateral drift.
        marker_heading = self.latest_marker_heading
        cos_h, sin_h   = np.cos(marker_heading), np.sin(marker_heading)
        err_tangential =  cos_h * err_x - sin_h * err_z  # lateral in marker frame
        err_radial     =  sin_h * err_x + cos_h * err_z  # depth   in marker frame

        # ── Build Twist command ───────────────────────────────────────────────
        twist = Twist()

        # Altitude hold: positive err_y → drone too low → climb
        twist.linear.z = -self.pid_y.compute(err_y, dt)

        # Depth hold in marker frame: drive to the desired stand-off distance
        twist.linear.x = self.pid_z.compute(err_radial, dt)

        # Tangential recentering in marker frame: drives lateral offset to zero
        twist.linear.y = -self.pid_x.compute(err_tangential, dt)

        # ── Yaw control ───────────────────────────────────────────────────────
        twist.angular.z = -self._compute_yaw(err_x, err_z, err_tangential, err_radial, dt)

        # ── Soft-start ramp after re-acquisition ─────────────────────────────
        twist = self._apply_soft_start(twist, now)

        self.cmd_pub.publish(twist)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _compute_yaw(self, err_x: float, err_z: float, err_tangential: float,
                      err_radial: float, dt: float) -> float:
        """
        Blended yaw controller that transitions between two strategies:
        - Small tangential error → pure PID on heading angle (enable fine error fixing to face the marker)
        - Large tangential error → tangential-assist term dominates (tackle robot rotatation with satellite like rotation)

        Angle wrapping uses the shortest-path convention to avoid 360° snaps.
        """

        raw_yaw_error = np.arctan2(err_x, err_z)

        # Continuous shortest-path angle wrapping (fixes the 360-degree snap trap)
        angle_diff = raw_yaw_error - self.last_yaw_error
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
        err_yaw    = self.last_yaw_error + angle_diff

        # Real-time tracking math
        yaw_error_diff    = angle_diff / dt
        is_angle_increasing = np.abs(err_yaw) > np.abs(self.last_yaw_error)

        # Update historical state for the next control cycle
        self.last_yaw_error = err_yaw

        # Dynamic real-time status logging
        self.get_logger().info(
            f"Yaw Error: {err_yaw:.3f} rad | "
            f"Delta: {yaw_error_diff:.3f} rad/s | "
            f"Expanding: {is_angle_increasing}"
        )

        pure_yaw_effort    = self.pid_yaw.compute(err_yaw, dt)
        tangential_assist  = self.pid_x.compute(err_tangential, dt) / (1.0 + max(0.0, err_radial))
        weight             = np.clip((np.abs(err_tangential) - 0.2) / 0.1, 0.0, 1.0)

        return (1.0 - weight) * pure_yaw_effort + weight * tangential_assist

    def _apply_soft_start(self, twist: Twist, now) -> Twist:
        """Scale all velocity commands by a linear ramp for SOFT_START_DURATION
        seconds after a re-acquisition event to avoid sudden motion commands."""
        if self._reacquire_time is None:
            return twist

        elapsed = (now - self._reacquire_time).nanoseconds / 1e9
        if elapsed >= self.SOFT_START_DURATION:
            self._reacquire_time = None
            return twist

        ramp            = elapsed / self.SOFT_START_DURATION
        twist.linear.x  *= ramp
        twist.linear.y  *= ramp
        twist.linear.z  *= ramp
        twist.angular.z *= ramp
        return twist

    def _reset_and_stop(self) -> None:
        """Publish a zero-velocity command and reset all PID state."""
        for pid in self._all_pids:
            pid.reset()
        self.cmd_pub.publish(Twist())

    def stop_drone(self) -> None:
        self.cmd_pub.publish(Twist())


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = TelloNavigationController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_drone()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
