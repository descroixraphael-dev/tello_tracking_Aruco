import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist, Vector3Stamped
from rclpy.node import Node
from scipy.linalg import cholesky
from std_msgs.msg import Empty

# ═════════════════════════════════════════════════════════════════════════
#  SINGLE-AXIS TEST CONTROLLER
#
#  Only ONE of x / y / z / yaw is ever commanded — the rest are forced to
#  zero. This isolates one PID at a time for tuning (e.g. checking the
#  pid_y altitude-hold fix in isolation) without interaction from the
#  other axes or the coarse-yaw zone logic.
#
#  FLIGHT SEQUENCE:
#    1. ALTITUDE_CALIBRATE — pid_y alone drives to the target height, once,
#       like a calibration step. Once settled, altitude is left to the
#       Tello's own onboard hover/barometer hold and is never re-driven.
#    2. AXIS_TEST — whichever single-axis block below is uncommented runs,
#       with altitude no longer actively controlled by this node.
#
#  HOW TO SWITCH AXES:
#  Exactly one block in control_loop()'s AXIS_TEST section should be
#  UNCOMMENTED. Comment out the active block and uncomment a different one
#  to switch. Everything else (UKF, locking, takeoff, safety halt, altitude
#  calibration) stays the same regardless of which axis is under test.
# ═════════════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────

# Stand-off offset (metres), expressed in the MARKER's own local frame.
# Same convention as orbit_nav.py / ukf_navigation.py:
#   Marker flat on floor facing up:
#     +Z  →  pointing straight up into the ceiling.
#     -Y  →  pointing backward along the ground towards the drone's parking spot.
# Applying this through the marker's measured rotation matrix means the
# drone holds 1m behind / 45cm above the marker regardless of how the
# marker itself is rotated on the floor.
OFFSET_TOWARD_M = 1.0   # desired stand-off distance behind the marker (m)
OFFSET_NORMAL_M = 0.35  # desired height above the marker (m)

# Altitude-calibration settle criteria (see FLIGHT SEQUENCE above).
#   ALTITUDE_SETTLE_TOL_M   : how close err_y must get to call it "settled" (m)
#   ALTITUDE_SETTLE_HOLD_SEC: must stay within tolerance this long (debounce,
#                             so a single noisy sample doesn't trigger early)
ALTITUDE_SETTLE_TOL_M    = 0.10
ALTITUDE_SETTLE_HOLD_SEC = 1.0

# Tracking-loss recovery timeline. No ML-based re-detection here -- ArUco
# detection is a simple per-frame binary hit/miss, so recovery is a scripted,
# timer-driven state machine rather than anything adaptive:
#   < LOSS_HOLD_SEC                   : normal tracking
#   LOSS_HOLD_SEC .. LOSS_SEARCH_SEC  : assume transient (blur/occlusion) -> hover in place
#   LOSS_SEARCH_SEC .. LOSS_ABORT_SEC : marker likely out of frame -> slow scan rotation
#   > LOSS_ABORT_SEC                  : give up -> zero output + land
# These are starting points, not measured values -- tune against how your
# own detector actually drops frames (occasional single-frame misses vs.
# the marker genuinely leaving the FOV) before trusting this on real flights.
LOSS_HOLD_SEC   = 2.0
LOSS_SEARCH_SEC = 5.0
LOSS_ABORT_SEC  = 20.0
SEARCH_YAW_RATE = 0.50  # Twist.angular.z while scanning -- keep conservative, this is untested


# ─────────────────────────────────────────────────────────────────────────
#  GEOMETRY HELPERS
# ─────────────────────────────────────────────────────────────────────────

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
    Same as orbit_nav.py / ukf_navigation.py, so this benchmark targets the
    same goal point the real controller flies to.
    """
    local_offset = np.array([0.0, -OFFSET_TOWARD_M, OFFSET_NORMAL_M])
    return marker_pos + R_marker @ local_offset


# ─────────────────────────────────────────────────────────────────────────
#  STATE ESTIMATION
# ─────────────────────────────────────────────────────────────────────────

class UKF_CTRV:
    """
    Unscented Kalman Filter using a Constant Turn Rate and Velocity (CTRV) Model.
    State vector x: [pos_x, pos_y, pos_z, velocity, heading_theta, turn_rate_omega]
    """

    def __init__(self, dt=0.05):
        self.dt = dt
        self.num_states = 6

        self.x = np.zeros(self.num_states)
        self.P = np.eye(self.num_states) * 1.0

        self.Q = np.diag([0.05, 0.05, 0.05, 0.1, 0.05, 0.05])
        self.R = np.diag([0.02, 0.02, 0.02])

        self.alpha = 1e-3
        self.kappa = 0.0
        self.beta = 2.0
        self.lambda_ = (self.alpha**2) * (self.num_states + self.kappa) - self.num_states

        n = self.num_states
        self.weights_m = np.full(2 * n + 1, 1.0 / (2.0 * (n + self.lambda_)))
        self.weights_c = self.weights_m.copy()
        self.weights_m[0] = self.lambda_ / (n + self.lambda_)
        self.weights_c[0] = self.weights_m[0] + (1 - self.alpha**2 + self.beta)

    def _generate_sigma_points(self):
        n = self.num_states
        sigma = np.zeros((n, 2 * n + 1))
        sigma[:, 0] = self.x

        try:
            L = cholesky((n + self.lambda_) * self.P, lower=True)
        except np.linalg.LinAlgError:
            L = cholesky((n + self.lambda_) * (self.P + np.eye(n) * 1e-6), lower=True)

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

    def predict(self, dt):
        self.dt = dt
        sigma_pts = self._generate_sigma_points()
        self.sigma_pred = self._apply_ctrv(sigma_pts)
        self.x = np.sum(self.weights_m * self.sigma_pred, axis=1)

        self.P = self.Q.copy()
        for i in range(2 * self.num_states + 1):
            diff = (self.sigma_pred[:, i] - self.x).reshape(-1, 1)
            self.P += self.weights_c[i] * (diff @ diff.T)

    def update(self, z):
        z_pred_pts = self.sigma_pred[0:3, :]
        z_pred = np.sum(self.weights_m * z_pred_pts, axis=1)

        S  = self.R.copy()
        Tc = np.zeros((self.num_states, 3))

        for i in range(2 * self.num_states + 1):
            z_diff = (z_pred_pts[:, i] - z_pred).reshape(-1, 1)
            x_diff = (self.sigma_pred[:, i] - self.x).reshape(-1, 1)
            S  += self.weights_c[i] * (z_diff @ z_diff.T)
            Tc += self.weights_c[i] * (x_diff @ z_diff.T)

        K = Tc @ np.linalg.inv(S)
        self.x += K @ (z - z_pred)
        self.P -= K @ S @ K.T
        self.P = 0.5 * (self.P + self.P.T)


# ─────────────────────────────────────────────────────────────────────────
#  CONTROL
# ─────────────────────────────────────────────────────────────────────────

class PIDController:
    def __init__(self, kp, ki, kd, max_out, min_effective_out=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_out = max_out
        # Smallest |output| that actually produces motion (motor/RC deadband).
        # Measure empirically per axis; leave at 0.0 to disable. See note below.
        self.min_effective_out = min_effective_out
        self.integral = 0.0
        self.last_error = 0.0

    def reset(self):
        self.integral = 0.0
        self.last_error = 0.0

    def compute(self, error, dt):
        self.integral += error * dt

        derivative = (error - self.last_error) / dt if dt > 0 else 0.0
        self.last_error = error

        output_unclamped = self.kp * error + self.ki * self.integral + self.kd * derivative
        output = np.clip(output_unclamped, -self.max_out, self.max_out)

        # Anti-windup
        if self.ki and output != output_unclamped:
            self.integral -= error * dt

        # Deadband compensation
        if 0.0 < abs(output) < self.min_effective_out:
            output = self.min_effective_out * np.sign(output)

        return output


# ─────────────────────────────────────────────────────────────────────────
#  NODE
# ─────────────────────────────────────────────────────────────────────────

class SingleAxisTestController(Node):

    # -- setup -------------------------------------------------------------

    def __init__(self):
        super().__init__('tello_single_axis_test')

        self._init_pubs_subs()
        self._init_debug_telemetry()
        self._init_estimator_and_control()
        self._init_state()

        self.trigger_takeoff()

    def _init_pubs_subs(self):
        self.pose_sub    = self.create_subscription(PoseStamped, '/tello/marker_pose', self.pose_callback, 10)
        self.cmd_pub     = self.create_publisher(Twist, '/cmd_vel', 10)
        self.takeoff_pub = self.create_publisher(Empty, '/tello/takeoff', 10)
        self.land_pub    = self.create_publisher(Empty, '/tello/land', 10)

    def _init_debug_telemetry(self):
        # UKF's continuous-valued speed estimate, for comparing against the
        # integer-quantized /tello/velocity in PlotJuggler. UKF speed state is
        # metres/second (marker/vision frame); Tello telemetry is cm/s -- the
        # x100 below is an exact unit conversion so both plot on the same axis.
        self.velocity_estimate_pub = self.create_publisher(Vector3Stamped, '/tello/nav/velocity_estimate', 10)
        self.M_TO_CM = 100.0

        # UKF turn-rate state (omega) is rad/s; Tello's /tello/attitude reports
        # yaw in degrees. Converting here so both are directly comparable --
        # note the SDK has NO yaw-rate telemetry field, so this topic has no
        # raw quantized counterpart the way /tello/velocity does.
        self.yaw_rate_estimate_pub = self.create_publisher(Vector3Stamped, '/tello/nav/yaw_rate_estimate', 10)
        self.RAD_TO_DEG = 180.0 / np.pi

    def _init_estimator_and_control(self):
        self.ukf = UKF_CTRV()
        self.last_time = self.get_clock().now()

        # min_effective_out is left at 0.0 (disabled) below. Measure the real
        # per-axis motor/RC deadband (smallest cmd_vel that produces visible
        # motion) and set it explicitly once known — see PIDController.compute().
        self.pid_x   = PIDController(kp=0.0490, ki=0.0048, kd=0.0146, max_out=0.90, min_effective_out=0.10)
        self.pid_y   = PIDController(kp=0.0450, ki=0.0041, kd=0.0146, max_out=0.90, min_effective_out=0.15)
        self.pid_z   = PIDController(kp=0.0490, ki=0.0048, kd=0.0146, max_out=0.90, min_effective_out=0.10)
        self.pid_yaw = PIDController(kp=0.2547, ki=0.0283, kd=0.0669, max_out=0.50, min_effective_out=0.18)
        self._all_pids = [self.pid_x, self.pid_y, self.pid_z, self.pid_yaw]

        self.control_timer = self.create_timer(0.05, self.control_loop)

    def _init_state(self):
        self.latest_goal = None
        # Raw marker position (camera frame), NOT the stand-off goal below.
        # Used for yaw bearing only -- translation control still targets
        # latest_goal (the offset stand-off point). Unfiltered (no UKF),
        # so expect it to be noisier frame-to-frame than latest_goal.
        self.latest_marker_pos = None
        self.last_measurement_time = self.get_clock().now()
        self.is_tracking_lost = True

        self._recovery_state = 'TRACKING'          # TRACKING | HOLD | SEARCHING | ABORTED
        self._flight_state = 'ALTITUDE_CALIBRATE'  # ALTITUDE_CALIBRATE | AXIS_TEST
        self._altitude_settle_start_time = None

        self.current_zone = 'CENTER'
        self._locked_marker_id = None

    def trigger_takeoff(self):
        self.get_logger().info("Initializing automatic drone takeoff process...")
        time.sleep(1.2)
        self.takeoff_pub.publish(Empty())
        self.get_logger().info("Takeoff command published to /tello/takeoff!")

    # -- perception ----------------------------------------------------------

    def pose_callback(self, msg: PoseStamped):
        current_time = self.get_clock().now()

        frame_id = msg.header.frame_id
        parts = frame_id.split(":")
        incoming_id = int(parts[0]) if parts[0].isdigit() else -1

        if self._locked_marker_id is None:
            self._locked_marker_id = incoming_id
            self.get_logger().info(f"Locked onto marker ID {self._locked_marker_id}")
        elif incoming_id != self._locked_marker_id:
            return

        self.current_zone = parts[1] if len(parts) > 1 else 'CENTER'

        dt = (current_time - self.last_time).nanoseconds / 1e9
        if dt <= 0:
            dt = 0.05
        self.last_time = current_time
        self.last_measurement_time = current_time

        if self.is_tracking_lost:
            for pid in self._all_pids:
                pid.reset()
            self.get_logger().info("Target re-acquired — PID integrals reset.")

        self.is_tracking_lost = False

        # Raw marker pose (camera frame) → same 1m-behind/45cm-above stand-off
        # goal that orbit_nav.py / ukf_navigation.py fly to, so this benchmark
        # is tuning against the same target the real controller uses.
        marker_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ])
        q = msg.pose.orientation
        R_marker = quaternion_to_rotation_matrix(q.x, q.y, q.z, q.w)
        z_measured = compute_stand_off_goal(marker_pos, R_marker)

        # Keep the raw marker position around for yaw bearing (see
        # latest_marker_pos comment in _init_state) -- this must be captured
        # here, BEFORE the stand-off offset above shifts it.
        self.latest_marker_pos = marker_pos

        self.ukf.predict(dt)
        self.ukf.update(z_measured)
        self.latest_goal = self.ukf.x[0:3]

        self._publish_velocity_estimate(current_time)
        self._publish_yaw_rate_estimate(current_time)

    def _publish_velocity_estimate(self, stamp):
        # NOTE: speed relative to the marker (vision frame), not the drone's
        # own body-frame velocity -- see M_TO_CM comment in _init_debug_telemetry.
        speed_m_s, heading_rad = self.ukf.x[3], self.ukf.x[4]
        velocity_estimate = Vector3Stamped()
        velocity_estimate.header.stamp = stamp.to_msg()
        velocity_estimate.header.frame_id = 'marker'
        velocity_estimate.vector.x = speed_m_s * np.cos(heading_rad) * self.M_TO_CM
        velocity_estimate.vector.y = speed_m_s * np.sin(heading_rad) * self.M_TO_CM
        velocity_estimate.vector.z = 0.0  # CTRV model has no independent z-velocity state
        self.velocity_estimate_pub.publish(velocity_estimate)

    def _publish_yaw_rate_estimate(self, stamp):
        turn_rate_rad_s = self.ukf.x[5]
        yaw_rate_estimate = Vector3Stamped()
        yaw_rate_estimate.header.stamp = stamp.to_msg()
        yaw_rate_estimate.header.frame_id = 'marker'
        yaw_rate_estimate.vector.z = turn_rate_rad_s * self.RAD_TO_DEG
        self.yaw_rate_estimate_pub.publish(yaw_rate_estimate)

    # -- tracking-loss recovery ------------------------------------------------

    def _handle_tracking_loss(self, time_since_last_meas: float) -> bool:
        """
        Tiered response to marker dropout, driven purely by elapsed time
        since the last successful detection (no re-detection logic involved).
        Returns True if control_loop() should stop here this tick (i.e. we're
        in some recovery state and the normal axis-select block must not run).
        """
        if time_since_last_meas > LOSS_ABORT_SEC:
            if self._recovery_state != 'ABORTED':
                self.get_logger().error(f"Target lost for > {LOSS_ABORT_SEC}s. Landing.")
                self._recovery_state = 'ABORTED'
                self.is_tracking_lost = True
                self._flight_state = 'ALTITUDE_CALIBRATE'
            self.stop_drone()
            self.land_pub.publish(Empty())
            return True

        if time_since_last_meas > LOSS_SEARCH_SEC:
            if self._recovery_state != 'SEARCHING':
                self.get_logger().warn(
                    f"Target lost for > {LOSS_SEARCH_SEC}s -- marker likely "
                    f"out of frame. Rotating slowly to scan for it."
                )
                self._recovery_state = 'SEARCHING'
                self.is_tracking_lost = True
                self._flight_state = 'ALTITUDE_CALIBRATE'
            search_twist = Twist()
            search_twist.angular.z = SEARCH_YAW_RATE
            self.cmd_pub.publish(search_twist)
            return True

        if time_since_last_meas > LOSS_HOLD_SEC:
            if self._recovery_state != 'HOLD':
                self.get_logger().warn(
                    f"Target lost for > {LOSS_HOLD_SEC}s -- assuming transient "
                    f"occlusion/blur. Holding position."
                )
                self._recovery_state = 'HOLD'
                self.is_tracking_lost = True
            self.cmd_pub.publish(Twist())  # hover; PID state left untouched
            return True

        self._recovery_state = 'TRACKING'
        return False

    # -- control loop --------------------------------------------------------

    def control_loop(self):
        if self.latest_goal is None or self.latest_marker_pos is None:
            self.cmd_pub.publish(Twist())
            for pid in self._all_pids:
                pid.reset()
            return

        now = self.get_clock().now()
        time_since_last_meas = (now - self.last_measurement_time).nanoseconds / 1e9
        dt = 0.05

        if self._handle_tracking_loss(time_since_last_meas):
            return

        # Stand-off goal errors -- drive translation (drone should stop at
        # the offset point, not on top of the marker).
        err_x = self.latest_goal[0]
        err_y = self.latest_goal[1]
        err_z = self.latest_goal[2]

        # Raw marker errors -- drive yaw bearing only. Using the marker's
        # own x/z (instead of the goal's) means the drone points AT the
        # marker regardless of the stand-off offset, which shifts x and z
        # differently depending on how the marker itself is rotated.
        marker_err_x = self.latest_marker_pos[0]
        marker_err_z = self.latest_marker_pos[2]

        if self._flight_state == 'ALTITUDE_CALIBRATE':
            self._run_altitude_calibration(err_y, dt, now)
            return

        self._run_axis_test(err_x, err_y, err_z, marker_err_x, marker_err_z, dt)

    def _run_altitude_calibration(self, err_y, dt, now):
        """
        FSM: ALTITUDE_CALIBRATE runs once, alone, like a calibration step.
        Once settled, altitude is left alone for good (relies on the Tello's
        own onboard hover/barometer hold) and control moves permanently to
        AXIS_TEST -- no going back except on tracking loss.
        """
        altitude_twist = Twist()
        altitude_twist.linear.z = -self.pid_y.compute(err_y, dt)
        self.cmd_pub.publish(altitude_twist)

        if abs(err_y) < ALTITUDE_SETTLE_TOL_M:
            if self._altitude_settle_start_time is None:
                self._altitude_settle_start_time = now
            elif (now - self._altitude_settle_start_time).nanoseconds / 1e9 > ALTITUDE_SETTLE_HOLD_SEC:
                self._flight_state = 'AXIS_TEST'
                self.get_logger().info(
                    "Altitude calibrated -- switching to AXIS_TEST, "
                    "altitude no longer actively re-driven."
                )
                self.cmd_pub.publish(Twist())
        else:
            self._altitude_settle_start_time = None  # drifted back out, reset debounce

    def _run_axis_test(self, err_x, err_y, err_z, marker_err_x, marker_err_z, dt):
        """
        ACTIVE AXIS SELECTION (AXIS_TEST state).
        Exactly ONE block below should be uncommented at a time.
        All other axes stay at 0.0 — only the active one moves the drone.
        Altitude is intentionally NOT re-driven here -- it was calibrated
        once in ALTITUDE_CALIBRATE and is left to the Tello's own hold.

        err_x/err_y/err_z    : stand-off GOAL errors -- drive translation.
        marker_err_x/_z      : raw MARKER errors -- drive yaw bearing only,
                                so the drone faces the marker itself rather
                                than the offset point it's translating to.
        """
        twist = Twist()

        # ── X AXIS (forward / back, pid_z, err_z) ──────────────────────────
        # twist.linear.x  = self.pid_z.compute(err_z, dt)
        # self.pid_yaw.reset()
        # self.pid_y.reset()
        # self.pid_z.reset()

        # ── Y AXIS (strafe left/right, pid_x, err_x) ────────────────────────
        # twist.linear.x  = 0.0
        # twist.linear.y  = self.pid_x.compute(err_x, dt)
        # twist.linear.z  = 0.0
        # twist.angular.z = 0.0
        # self.pid_z.reset()
        # self.pid_y.reset()
        # self.pid_yaw.reset()

        # ── Z AXIS (altitude, pid_y, err_y) ─────────────────────────────────
        # twist.linear.x  = 0.0
        # twist.linear.y  = 0.0
        # twist.linear.z  = self.pid_y.compute(err_y, dt)
        # twist.angular.z = 0.0
        # self.pid_x.reset()
        # self.pid_z.reset()
        # self.pid_yaw.reset()

        # ── YAW AXIS (rotation, pid_yaw, err_tangential/err_radial) ─────────
        yaw_error = np.arctan2(marker_err_x, marker_err_z)
        twist.linear.x  = 0.0
        twist.linear.y  = 0.0
        twist.linear.z  = 0.0
        twist.angular.z = self.pid_yaw.compute(yaw_error, dt)
        self.pid_x.reset()
        self.pid_y.reset()
        self.pid_z.reset()

        self.cmd_pub.publish(twist)

    def stop_drone(self):
        self.cmd_pub.publish(Twist())


# ─────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = SingleAxisTestController()
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
