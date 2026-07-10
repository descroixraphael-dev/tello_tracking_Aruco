# `single_axis.py` — Documentation

## 1. General Overview

This script is a **single-axis PID tuning bench** for the Tello drone. Instead of
running the full navigation controller (all four axes — lateral, altitude, depth,
yaw — acting together), it isolates one axis at a time so that its PID gains can
be tuned and observed without interference from the other three.

The pipeline on every marker detection is:

1. **Receive** a raw marker pose from `/tello/marker_pose` (published by
   `aruco_detector.py`).
2. **Convert** that raw pose into the same *1m-behind / 45cm-above* stand-off
   goal point  using the marker's own orientation.
3. **Filter** that goal point through a UKF (Unscented Kalman Filter, CTRV
   motion model) to smooth out noisy vision measurements and estimate
   velocity/heading/turn-rate.
4. **Feed** the filtered position error into whichever axis's PID block is
   currently active in `control_loop()`, and publish the resulting velocity
   command on `/cmd_vel`.
5. A 20 Hz timer drives the control loop; a safety check stops the drone if
   the marker hasn't been seen for more than 5 seconds.

Only one block in `control_loop()` is meant to be active (uncommented) at a
time — the file is a bench, not a flight-ready controller.

---

## 2. Module-Level Constants

| Name | Value | Meaning |
|---|---|---|
| `OFFSET_TOWARD_M` | `1.0` | Desired stand-off distance behind the marker, in metres, measured along the marker's own local frame. |
| `OFFSET_NORMAL_M` | `0.45` | Desired height above the marker, in metres, along the marker's own local frame. |

These are combined into a single offset vector inside `compute_stand_off_goal`
(see below) and rotated into the camera frame using the marker's measured
orientation, so the stand-off point stays correct even if the marker itself is
rotated.

**Note:** the convention assumes the marker is lying flat, facing up (its
local +Z axis points at the ceiling, its local -Y axis points back toward the
drone's parking spot).

---

## 3. Module-Level Functions

### `quaternion_to_rotation_matrix(qx, qy, qz, qw) -> np.ndarray`
Converts a quaternion (as received in `msg.pose.orientation`) into a 3×3
rotation matrix, using the standard closed-form quaternion→matrix formula.
Used to express the marker's measured orientation as a matrix so the stand-off
offset (defined in the marker's own local frame) can be rotated into the
camera frame.

### `compute_stand_off_goal(marker_pos, R_marker) -> np.ndarray`
- **`marker_pos`**: the marker's raw `[x, y, z]` position, in the camera
  frame, as reported by the detector.
- **`R_marker`**: the marker's orientation as a rotation matrix (output of
  `quaternion_to_rotation_matrix`).
- **Returns**: the 3D point the drone should actually try to hold at — the
  marker position plus the `[0, -OFFSET_TOWARD_M, OFFSET_NORMAL_M]` offset,
  rotated by `R_marker` so it's expressed correctly regardless of how the
  marker is rotated.


---

## 4. `UKF_CTRV` class — the state estimator

Implements an Unscented Kalman Filter with a **Constant Turn Rate and
Velocity (CTRV)** motion model. This models a target moving in a horizontal
plane at some speed `v`, with heading `theta` that can rotate at rate
`omega` — a natural model for something (in this case, the "target point" the
drone is trying to hold at) which can drift and turn.

**State vector `self.x`** (length 6):

| Index | Name | Meaning |
|---|---|---|
| 0 | `pos_x` | Lateral position (camera-frame X) |
| 1 | `pos_y` | Vertical position / altitude (camera-frame Y) |
| 2 | `pos_z` | Depth position (camera-frame Z, "forward") |
| 3 | `v` | Speed in the horizontal (X-Z) plane |
| 4 | `theta` | Heading angle in the horizontal (X-Z) plane |
| 5 | `omega` | Turn rate (rad/s) |

**Key attributes:**
- `self.P` — state covariance matrix (6×6), representing uncertainty in the
  state estimate.
- `self.Q` — process noise covariance (how much we expect the true state to
  drift between predictions, independent of measurement).
- `self.R` — measurement noise covariance (how much we trust a single vision
  measurement).
- `self.alpha`, `self.kappa`, `self.beta`, `self.lambda_` — the van der Merwe
  sigma-point spread parameters that control how sigma points are spaced
  around the mean during the unscented transform.
- `self.weights_m`, `self.weights_c` — the weights applied to each sigma point
  when reconstructing the predicted mean and covariance.

**Methods:**

- **`_generate_sigma_points()`**: builds `2n+1 = 13` sigma points around the
  current state estimate `self.x`, spread according to the Cholesky
  decomposition of the (scaled) covariance `self.P`. Falls back to a
  regularized version of `P` if the Cholesky decomposition fails (i.e., `P`
  isn't positive-definite due to numerical drift).

- **`_apply_ctrv(sigma_pts)`**: propagates every sigma point forward by `dt`
  using the CTRV motion equations. Heading `theta` is measured in the
  camera's X-Z plane (0 pointing along +Z / "forward"), matching the
  convention used elsewhere in the codebase (e.g. `atan2(err_x, err_z)` for
  yaw). When turn rate `omega` is near zero, it falls back to a straight-line
  (constant velocity) approximation to avoid dividing by zero. Altitude
  (`pos_y`) is explicitly **not** part of this rotational plane — it's carried
  through unchanged, since climbing/descending has nothing to do with
  horizontal heading dynamics.

- **`predict(dt)`**: the UKF time-update step. Generates sigma points,
  propagates them through `_apply_ctrv`, recombines them (weighted by
  `weights_m`/`weights_c`) into a new mean `self.x` and covariance `self.P`,
  and adds process noise `self.Q`.

- **`update(z)`**: the UKF measurement-update step. `z` is a 3-element
  `[x, y, z]` position measurement (the stand-off goal point computed
  upstream). Projects the predicted sigma points into measurement space
  (just the first 3 state components — position), computes the innovation
  covariance `S` and cross-covariance `Tc`, derives the Kalman gain `K`, and
  corrects `self.x` and `self.P` accordingly. Enforces symmetry on `P`
  afterward to guard against numerical asymmetry accumulating over time.

---

## 5. `PIDController` class

A standard discrete PID controller with anti-windup and an (optional) output
deadband compensation.

**Constructor arguments:**
- `kp`, `ki`, `kd` — proportional/integral/derivative gains.
- `max_out` — symmetric output clamp (command saturates at ±`max_out`).
- `min_effective_out` — smallest |output| that will actually produce motion
  on the real drone (motor/RC deadband). Left at `0.0` disables this
  compensation; should be measured empirically per axis.

**Internal state:**
- `self.integral` — running sum of `error * dt`.
- `self.last_error` — previous cycle's error, used for the derivative term.

**Methods:**
- **`reset()`**: zeroes `integral` and `last_error`. Called whenever tracking
  is lost/re-acquired (to avoid a stale integral term causing a lurch), and
  also called on axes that are *not* currently active in `control_loop()` so
  their internal state doesn't drift while unused.
  
- **`compute(error, dt)`**: the standard PID update —
  1. Accumulate the integral.
  2. Compute the derivative from the change in error since last call.
  3. Compute the raw (unclamped) output as `kp*error + ki*integral + kd*derivative`.
  4. Clamp the output to `±max_out`.
  5. **Anti-windup**: if clamping actually changed the output, undo the
     integral accumulation from this step so the integral doesn't keep
     growing while saturated.
  6. **Deadband compensation** : if the
     output is nonzero but smaller in magnitude than
     `min_effective_out`, bump it up to `min_effective_out` with the correct
     sign, so small commands aren't silently absorbed by motor deadband.

---

## 6. `SingleAxisTestController` class — the ROS 2 node

### `__init__(self)`
Sets up:
- **Subscriptions/publishers**: subscribes to `/tello/marker_pose`
  (`pose_callback`), publishes `/cmd_vel` (velocity commands) and
  `/tello/takeoff` (triggers takeoff once at startup). Also publishes two
  
  debug topics for PlotJuggler:
  - `/tello/nav/velocity_estimate` — the UKF's continuous speed estimate,
    decomposed into X/Y components and converted from m/s to cm/s
    (`self.M_TO_CM = 100.0`) so it's directly comparable to the Tello's
    integer-quantized `/tello/velocity` telemetry.
  - `/tello/nav/yaw_rate_estimate` — the UKF's turn-rate estimate, converted
    from rad/s to deg/s (`self.RAD_TO_DEG`) for comparison against
    `/tello/attitude` (noting the Tello SDK has no raw yaw-rate telemetry to
    compare against).
    
- **`self.ukf`**: one `UKF_CTRV` instance shared across all axes.

- **Four `PIDController` instances** — `pid_x` (lateral/strafe), `pid_y`
  (altitude), `pid_z` (forward/depth), `pid_yaw` (rotation) — each with
  independently tuned gains, plus `self._all_pids`, a list of all four used
  for bulk reset operations.
  
- **Tracking/state variables**: `self.latest_goal` (most recent UKF position
  estimate, `None` until the first measurement arrives), `self.is_tracking_lost`
  (starts `True` so the very first measurement is treated as a fresh
  acquisition), `self.current_zone` (left/center/right zone parsed from the
  detector's `frame_id`, currently unused by the active control block),
  `self._locked_marker_id` (the marker ID this run has locked onto — later
  detections of other IDs are ignored).
- Starts a 20 Hz (`0.05`s period) control timer and triggers takeoff.

### `trigger_takeoff(self)`
Logs, sleeps 1.2s (to give the node time to fully initialize before flight),
then publishes an `Empty` message to `/tello/takeoff`.

### `pose_callback(self, msg: PoseStamped)`
Runs every time a new marker pose arrives. In order:
1. **Marker ID lock**: parses `incoming_id` from `msg.header.frame_id`
   (format `"<id>:<zone>"`). Locks onto the first ID seen; ignores all
   subsequent detections of any other ID for the rest of the run.
2. **Zone parsing**: extracts the left/center/right zone string (currently
   informational only — not used by the active X-axis test block).
3. **`dt` computation**: time since the last pose callback, in seconds.
   Falls back to `0.05` if `dt` comes out non-positive (e.g. clock or
   simulation-time weirdness).
4. **Re-acquisition handling**: if `self.is_tracking_lost` was `True`
   (either the very first callback, or after a gap), resets all four PIDs
   so stale integral terms don't cause a lurch when the target reappears.
5. **Stand-off goal computation**: builds `marker_pos` from the raw pose,
   converts `msg.pose.orientation` to a rotation matrix, and calls
   `compute_stand_off_goal` to get `z_measured` — the actual target point fed
   into the UKF.
6. **UKF update**: calls `predict(dt)` then `update(z_measured)`, and stores
   the resulting position estimate in `self.latest_goal`.
7. **Debug publishing**: publishes the velocity and yaw-rate estimate topics
   described above.

### `control_loop(self)`
Runs at 20 Hz via the timer:
1. If no measurement has arrived yet (`self.latest_goal is None`), publishes
   a zero `Twist`, resets all PIDs, and returns.
   
2. **Staleness/safety check**: if more than 5.0 seconds have passed since the
   last measurement, logs an error and calls `stop_drone()` (publishes zero
   `Twist`) — this is the hard safety cutoff. (There's a second, softer
   staleness check at 1.0s left commented out — it would mark tracking as
   lost and free-run the UKF prediction without a fresh measurement, but it's
   currently disabled.)
3. Unpacks `err_x`, `err_y`, `err_z` from `self.latest_goal` — these are the
   filtered position errors (target minus current, in camera-frame X/Y/Z)
   the active PID block will drive toward zero.
4. **Active axis block**: exactly one of the four commented/uncommented
   blocks below should be live at a time:
   - **X AXIS** : drives `pid_z` on `err_z` for
     forward/back motion.
   - **Y AXIS** : pure strafe test — `pid_x` on `err_x` only.
   - **Z AXIS** : pure altitude test — `pid_y` on `err_y`
     only, with the output negated (`-self.pid_y.compute(...)`).
   - **YAW AXIS** : pure yaw test — computes marker-frame
     tangential/radial errors and drives `pid_yaw` on `atan2(err_x, err_z)`.
5. Publishes the resulting `twist` on `/cmd_vel`.

### `stop_drone(self)`
Publishes a zero-velocity `Twist` — used both on the safety timeout and on
node shutdown.

### `main(args=None)`
Standard ROS 2 entry point: initializes rclpy, spins the node, and on
shutdown (including `KeyboardInterrupt`) calls `stop_drone()` before
destroying the node.

---

