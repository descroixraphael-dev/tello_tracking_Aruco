# `single_axis.py` — Documentation

## 1. General Overview

This script is a **single-axis PID tuning bench** for the Tello drone. Instead of
running the full navigation controller (all four axes — lateral, altitude, depth,
yaw — acting together), it isolates one axis at a time so its PID gains can be
tuned and observed without interference from the other three.

**Flight sequence** (`self._flight_state`):
1. **`ALTITUDE_CALIBRATE`** — `pid_y` alone drives to the target height, once,
   like a calibration step. Once settled (within tolerance, for a debounce
   hold time), altitude is handed off to the Tello's own onboard
   hover/barometer hold.
2. **`AXIS_TEST`** — whichever single-axis block is uncommented in
   `_run_axis_test()` runs, with altitude no longer commanded by this node.

A long enough tracking loss (`SEARCHING` or `ABORTED`, see §2) forces
`_flight_state` back to `ALTITUDE_CALIBRATE` on re-acquisition, so a
mid-flight recovery re-does the altitude calibration step rather than
resuming `AXIS_TEST` blind.

The pipeline on every marker detection is:

1. **Receive** a raw marker pose from `/tello/marker_pose` (published by
   `aruco_detector.py`).
2. **Convert** that raw pose into the same *1m-behind / 0.35m-above*
   stand-off goal point using the marker's own measured orientation
   (`compute_stand_off_goal`).
3. **Filter** that goal point through a UKF (Unscented Kalman Filter, CTRV
   motion model) to smooth out noisy vision measurements and estimate
   velocity/heading/turn-rate.
4. **Feed** the filtered stand-off goal error into whichever axis's PID
   block is currently active for **translation**, but drive **yaw** off the
   *raw, unfiltered marker position* instead — so the drone points at the
   marker itself rather than at the offset stand-off point (which shifts
   off the marker's true bearing whenever the marker is rotated relative to
   the camera). Publish the resulting velocity command on `/cmd_vel`.
5. A 20 Hz timer (`control_loop`) drives control. A tiered, timer-driven
   recovery FSM (§ "Tracking-loss recovery" below) handles marker dropout —
   hover, then scan, then land — rather than a single hard cutoff.

Only one block in `_run_axis_test()` is meant to be active (uncommented) at
a time — this file is a bench, not a flight-ready controller.

---

## 2. Module-Level Constants

| Name | Value | Meaning |
|---|---|---|
| `OFFSET_TOWARD_M` | `1.0` | Desired stand-off distance behind the marker (m), measured along the marker's own local frame. |
| `OFFSET_NORMAL_M` | `0.35` | Desired height above the marker (m), applied in **world frame** — see `compute_stand_off_goal` note below. |
| `ALTITUDE_SETTLE_TOL_M` | `0.10` | How close `err_y` must get to call `ALTITUDE_CALIBRATE` "settled" (m). |
| `ALTITUDE_SETTLE_HOLD_SEC` | `1.0` | Must stay within `ALTITUDE_SETTLE_TOL_M` this long before switching to `AXIS_TEST` (debounce, so one noisy sample doesn't trigger early). |
| `LOSS_HOLD_SEC` | `2.0` | Below this, tracking is normal. Between this and `LOSS_SEARCH_SEC`: assume transient occlusion/blur — hover in place. |
| `LOSS_SEARCH_SEC` | `5.0` | Above this (and below `LOSS_ABORT_SEC`): marker likely out of frame — rotate slowly to scan for it. |
| `LOSS_ABORT_SEC` | `20.0` | Above this: give up — zero output and land. |
| `SEARCH_YAW_RATE` | `0.50` | `Twist.angular.z` used while scanning in the `SEARCHING` state. Conservative default. |

`OFFSET_TOWARD_M` and `OFFSET_NORMAL_M` are combined inside
`compute_stand_off_goal` (see below): `OFFSET_TOWARD_M` is rotated into the
camera frame using the marker's measured orientation so the horizontal
stand-off stays correct regardless of how the marker is rotated;
`OFFSET_NORMAL_M` is **not** rotated — it's applied as a fixed world-frame
offset (see the sign/coupling note in §3).

The tracking-loss constants above encode a purely time-driven state machine
— there's no re-detection heuristic, since ArUco detection is a simple
per-frame hit/miss. These are starting points, not measured values; tune
them against how your own detector actually drops frames (occasional
single-frame misses vs. the marker genuinely leaving the FOV).

---

## 3. Module-Level Functions

### `quaternion_to_rotation_matrix(qx, qy, qz, qw) -> np.ndarray`
Converts a quaternion (as received in `msg.pose.orientation`) into a 3×3
rotation matrix, using the standard closed-form quaternion→matrix formula.
Used to express the marker's measured orientation as a matrix so the
horizontal stand-off offset (defined in the marker's own local frame) can be
rotated into the camera frame.

### `compute_stand_off_goal(marker_pos, R_marker) -> np.ndarray`
- **`marker_pos`**: the marker's raw `[x, y, z]` position, in the camera
  frame, as reported by the detector.
- **`R_marker`**: the marker's orientation as a rotation matrix (output of
  `quaternion_to_rotation_matrix`).
- **Returns**: the 3D point the drone should actually try to hold at.

Computed as:
```python
local_offset_horizontal = np.array([0.0, -OFFSET_TOWARD_M, 0.0])
goal = marker_pos + R_marker @ local_offset_horizontal
goal[1] -= OFFSET_NORMAL_M
return goal
```

**Why the vertical offset is applied separately (world frame), not rotated:**
Routing `OFFSET_NORMAL_M` through `R_marker` (as an earlier version of this
function did, via `local_offset = [0, -TOWARD, NORMAL]` rotated as one
vector) couples the altitude goal to the marker's *apparent* tilt. That
apparent tilt grows with solvePnP noise at range and with
viewing-angle/perspective changes as the drone's distance to the marker
changes — so the altitude target would drift as a function of horizontal
distance even when the marker hasn't physically moved. Applying
`OFFSET_NORMAL_M` after rotation, directly on world-frame Y, makes the
altitude goal depend only on `marker_pos.y` and a fixed offset, regardless
of range or tilt noise.

**Sign convention:** this pose frame is **Y-down**, consistent with the
existing control law `linear.z = -pid_y.compute(err_y, dt)` (see
`_run_altitude_calibration` / the Z-AXIS block in `_run_axis_test`). Under
Y-down, "above the marker" is a **more negative** Y value, which is why the
offset is *subtracted* (`goal[1] -= OFFSET_NORMAL_M`), not added. Getting
this sign wrong manifests as the drone settling far too low and losing the
marker from the camera's FOV — if you ever see that symptom after touching
this function, check this sign first.

---

## 4. `UKF_CTRV` class — the state estimator

Implements an Unscented Kalman Filter with a **Constant Turn Rate and
Velocity (CTRV)** motion model. This models a target moving in a horizontal
plane at some speed `v`, with heading `theta` that can rotate at rate
`omega` — a natural model for the "target point" (the stand-off goal) the
drone is trying to hold at.

**State vector `self.x`** (length 6):

| Index | Name | Meaning |
|---|---|---|
| 0 | `pos_x` | Lateral position (camera-frame X) |
| 1 | `pos_y` | Vertical position / altitude (camera-frame Y, Y-down) |
| 2 | `pos_z` | Depth position (camera-frame Z, "forward") |
| 3 | `v` | Speed in the horizontal (X-Z) plane |
| 4 | `theta` | Heading angle in the horizontal (X-Z) plane |
| 5 | `omega` | Turn rate (rad/s) |

**Key attributes:**
- `self.P` — state covariance matrix (6×6), representing uncertainty in the
  state estimate. Initialized to `I * 1.0`.
- `self.Q` — process noise covariance:
  `diag([0.05, 0.05, 0.05, 0.1, 0.05, 0.05])` — how much the true state is
  expected to drift between predictions, independent of measurement.
- `self.R` — measurement noise covariance: `diag([0.002, 0.002, 0.002])`.
  This matches `ukf_navigation.py`'s value (previously `0.02`, i.e. this
  bench used to trust each vision measurement 10x less than the real
  controller does). At `0.002` the filter tracks fresh measurements
  quickly — more responsive, but more sensitive to per-frame vision noise —
  matching the real controller's dynamics so this bench tunes `pid_y`
  against the same estimator behavior that flies on the drone.
- `self.alpha`, `self.kappa`, `self.beta`, `self.lambda_` — the van der Merwe
  sigma-point spread parameters controlling how sigma points are spaced
  around the mean during the unscented transform.
- `self.weights_m`, `self.weights_c` — weights applied to each sigma point
  when reconstructing the predicted mean and covariance.

**Methods:**

- **`_generate_sigma_points()`**: builds `2n+1 = 13` sigma points around the
  current state estimate `self.x`, spread according to the Cholesky
  decomposition of the (scaled) covariance `self.P`. Falls back to a
  regularized version of `P` if the Cholesky decomposition fails (i.e. `P`
  isn't positive-definite due to numerical drift).

- **`_apply_ctrv(sigma_pts)`**: propagates every sigma point forward by `dt`
  using the CTRV motion equations. Heading `theta` is measured in the
  camera's X-Z plane (0 pointing along +Z / "forward"), matching the
  convention used elsewhere (e.g. `atan2(marker_err_x, marker_err_z)` for
  yaw). When turn rate `omega` is near zero, falls back to a straight-line
  (constant velocity) approximation to avoid dividing by zero. Altitude
  (`pos_y`) is explicitly **not** part of this rotational plane — it's
  carried through unchanged, since climbing/descending has nothing to do
  with horizontal heading dynamics.

- **`reinitialise(position)`**: hard-resets the filter to a known position —
  `x[:3] = position`, zeroes velocity/heading/turn-rate, and tightens `P` to
  `I * 0.05`. Called on target re-acquisition (see `pose_callback` below) so
  that drift accumulated while the target was lost doesn't cause a slow
  "creep back" or a position snap once fresh measurements resume; the
  filter instead starts clean from the first fresh measurement. Matches
  `ukf_navigation.py`'s re-acquisition behavior.

- **`predict(dt)`**: the UKF time-update step. Generates sigma points,
  propagates them through `_apply_ctrv`, recombines them (weighted by
  `weights_m`/`weights_c`) into a new mean `self.x` and covariance `self.P`,
  and adds process noise `self.Q`.

- **`update(z)`**: the UKF measurement-update step. `z` is a 3-element
  `[x, y, z]` position measurement (the stand-off goal point computed
  upstream). Projects the predicted sigma points into measurement space
  (the first 3 state components — position), computes the innovation
  covariance `S` and cross-covariance `Tc`, derives the Kalman gain `K`, and
  corrects `self.x` and `self.P` accordingly. Enforces symmetry on `P`
  afterward to guard against numerical asymmetry accumulating over time.

---

## 5. `PIDController` class

A discrete PID controller with **sign-change-reset anti-windup** and an
optional output deadband compensation.

**Constructor arguments:**
- `kp`, `ki`, `kd` — proportional/integral/derivative gains.
- `max_out` — symmetric output clamp (command saturates at ±`max_out`).
- `min_effective_out` — smallest `|output|` that will actually produce
  motion on the real drone (motor/RC deadband). Left at `0.0` disables this
  compensation; should be measured empirically per axis.

**Internal state:**
- `self.integral` — running sum of `error * dt`.
- `self.last_error` — previous cycle's error, used for the derivative term.

**Methods:**
- **`reset()`**: zeroes `integral` and `last_error`. Called whenever
  tracking is lost/re-acquired (to avoid a stale integral term causing a
  lurch), and also called every cycle on axes that are *not* currently
  active in `_run_axis_test()` so their internal state doesn't drift while
  unused.

- **`compute(error, dt)`**: the PID update —
  1. **Sign-change anti-windup**: if `error` has flipped sign relative to
     `last_error` (and `last_error` isn't zero), zero the integral before
     accumulating this cycle's contribution. This replaces an earlier
     saturation-clamp-based anti-windup (undo the integral step whenever
     clamping changed the output) — sign-change reset was found to behave
     better in practice and is now the only anti-windup mechanism; **there
     is no clamp-based integral undo anymore**.
  2. Accumulate the integral (`integral += error * dt`).
  3. Compute the derivative from the change in error since last call.
  4. Compute the raw (unclamped) output as
     `kp*error + ki*integral + kd*derivative`.
  5. Clamp the output to `±max_out`.
  6. **Deadband compensation**: if the (clamped) output is nonzero but
     smaller in magnitude than `min_effective_out`, bump it up to
     `min_effective_out` with the correct sign, so small commands aren't
     silently absorbed by motor deadband.

---

## 6. `SingleAxisTestController` class — the ROS 2 node

### `__init__(self)`
Delegates to four setup helpers, then triggers takeoff:
- **`_init_pubs_subs()`**: subscribes to `/tello/marker_pose`
  (`pose_callback`); publishes `/cmd_vel` (velocity commands),
  `/tello/takeoff` (triggers takeoff once at startup), and `/tello/land`
  (published by the tracking-loss FSM on `ABORTED`).
- **`_init_debug_telemetry()`**: sets up two PlotJuggler debug topics:
  - `/tello/nav/velocity_estimate` — the UKF's continuous speed estimate,
    decomposed into X/Y components and converted from m/s to cm/s
    (`self.M_TO_CM = 100.0`) for direct comparison against the Tello's
    integer-quantized `/tello/velocity` telemetry.
  - `/tello/nav/yaw_rate_estimate` — the UKF's turn-rate estimate,
    converted from rad/s to deg/s (`self.RAD_TO_DEG`) for comparison
    against `/tello/attitude` (noting the Tello SDK has no raw yaw-rate
    telemetry field to compare against directly).
- **`_init_estimator_and_control()`**: creates `self.ukf` (one `UKF_CTRV`
  instance shared across all axes) and the four `PIDController` instances —
  current gains:

  | PID | kp | ki | kd | max_out | min_effective_out |
  |---|---|---|---|---|---|
  | `pid_x` (lateral/strafe) | 0.0490 | 0.0048 | 0.0146 | 0.90 | 0.10 |
  | `pid_y` (altitude) | 0.0450 | 0.0041 | 0.0146 | 0.90 | 0.15 |
  | `pid_z` (forward/depth) | 0.0490 | 0.0048 | 0.0146 | 0.90 | 0.10 |
  | `pid_yaw` (rotation) | 0.2547 | 0.0283 | 0.0669 | 0.50 | 0.18 |

  plus `self._all_pids`, a list of all four used for bulk reset operations.
  Also starts the 20 Hz (`0.05`s period) control timer.
- **`_init_state()`**: initializes tracking/state variables:
  - `self.latest_goal` — most recent UKF-filtered stand-off position
    estimate, `None` until the first measurement arrives.
  - `self.latest_marker_pos` — most recent *raw, unfiltered* marker
    position (camera frame), `None` until the first measurement arrives.
    Captured separately from `latest_goal` so yaw can bear toward the
    marker itself rather than the offset stand-off point.
  - `self.last_measurement_time`, `self.is_tracking_lost` (starts `True` so
    the very first measurement is treated as a fresh acquisition).
  - `self._recovery_state` — `'TRACKING' | 'HOLD' | 'SEARCHING' | 'ABORTED'`
    (see `_handle_tracking_loss`).
  - `self._flight_state` — `'ALTITUDE_CALIBRATE' | 'AXIS_TEST'` (see §1).
  - `self._altitude_settle_start_time` — debounce timer for altitude
    settling.
  - `self.current_zone` (left/center/right zone parsed from the detector's
    `frame_id`, currently informational only), `self._locked_marker_id`
    (the marker ID this run has locked onto — later detections of other IDs
    are ignored).

### `trigger_takeoff(self)`
Logs, sleeps 1.2s (to give the node time to fully initialize before
flight), then publishes an `Empty` message to `/tello/takeoff`.

### `pose_callback(self, msg: PoseStamped)`
Runs every time a new marker pose arrives. In order:
1. **Marker ID lock**: parses `incoming_id` from `msg.header.frame_id`
   (format `"<id>:<zone>"`). Locks onto the first ID seen; ignores all
   subsequent detections of any other ID for the rest of the run.
2. **Zone parsing**: extracts the left/center/right zone string (currently
   informational only).
3. **`dt` computation**: time since the last pose callback, in seconds.
   Falls back to `0.05` if `dt` comes out non-positive (e.g. clock or
   simulation-time weirdness). Also updates `self.last_measurement_time`,
   used by the control loop's staleness check.
4. **Stand-off goal computation**: builds `marker_pos` from the raw pose,
   converts `msg.pose.orientation` to a rotation matrix, and calls
   `compute_stand_off_goal` to get `z_measured` — the actual target point
   fed into the UKF.
5. **Raw marker capture**: stores `marker_pos` (unmodified, before any
   stand-off offset) in `self.latest_marker_pos` — this is the value yaw
   bearing is computed from in `control_loop()`.
6. **Re-acquisition handling**: if `self.is_tracking_lost` was `True`
   (either the very first callback, or after a gap that tripped `HOLD`,
   `SEARCHING`, or `ABORTED`), calls `self.ukf.reinitialise(z_measured)` to
   hard-snap the filter to the fresh measurement (rather than blending it in
   through the normal Kalman update from a possibly-drifted prior state),
   then resets all four PIDs so stale integral terms don't cause a lurch.
7. **UKF update**: calls `predict(dt)` then `update(z_measured)`, and stores
   the resulting position estimate in `self.latest_goal`.
8. **Debug publishing**: publishes the velocity and yaw-rate estimate topics
   described above.

### `_publish_velocity_estimate(self, stamp)` / `_publish_yaw_rate_estimate(self, stamp)`
Publish the UKF's `v`/`theta` and `omega` states (converted to cm/s and
deg/s respectively) on the PlotJuggler debug topics described in
`_init_debug_telemetry`.

### `_handle_tracking_loss(self, time_since_last_meas) -> bool`
Tiered response to marker dropout, driven purely by elapsed time since the
last successful detection — no re-detection logic is involved. Returns
`True` if `control_loop()` should stop here this tick (i.e. we're in a
recovery state and the normal axis-select/altitude-calibrate logic must not
run this cycle).

| Elapsed since last detection | State | Behavior |
|---|---|---|
| `< LOSS_HOLD_SEC` (2.0s) | `TRACKING` | Normal — returns `False`. |
| `LOSS_HOLD_SEC` – `LOSS_SEARCH_SEC` | `HOLD` | Publish zero `Twist` (hover). PID state left untouched (not treated as a full loss — `self.is_tracking_lost` **is** set `True` here too, so the next fresh measurement still reinitializes the UKF, but `_flight_state` is left alone). |
| `LOSS_SEARCH_SEC` – `LOSS_ABORT_SEC` (5.0–20.0s) | `SEARCHING` | Publish `Twist(angular.z = SEARCH_YAW_RATE)` — slow scan rotation. Also sets `_flight_state = 'ALTITUDE_CALIBRATE'`, so recovery re-does altitude calibration. |
| `> LOSS_ABORT_SEC` (20.0s) | `ABORTED` | Zero `Twist`, publish `/tello/land`, and set `_flight_state = 'ALTITUDE_CALIBRATE'`. |

Each state transition is logged once (guarded by
`self._recovery_state != <state>`) rather than spamming every tick.

### `control_loop(self)`
Runs at 20 Hz via the timer:
1. If no measurement has ever arrived (`self.latest_goal is None or
   self.latest_marker_pos is None`), publish a zero `Twist`, reset all
   PIDs, and return.
2. Compute `time_since_last_meas` and call `_handle_tracking_loss`; if it
   returns `True`, return immediately (a recovery state is handling output
   this tick).
3. Unpack two separate error sets:
   - `err_x`, `err_y`, `err_z` from `self.latest_goal` — the UKF-filtered
     stand-off goal errors, used to drive **translation**.
   - `marker_err_x`, `marker_err_z` from `self.latest_marker_pos` — the raw,
     unfiltered marker errors, used to drive **yaw bearing** only.
4. If `self._flight_state == 'ALTITUDE_CALIBRATE'`, call
   `_run_altitude_calibration(err_y, dt, now)` and return — altitude is
   handled exclusively, no axis-test block runs during calibration.
5. Otherwise, call `_run_axis_test(err_x, err_y, err_z, marker_err_x,
   marker_err_z, dt)`.

Note: `dt` here is hardcoded to `0.05` (matching the 20 Hz timer period),
distinct from the measured `dt` used in `pose_callback`.

### `_run_altitude_calibration(self, err_y, dt, now)`
Runs once, alone, as a calibration step: `linear.z =
-self.pid_y.compute(err_y, dt)`. Once `|err_y|` stays under
`ALTITUDE_SETTLE_TOL_M` for `ALTITUDE_SETTLE_HOLD_SEC` continuously (a
single excursion outside tolerance resets the debounce timer), logs and
switches `_flight_state` to `AXIS_TEST` permanently — altitude is then left
to the Tello's own onboard hover/barometer hold and is not re-driven again
unless a tracking-loss recovery (`SEARCHING`/`ABORTED`) forces
`_flight_state` back to `ALTITUDE_CALIBRATE`.

### `_run_axis_test(self, err_x, err_y, err_z, marker_err_x, marker_err_z, dt)`
Exactly **one** block below should be uncommented at a time. All other axes
stay at `0.0` — altitude is intentionally **not** re-driven here (it was
calibrated once in `ALTITUDE_CALIBRATE`).

- **X AXIS** *(currently active)*: `twist.linear.x = self.pid_z.compute(err_z, dt)`;
  resets `pid_yaw`, `pid_y`, `pid_z`.

  > ⚠️ **Known issue**: `pid_z.reset()` is called immediately after
  > `pid_z.compute(...)` in this same block, every control-loop tick. That
  > zeroes `pid_z`'s integral and `last_error` right after they're used, so
  > `pid_z`'s integral term never actually accumulates across cycles — it
  > runs effectively as a P+D-only controller regardless of its tuned `ki`,
  > and the derivative term sees `(error - 0)` each cycle instead of a true
  > frame-to-frame delta. If forward/back tuning looks off despite `ki`
  > being nonzero, this is why. Fix (not yet applied): drop `pid_z.reset()`
  > from this block — only the *inactive* axes' PIDs need resetting here.

- **Y AXIS** *(commented out)*: pure strafe test —
  `twist.linear.y = self.pid_x.compute(err_x, dt)` only.
- **Z AXIS** *(commented out)*: pure altitude test —
  `twist.linear.z = self.pid_y.compute(err_y, dt)` only (**not** negated in
  this block, unlike `_run_altitude_calibration`, which does negate it — see
  sign convention note in §3 if re-enabling this block for standalone
  altitude testing).
- **YAW AXIS** *(commented out)*: pure yaw test —
  `yaw_error = atan2(marker_err_x, marker_err_z)`,
  `twist.angular.z = self.pid_yaw.compute(yaw_error, dt)`.

### `stop_drone(self)`
Publishes a zero-velocity `Twist`.

### `main(args=None)`
Standard ROS 2 entry point: initializes `rclpy`, spins the node, and on
shutdown (including `KeyboardInterrupt`) calls `stop_drone()` before
destroying the node.

---

## 7. Summary of recent changes (for anyone diffing against an older copy)

- `self.R` (UKF measurement noise) tightened from `diag([0.02]*3)` to
  `diag([0.002]*3)`, matching `ukf_navigation.py` — the bench's estimator
  now behaves like the real controller's instead of being 10x more
  conservative about trusting fresh measurements.
- Added `UKF_CTRV.reinitialise()` and wired it into `pose_callback`'s
  re-acquisition path (previously only the PIDs were reset on re-lock; the
  UKF kept blending in from a potentially drifted state).
- `compute_stand_off_goal` no longer routes `OFFSET_NORMAL_M` through
  `R_marker` — it's applied as a fixed world-frame Y offset after rotation,
  so the altitude target no longer drifts with marker tilt/range noise.
- Fixed a sign error introduced during the above change: the offset is
  **subtracted** from world-frame Y (`goal[1] -= OFFSET_NORMAL_M`), not
  added, per the Y-down convention implied by the existing
  `-pid_y.compute(...)` control law. The wrong sign caused the drone to
  settle far too low and lose the marker from view.
- `OFFSET_NORMAL_M` retuned from `0.45` to `0.35`.# `single_axis.py` — Documentation

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
4. **Feed** the filtered stand-off goal error into whichever axis's PID
   block is currently active in `control_loop()` for translation, but drive
   yaw off the *raw, unfiltered marker position* instead — so the drone
   points at the marker itself rather than at the offset stand-off point
   (which shifts off the marker's bearing whenever the marker is rotated
   relative to the camera). Publish the resulting velocity command on
   `/cmd_vel`.
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
  
- **Tracking/state variables**: `self.latest_goal` (most recent UKF-filtered
  stand-off position estimate, `None` until the first measurement arrives),
  `self.latest_marker_pos` (most recent *raw, unfiltered* marker position in
  the camera frame, `None` until the first measurement arrives — captured
  separately from `latest_goal` so yaw can bear toward the marker itself
  instead of toward the offset stand-off point), `self.is_tracking_lost`
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
6. **Raw marker capture**: stores `marker_pos` (before the stand-off offset
   is applied) in `self.latest_marker_pos`. This is the value yaw bearing
   will be computed from in `control_loop()` — kept separate from the
   UKF-filtered `latest_goal` so it isn't shifted by the stand-off offset.
7. **UKF update**: calls `predict(dt)` then `update(z_measured)`, and stores
   the resulting position estimate in `self.latest_goal`.
8. **Debug publishing**: publishes the velocity and yaw-rate estimate topics
   described above.

### `control_loop(self)`
Runs at 20 Hz via the timer:
1. If no measurement has arrived yet (`self.latest_goal is None or
   self.latest_marker_pos is None`), publishes a zero `Twist`, resets all
   PIDs, and returns.
   
2. **Staleness/safety check**: tiered tracking-loss handling based on time
   since the last measurement (`HOLD` → hover, `SEARCHING` → slow scan yaw,
   `ABORTED` → stop and land); see `_handle_tracking_loss`.
3. Unpacks two separate error sets:
   - `err_x`, `err_y`, `err_z` from `self.latest_goal` — the UKF-filtered
     stand-off goal errors, used to drive **translation**.
   - `marker_err_x`, `marker_err_z` from `self.latest_marker_pos` — the raw,
     unfiltered marker errors, used to drive **yaw bearing** only, so the
     drone points at the marker itself rather than at the offset stand-off
     point.
4. **Active axis block**: exactly one of the four commented/uncommented
   blocks below should be live at a time:
   - **X AXIS** : drives `pid_z` on `err_z` for
     forward/back motion, and `pid_yaw` on
     `atan2(marker_err_x, marker_err_z)` for yaw bearing toward the marker.
   - **Y AXIS** : pure strafe test — `pid_x` on `err_x` only.
   - **Z AXIS** : pure altitude test — `pid_y` on `err_y`
     only, with the output negated (`-self.pid_y.compute(...)`).
   - **YAW AXIS** : pure yaw test — drives `pid_yaw` on
     `atan2(marker_err_x, marker_err_z)`, the raw marker bearing.
5. Publishes the resulting `twist` on `/cmd_vel`.

### `stop_drone(self)`
Publishes a zero-velocity `Twist` — used both on the safety timeout and on
node shutdown.

### `main(args=None)`
Standard ROS 2 entry point: initializes rclpy, spins the node, and on
shutdown (including `KeyboardInterrupt`) calls `stop_drone()` before
destroying the node.

---# `single_axis.py` — Documentation

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



### PID_values:
the pid values for conservative or fast control:
1. conservative
- self.pid_x    = PIDController(kp=0.0490, ki=0.0048, kd=0.0146, max_out=0.90,min_effective_out=0.15)
- self.pid_y    = PIDController(kp=0.0490, ki=0.0048, kd=0.0146, max_out=0.90, min_effective_out=0.15)
- self.pid_z    = PIDController(kp=0.0450, ki=0.0041, kd=0.0146, max_out=0.90, min_effective_out=0.15)
- self.pid_yaw  = PIDController(kp=0.2547, ki=0.0283, kd=0.0669, max_out=0.90, min_effective_out=0.18)
2. fast 
- self.pid_x    = PIDController(kp=0.0735, ki=0.01081, kd=0.0219, max_out=0.90, - min_effective_out=0.15)
- self.pid_y    = PIDController(kp=0.0735, ki=0.01081, kd=0.0219, max_out=0.90, min_effective_out=0.15)
- self.pid_z    = PIDController(kp=0.0676, ki=0.00913, kd=0.0219, max_out=0.90, min_effective_out=0.15)
- self.pid_yaw  = PIDController(kp=0.382,  ki=0.0637,  kd=0.100,  max_out=0.90, min_effective_out=0.18)

---

