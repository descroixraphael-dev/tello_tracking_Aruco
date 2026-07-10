## 1. General Overview

This node is the hardware interface to the physical Tello drone: it's
the only node in the system that actually talks to the drone over Wi-Fi via
djitellopy. It has two responsibilities:


1. Command path (downstream → drone): subscribes to **/cmd_vel**, 
**/tello/takeoff** , **/tello/land**, and translates them into direct
**djitellopy** calls (send_rc_control, takeoff, land).

2. Telemetry path (drone → upstream): on timers, pulls video frames and
flight-state data (battery, altitude, attitude, velocity) from the drone
and republishes them as standard ROS message types so tools like
PlotJuggler and the nav nodes can consume them.


It also includes a basic safety behavior: automatic landing if battery drops
below a critical threshold.

## 2. Module-Level Constants

Name | Value | Meaning
**VIDEO_RATE_HZ**| 30.0 | Rate at which video frames are polled and republished.
**PERIOD_STATE**| 0.05 | Timer period (seconds) for polling flight telemetry — i.e. 20 Hz.
**RC_SCALE_FACTOR**|  100  | Multiplier converting normalized [-1, 1]-ish Twist values into the [-100, 100] integer range send_rc_control expects.
**BATTERY_CRITICAL_PCT**|  10  | Battery percentage threshold that triggers an automatic landing.
**STATE_FRAME_ID**| 'tello' |frame_id string stamped onto the telemetry Vector3Stamped messages.

## 3. TelloDriverNode class

### __init__(self)


- Constructs a **djitellopy.Tello()** instance, connects to it, and starts the
video stream (**streamon()**).

- **self.frame_reader = self.tello.get_frame_read()** — grabbed exactly
once here. The comment explains why this matters: calling
get_frame_read() repeatedly (e.g. from inside a timer callback) spins up
a brand-new background thread every single call instead of reusing the
existing reader — a thread leak that would compound at 30 Hz. Grabbing it
once in __init__ and reusing self.frame_reader.frame in
publish_video avoids this.

- Sets up publishers: /tello/image_raw (video), /tello/battery,
/tello/altitude, /tello/velocity (stamped Vector3, body-frame cm/s),
/tello/attitude (stamped Vector3, degrees — roll/pitch/yaw).

- Sets up subscriptions: /cmd_vel → vel_callback, /tello/takeoff →
takeoff_callback, /tello/land → land_callback.

- Starts two timers: one at **VIDEO_RATE_HZ** (30 Hz) for publish_video, one
at **PERIOD_STATE** (20 Hz) for retrieve_flight_data.


**vel_callback(self, msg)**

Translates an incoming **Twist** command into a single **send_rc_control** call:


- left_right_speed = msg.linear.y * 100
- forward_back_speed = msg.linear.x * 100
- up_down_speed = msg.linear.z * 100
- yaw_speed = msg.angular.z * 100


All four are cast to int (the SDK expects integers in [-100, 100]) and
sent together in one send_rc_control(left_right, forward_back, up_down, yaw) call. This is the point where the nav/test nodes' unitless Twist
values get turned into the actual RC stick values sent to the drone — so any
sign or scale mismatch upstream (e.g. in the nav nodes' PID outputs) shows up
directly here as physical drone motion.

**takeoff_callback(self, _)** / **land_callback(self, _)**

Trivial passthroughs: call self.tello.takeoff() / self.tello.land() on
receipt of an Empty message. The _ parameter name signals the message
content itself is irrelevant — only the event of receiving it matters.

**publish_video(self)**

Reads the current frame from the already-created self.frame_reader
(.frame attribute, updated continuously in the background by djitellopy),
and if it's not None, converts it to a ROS Image message via cv_bridge
and publishes it. Guards against publishing before the stream has produced a
first frame.

**retrieve_flight_data(self)**

Runs at 20 Hz:


1. Pulls the latest telemetry dict from the Tello via
self.tello.get_current_state(). This can be empty/falsy if no packet
has arrived yet — guarded by the if state_dict: check.
2. Extracts and type-casts individual fields out of the raw dict:

- battery_pct ('bat', int), altitude_cm ('h', int)
- pitch_deg, roll_deg, yaw_deg (floats, degrees)
- forward_back_velocity ('vgx'), left_right_velocity ('vgy'),
- up_down_velocity ('vgz') — all floats, cm/s, in the drone's body
frame per the Tello SDK.
- Each uses .get(key, default) so a missing field doesn't raise a
KeyError, just falls back to 0/0.0.



3. Publishes:

- battery_pct and altitude_cm as plain Int32 messages.

- Velocity as a Vector3Stamped (.vector.x/y/z = forward/back,
left/right, up/down) — stamped with a shared stamp and
STATE_FRAME_ID so PlotJuggler can plot all three as time-aligned
curves.

- Attitude as a Vector3Stamped (.vector.x/y/z = roll, pitch, yaw)
using the same stamp/frame convention.



4. Safety check: if battery_pct < BATTERY_CRITICAL_PCT (10%), logs a
warning and immediately calls self.tello.land() — this fires on every
telemetry cycle once battery drops below threshold, so it will attempt to
re-issue land() repeatedly until the drone is actually down (harmless
given djitellopy/the drone itself will just no-op or already be
landing, but worth knowing it's not a one-shot trigger).


**main()**

Initializes rclpy, constructs the node (which immediately connects to the
drone and starts streaming as part of __init__), spins, then destroys the
node and shuts down. Notably, unlike the nav nodes, there's no
try/except around rclpy.spin(node) and no explicit safety-stop call
(like stop_drone()) in a finally block — a KeyboardInterrupt here will
propagate up without an explicit landing command being issued from this
function, relying instead on whatever cleanup djitellopy/rclpy do
internally on teardown.

4. Cross-File Notes Worth Keeping in Mind


- Axis convention hand-off: driver_node.py's vel_callback maps
Twist.linear.x → forward/back and Twist.linear.y → left/right. This is
the Tello's body-frame RC convention, and it's not automatically the
same as the camera-frame X/Y/Z convention used in aruco_detector.py and
the nav nodes (camera X = lateral, Y = altitude, Z = depth/forward). The
nav nodes are responsible for mapping their camera-frame errors onto the
correct Twist fields before publishing — any mismatch there (as seen in
earlier debugging of single_axis.py) will surface here as the drone
moving in an unexpected direction, even though this file itself is a
faithful, mechanical translation of whatever Twist it receives.

- No offset logic here: as its own comment states, aruco_detector.py
intentionally reports only the raw marker pose. Any bugs related to the
1m-behind/45cm-above stand-off point are not in this file — they're in
whichever nav node computes compute_stand_off_goal.
