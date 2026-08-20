## 1. General Overview

This node is the vision front-end of the pipeline: it takes raw camera
frames from the Tello, detects a specific ArUco marker in each frame, solves
for that marker's 3D pose (position + orientation) relative to the camera,
and publishes it as a PoseStamped message.

It deliberately does not compute any stand-off/goal-point offset — that
logic lives downstream in the navigation nodes (ukf_navigation.py, single_axis.py), since those are the places where the
marker's orientation gets combined with the desired stand-off distance. This
node's only job is: find the marker, report exactly where and how it's
oriented, nothing more.

Per-frame pipeline in **img_cb**:


1. Convert the incoming ROS Image to an OpenCV BGR frame, then to RGB.
2. Run ArUco detection to get marker corners + IDs.
3. Filter for only the one target_marker_id this node cares about.
4. Use solvePnP to recover the marker's pose (rvec/tvec) relative to
the camera, given the camera's intrinsics and known marker size.
5.Convert the rotation vector to a rotation matrix, then to a quaternion
(for publishing in a standard ROS Pose orientation field).
6. Publish position + orientation as a PoseStamped, with the marker ID and
on-screen "zone" (LEFT/CENTER/RIGHT) encoded in frame_id.
7. Draw debug overlays (marker outline, pose axes, zone HUD) and display the
frame locally with cv2.imshow.


## 2. Module-Level Function

**rotation_matrix_to_quaternion(R)** -> [qx, qy, qz, qw]

Converts a 3×3 rotation matrix into a quaternion using Shepperd's method
— it branches on which diagonal term of R is largest to pick the
numerically stable formula (avoids division-by-near-zero problems that a
single fixed formula would hit for certain rotations).


##3. ArUcoDetectorNode class

### __init__(self)


- Declares a ROS parameter **target_marker_id** (default 2) — only markers
with this ID are processed; everything else is ignored.

- Subscribes to **/tello/image_raw**, publishes **PoseStamped** on
**/tello/marker_pose**.

- Camera intrinsics (self.k, self.d): a hard-coded camera matrix
calibrated for "standard Tello 720p optics" (focal length ≈730px, principal
point at (640, 360)), with zero distortion coefficients (self.d) — i.e.
the model assumes the Tello's lens distortion is negligible or
pre-corrected elsewhere.

- self.marker_size = 0.15 — the physical side length of the ArUco marker,
in meters. This must match the real printed marker exactly, since
solvePnP's scale comes entirely from this value.

- self.img_w, self.img_h — image dimensions, initialized to 1280×720 but
overwritten every frame from the actual incoming image shape.
self.aruco_dict / self.detector — OpenCV's ArUco detector, configured
for the **DICT_4X4_50** marker dictionary.


**_column_zone(self, cx)** -> str

Given a marker's horizontal pixel center cx, splits the image into three
equal vertical thirds and returns 'LEFT', 'CENTER', or 'RIGHT'
depending on which third cx falls in. Used purely for the on-screen HUD
and encoded into the published frame_id — not used in any control math in
this node.

**_draw_hud(self, frame, corners, zone)**

Debug/visualization only:


- Draws two vertical guide lines splitting the frame into thirds.
- Draws the marker's outline (polyline through its 4 corners) and a dot at
its center.
- Labels the zone text next to the marker, colored green if CENTER,
orange otherwise.

**img_cb(self, msg)**

The main per-frame callback:


1. Converts the incoming Image message to an OpenCV BGR frame via
cv_bridge, then to RGB for detection (detectMarkers expects RGB/gray
depending on config; converting explicitly avoids ambiguity).

2. Updates self.img_h, self.img_w from the actual frame shape each call
(so _column_zone's thirds stay correct even if resolution changes).

3. Runs self.detector.detectMarkers(...) to get corners and ids.

4. If any markers were found:

- Defines obj_points — the marker's 4 corners in its own local 3D frame
(a flat square centered at the origin, Z=0), in the order
top-left → top-right → bottom-right → bottom-left. This ordering
matters: it fixes which direction the marker's local X/Y/Z axes point,
and downstream code (in the nav nodes) assumes this specific
convention when reasoning about "the marker's own frame."

- Loops over every detected marker, but skips (continue) any ID that
isn't self.target_marker_id — so even if multiple markers are
visible, only the one being tracked is processed.

- For the matching marker: runs solvePnP to get rvec/tvec, converts
rvec to a rotation matrix via cv2.Rodrigues, flattens tvec into
marker_pos (the raw [x, y, z] position in the camera frame), and
converts the rotation matrix to a quaternion q.

- Computes the marker's on-screen center X pixel and derives its zone.

- Builds and publishes a PoseStamped:
  - frame_id is set to "<id>:<zone>" — this is how downstream nodes
parse out both the locked marker ID and the on-screen zone from a
single string field, without needing a custom message type.
  - Position and orientation fields are filled directly from
marker_pos and q — no offset is applied here, this is the raw
marker pose.



- Logs the marker's straight-line distance (**np.linalg.norm(marker_pos)**)
and raw [x, y, z] for debugging.

- Draws the pose axes (**cv2.drawFrameAxes**) and the HUD overlay on the
frame.

5. Displays the annotated frame in an OpenCV window (cv2.imshow) and calls
cv2.waitKey(1) to let the window refresh (required for imshow to
actually render/update on most platforms).


**main()**

Initializes rclpy, wraps the node in a **MultiThreadedExecutor** (rather than
a single-threaded spin) — likely so image callbacks don't block other
potential callbacks/timers, though this node currently only has the one
subscription. Spins until KeyboardInterrupt, then cleans up.
