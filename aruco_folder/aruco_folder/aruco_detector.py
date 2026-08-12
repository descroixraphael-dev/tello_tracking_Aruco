import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
from rclpy.qos import qos_profile_sensor_data

# ─────────────────────────────────────────────────────────────────────────────
#  This node now reports the marker's own pose (position + orientation) only.
#  The "stand 1m behind / 45cm above the marker, however it's rotated" offset
#  logic lives downstream in orbit_nav.py, since that's where the marker's
#  orientation is combined with the desired stand-off to build the goal point.
# ─────────────────────────────────────────────────────────────────────────────

def rotation_matrix_to_quaternion(R):
    """
    Convert a 3x3 rotation matrix to a quaternion [qx, qy, qz, qw].

    Uses Shepperd's method (branches on the largest diagonal term) for
    numerical stability, rather than assuming a small-angle Euler
    decomposition. NOTE: rvec from cv2.solvePnP is an axis-angle (Rodrigues)
    vector, NOT a set of Euler angles -- feeding it straight into an
    Euler->quaternion formula (as the old get_quaternion_from_euler did)
    silently produces a wrong orientation for anything but tiny rotations.
    """
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return [qx, qy, qz, qw]

class ArUcoDetectorNode(Node):
    def __init__(self):
        super().__init__('aruco_detector_node')
        self.bridge = CvBridge()
        
        self.declare_parameter('target_marker_id', 2)
        self.target_marker_id = self.get_parameter('target_marker_id').get_parameter_value().integer_value
        self.get_logger().info(f"Detector filtering active. Only processing Marker ID: {self.target_marker_id}")
        
        # Matches the publisher's sensor-data QoS: best-effort, depth 1, so a
        # slower detection loop drops old frames instead of processing a
        # growing backlog (which is what made the imshow feed feel laggy).
        self.sub = self.create_subscription(Image, '/tello/image_raw', self.img_cb, qos_profile_sensor_data)
        self.pose_pub = self.create_publisher(PoseStamped, '/tello/marker_pose', 10)

        # Camera Matrix (Standard Tello 720p optics)
        self.k = np.array([[730, 0, 640], [0, 730, 360], [0, 0, 1]], dtype=float)
        self.d = np.zeros(5)
        self.marker_size = 0.15   

        self.img_w = 1280
        self.img_h = 720

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.detector  = cv2.aruco.ArucoDetector(self.aruco_dict)

    def _column_zone(self, cx):
        third = self.img_w / 3.0
        if cx < third:
            return 'LEFT'
        elif cx < 2 * third:
            return 'CENTER'
        else:
            return 'RIGHT'

    def _draw_hud(self, frame, corners, zone):
        h, w = frame.shape[:2]
        t3, t23 = w // 3, 2 * w // 3

        cv2.line(frame, (t3,  0), (t3,  h), (200, 200, 200), 1)
        cv2.line(frame, (t23, 0), (t23, h), (200, 200, 200), 1)

        pts = corners.reshape(4, 2).astype(int)
        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
        cx, cy = pts.mean(axis=0).astype(int)
        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

        color = (0, 255, 0) if zone == 'CENTER' else (0, 100, 255)
        cv2.putText(frame, zone, (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        # Note: the virtual stand-off goal is no longer computed here -- it's
        # derived downstream in orbit_nav.py from this raw marker pose. If you
        # want it visualized on this HUD too, say so and I'll add it back as a
        # display-only computation (not published).

    def img_cb(self, msg):
        frame     = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.img_h, self.img_w = rgb_frame.shape[:2]

        corners, ids, _ = self.detector.detectMarkers(rgb_frame)

        if ids is not None:
            obj_points = np.array([
                [-self.marker_size/2,  self.marker_size/2, 0],
                [ self.marker_size/2,  self.marker_size/2, 0],
                [ self.marker_size/2, -self.marker_size/2, 0],
                [-self.marker_size/2, -self.marker_size/2, 0],
            ], dtype=np.float32)

            for i in range(len(ids)):
                incoming_id = int(ids[i][0])
                
                # FIXED: Respect code structure while discarding any unrequested/false-positive IDs immediately
                if incoming_id != self.target_marker_id:
                    continue

                _, rvec, tvec = cv2.solvePnP(obj_points, corners[i], self.k, self.d)
                R_marker, _   = cv2.Rodrigues(rvec)
                marker_pos    = tvec.flatten()
                q             = rotation_matrix_to_quaternion(R_marker)

                cx_px = corners[i].reshape(4, 2).mean(axis=0)[0]
                zone  = self._column_zone(cx_px)

                pose = PoseStamped()
                pose.header.stamp    = self.get_clock().now().to_msg()
                pose.header.frame_id = f"{incoming_id}:{zone}"

                # Raw marker pose (camera frame) -- downstream nav node is
                # responsible for turning this into a stand-off goal point.
                pose.pose.position.x = float(marker_pos[0])
                pose.pose.position.y = float(marker_pos[1])
                pose.pose.position.z = float(marker_pos[2])

                pose.pose.orientation.x = q[0]
                pose.pose.orientation.y = q[1]
                pose.pose.orientation.z = q[2]
                pose.pose.orientation.w = q[3]

                self.pose_pub.publish(pose)

                # ── Logging ──────────────────────────────────────────────────
                marker_dist = float(np.linalg.norm(marker_pos))
                mx, my, mz  = marker_pos
                self.get_logger().info(
                    f"[ID {incoming_id}] zone={zone} | "
                    f"marker_dist={marker_dist:.3f}m | "
                    f"marker_pos=({mx:+.3f}, {my:+.3f}, {mz:+.3f}) m"
                )
                # ─────────────────────────────────────────────────────────────

                cv2.drawFrameAxes(rgb_frame, self.k, self.d, rvec, tvec, 0.05)
                self._draw_hud(rgb_frame, corners[i], zone)

        cv2.imshow("Detection", rgb_frame)
        cv2.waitKey(1)

def main():
    rclpy.init()
    node = ArUcoDetectorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
