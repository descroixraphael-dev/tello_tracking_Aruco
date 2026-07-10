import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist, Vector3Stamped
from std_msgs.msg import Empty, Int32
from cv_bridge import CvBridge
from djitellopy import Tello
import cv2

# ---------------------------------------------------------------------------
# Named constants (avoid magic numbers scattered through the callbacks)
# ---------------------------------------------------------------------------
VIDEO_RATE_HZ = 30.0
PERIOD_STATE=0.05
RC_SCALE_FACTOR = 100          # Tello send_rc_control expects [-100, 100]
BATTERY_CRITICAL_PCT = 10
STATE_FRAME_ID = 'tello'       # frame_id used on stamped telemetry topics


class TelloDriverNode(Node):
    def __init__(self):
        super().__init__('tello_driver_node')
        self.tello = Tello()
        self.tello.connect()
        self.tello.streamon()

        # Grab the background frame reader ONCE. Calling get_frame_read()
        # repeatedly (e.g. inside a timer callback) spins up a new
        # BackgroundFrameRead thread every call instead of reusing the
        # existing one -- that was silently leaking threads at 30Hz.
        self.frame_reader = self.tello.get_frame_read()

        self.bridge = CvBridge()
        self.image_pub = self.create_publisher(Image, '/tello/image_raw', 10)
        self.battery_pub = self.create_publisher(Int32, '/tello/battery', 10)
        self.altitude_pub = self.create_publisher(Int32, '/tello/altitude', 10)

        # Stamped Vector3 topics so PlotJuggler can plot .vector.x/y/z as
        # separate curves with real timestamps, without needing a custom msg.
        # velocity: x=forward/back speed, y=left/right speed, z=up/down speed (cm/s, body frame per Tello SDK)
        self.velocity_pub = self.create_publisher(Vector3Stamped, '/tello/velocity', 10)
        # attitude: x=roll, y=pitch, z=yaw (degrees)
        self.attitude_pub = self.create_publisher(Vector3Stamped, '/tello/attitude', 10)

        # Command Subscribers (Inputs)
        self.create_subscription(Twist, '/cmd_vel', self.vel_callback, 10)
        self.create_subscription(Empty, '/tello/takeoff', self.takeoff_callback, 10)
        self.create_subscription(Empty, '/tello/land', self.land_callback, 10)

        # Timers: 30Hz for Video, 5Hz to retrieve and read flight data
        self.create_timer(1.0 / VIDEO_RATE_HZ, self.publish_video)
        self.create_timer(PERIOD_STATE, self.retrieve_flight_data)

        self.get_logger().info("Tello Driver Ready (Data Retrieval active)")

    def vel_callback(self, msg):
        left_right_speed = int(msg.linear.y * RC_SCALE_FACTOR)
        forward_back_speed = int(msg.linear.x * RC_SCALE_FACTOR)
        up_down_speed = int(msg.linear.z * RC_SCALE_FACTOR)
        yaw_speed = int(msg.angular.z * RC_SCALE_FACTOR)
        self.tello.send_rc_control(left_right_speed, forward_back_speed, up_down_speed, yaw_speed)

    def takeoff_callback(self, _):
        self.tello.takeoff()

    def land_callback(self, _):
        self.tello.land()

    def publish_video(self):
        frame = self.frame_reader.frame
        if frame is not None:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))

    def retrieve_flight_data(self):
        # 1. Pull the data packet from the Tello Wi-Fi client
        state_dict = self.tello.get_current_state()

        if state_dict:
            # 2. Extract any value you want directly into normal Python variables
            battery_pct = int(state_dict.get('bat', 0))
            altitude_cm = int(state_dict.get('h', 0))

            pitch_deg = float(state_dict.get('pitch', 0))
            roll_deg = float(state_dict.get('roll', 0))
            yaw_deg = float(state_dict.get('yaw', 0))

            # Velocities (Extracted as pure Python floats, cm/s, body frame)
            forward_back_velocity = float(state_dict.get('vgx', 0.0))
            left_right_velocity = float(state_dict.get('vgy', 0.0))
            up_down_velocity = float(state_dict.get('vgz', 0.0))

            # 3. Publish everything so it shows up live in PlotJuggler
            stamp = self.get_clock().now().to_msg()

            self.battery_pub.publish(Int32(data=battery_pct))
            self.altitude_pub.publish(Int32(data=altitude_cm))

            velocity_msg = Vector3Stamped()
            velocity_msg.header.stamp = stamp
            velocity_msg.header.frame_id = STATE_FRAME_ID
            velocity_msg.vector.x = forward_back_velocity
            velocity_msg.vector.y = left_right_velocity
            velocity_msg.vector.z = up_down_velocity
            self.velocity_pub.publish(velocity_msg)

            attitude_msg = Vector3Stamped()
            attitude_msg.header.stamp = stamp
            attitude_msg.header.frame_id = STATE_FRAME_ID
            attitude_msg.vector.x = roll_deg
            attitude_msg.vector.y = pitch_deg
            attitude_msg.vector.z = yaw_deg
            self.attitude_pub.publish(attitude_msg)

            # Example logic: Emergency land if battery is too low
            if battery_pct < BATTERY_CRITICAL_PCT:
                self.get_logger().warn("Battery critically low! Landing drone...")
                self.tello.land()


def main():
    rclpy.init()
    node = TelloDriverNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
