import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Empty
import numpy as np
import time

class KalmanFilter:
    def __init__(self, dt):
        self.dt = dt
        # State: [x, y, z, vx, vy, vz]
        self.x = np.zeros(6)
        
        # Transition Matrix (CVM)
        self.F = np.eye(6)
        self.F[0, 3] = self.F[1, 4] = self.F[2, 5] = dt
        
        # Measurement Matrix (x, y, z)
        self.H = np.zeros((3, 6))
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = 1
        
        self.P = np.eye(6) * 1.0
        self.Q = np.eye(6) * 0.01  # Process noise
        self.R = np.eye(3) * 0.001 # Camera confidence (High)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y)
        self.P = (np.eye(6) - (K @ self.H)) @ self.P

class TelloLinearFollower(Node):
    def __init__(self):
        super().__init__('tello_linear_follower')
        self.kf = KalmanFilter(dt=0.1)
        self.target_dist = 0.50  # 50cm Target
        self.last_detection = time.time()
        
        # Publishers
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.takeoff_pub = self.create_publisher(Empty, '/tello/takeoff', 10)
        
        # Subscriber
        self.create_subscription(PoseStamped, '/tello/marker_pose', self.pose_cb, 10)
        
        # Control Loop (10Hz)
        self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info("LINEAR FOLLOW-MODE: Initializing...")
        time.sleep(2.0)
        self.takeoff_pub.publish(Empty())
        self.get_logger().info("ACTION: Takeoff Sent.")

    def pose_cb(self, msg):
        z_measure = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        self.kf.update(z_measure)
        self.last_detection = time.time()

    def control_loop(self):
        # Stop safety if marker is gone
        if time.time() - self.last_detection > 1.0:
            self.vel_pub.publish(Twist())
            return

        self.kf.predict()
        curr_z = self.kf.x[2] # Current estimated distance from marker
        
        # ERROR: How far we are from the 50cm goal
        # +ve = Too far (move forward), -ve = Too close (move back)
        error = curr_z - self.target_dist

        twist = Twist()
        
        # --- LINEAR MOTION ONLY ---
        # Gain: 0.7 (snappy but controlled), Max Speed: 0.15 (safe)
        twist.linear.x = np.clip(error * 3.0, -0.2, 0.2)
        
        # Explicitly zeroing everything else
        twist.linear.y = 0.0
        twist.linear.z = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = 0.0

        # Deadzone: Ignore errors smaller than 3cm to stay steady
        if abs(error) < 0.03:
            twist.linear.x = 0.0
                
        self.vel_pub.publish(twist)
        self.log_status(error, curr_z)

    def log_status(self, err, dist):
        if abs(err) < 0.04:
            status = "POSITION LOCKED"
        elif err > 0:
            status = "MOVING FORWARD (Targeting 0.50m)"
        else:
            status = "MOVING BACKWARD (Targeting 0.50m)"
        
        self.get_logger().info(f"DIST: {dist:.2f}m | STATUS: {status}")

def main():
    rclpy.init()
    node = TelloLinearFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Emergency stop on exit
        node.vel_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()
