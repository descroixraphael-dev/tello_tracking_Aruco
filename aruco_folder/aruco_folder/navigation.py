import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Empty
import numpy as np
import time

class KalmanFilter:
    def __init__(self, dt):
        self.dt = dt
        self.x = np.zeros(6)
        self.F = np.eye(6)
        self.F[0, 3] = self.F[1, 4] = self.F[2, 5] = dt
        self.H = np.zeros((3, 6))
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = 1
        self.P = np.eye(6) * 1.0  
        self.R = np.eye(3) * 0.00001   
        self.Q = np.eye(6) * 0.1  

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y)
        self.P = (np.eye(6) - (K @ self.H)) @ self.P

class TelloController(Node):
    def __init__(self):
        super().__init__('tello_controller')
        
        self.target_id = 0      
        self.target_dist = 0.50 
        self.last_log_time = self.get_clock().now()
        
        self.kf = KalmanFilter(dt=0.1)
        self.last_detection_time = time.time()
        self.marker_found = False

        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.takeoff_pub = self.create_publisher(Empty, '/tello/takeoff', 10)
        self.land_pub = self.create_publisher(Empty, '/tello/land', 10)
        
        self.create_subscription(PoseStamped, '/tello/marker_pose', self.pose_cb, 10)
        self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info("--- SYSTEM START: Awaiting Tello Connection ---")
        time.sleep(2)
        self.get_logger().info("ACTION: Taking Off!")
        self.takeoff_pub.publish(Empty())
        
        #integral parameter
        self.integral_x = 0.0
        self.integral_y = 0.0

    def pose_cb(self, msg):
        if int(msg.header.frame_id) == self.target_id:
            z_measure = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
            self.kf.update(z_measure)
            if not self.marker_found:
                self.get_logger().info("TARGET SPOTTED: Locked onto ArUco Marker.")
            self.marker_found = True
            self.last_detection_time = time.time()

    def control_loop(self):
        now = time.time()
        time_since_last_seen = now - self.last_detection_time

        # 1. CRITICAL SAFETY: 10s loss = Land
        #if time_since_last_seen > 10.0:
        #    self.get_logger().error("SAFETY ALERT: Target lost for 10s. Landing immediately.")
         #   self.land_pub.publish(Empty())
          #  return

        # 2. INTERMEDIATE SAFETY: 1s loss = Stop/Hover
        #if time_since_last_seen > 1.0:
         #   self.get_logger().warn("TARGET LOST: Stopping movement...", once=False)
          #  self.integral_x = 0.0
           # self.integral_y = 0.0
            #stop_msg = Twist() # All values are 0.0 by default
            #self.vel_pub.publish(stop_msg)
            #return

        # 3. ACTIVE TRACKING (Marker is spotted and recently seen)
        self.kf.predict()
        curr_pos = self.kf.x[:3]
        
        err_x = curr_pos[0] 
        err_y = -curr_pos[1] 
        err_z = curr_pos[2] - self.target_dist
        
        #increase integral parameter
        if abs(err_x) > 0.02: self.integral_x += err_x * 0.1
        if abs(err_y) > 0.02: self.integral_y += err_y * 0.1
        
        #capping
        self.integral_x = np.clip(self.integral_x, -0.1, 0.1)
        self.integral_y = np.clip(self.integral_y, -0.1, 0.1)
        
        twist = Twist()
        # Gain Reactivity  with clipping 
        twist.linear.x = np.clip(err_z * 2.0, -0.2, 0.2) 
        twist.linear.y = np.clip((err_x * 0.4)+ (self.integral_x * 0.5), -0.2, 0.2)
        twist.linear.z = np.clip((err_y * 0.4)+ (self.integral_y * 0.5), -0.2, 0.2)
        
        # Deadzone logic
        if abs(err_z) < 0.03: twist.linear.x = 0.0
        if abs(err_x) < 0.05:  twist.linear.y = 0.0
        if abs(err_y) < 0.1: twist.linear.z = 0.0
            
        self.vel_pub.publish(twist)

        if (self.get_clock().now() - self.last_log_time).nanoseconds > 1e9:
            self.log_action_strings(err_x, err_y, err_z, curr_pos[2])
            self.last_log_time = self.get_clock().now()

    def log_action_strings(self, ex, ey, ez, dist):
        actions = []
        deadzone = 0.05
        if ez > deadzone: actions.append(f"FORWARD ({dist:.2f}m)")
        elif ez < -deadzone: actions.append(f"BACKING UP")
        else: actions.append("Z-STABLE")

        if ex > deadzone: actions.append("STRAFE RIGHT")
        elif ex < -deadzone: actions.append("STRAFE LEFT")
        else: actions.append("X-STABLE")

        if ey > deadzone: actions.append("RISING")
        elif ey < -deadzone: actions.append("DESCENDING")
        else: actions.append("Y-STABLE")

        log_msg = " | ".join(actions)
        self.get_logger().info(f"FLIGHT: {log_msg}")

def main():
    rclpy.init()
    node = TelloController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
