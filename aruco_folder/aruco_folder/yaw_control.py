import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Empty
import numpy as np
import time
from scipy.linalg import cholesky

class UKF_CTRV:
    def __init__(self, dt):
        self.dt = dt
        # State: [x, y, z, v, yaw, yaw_rate]
        self.n_x = 6
        self.x = np.zeros(self.n_x)
        self.P = np.eye(self.n_x) * 0.1
        
        # UKF Parameters
        self.alpha = 0.001
        self.kappa = 0
        self.beta = 2
        self.lambda_ = self.alpha**2 * (self.n_x + self.kappa) - self.n_x
        
        # Weights for mean and covariance
        self.weights_m = np.full(2 * self.n_x + 1, 1 / (2 * (self.n_x + self.lambda_)))
        self.weights_m[0] = self.lambda_ / (self.n_x + self.lambda_)
        self.weights_c = self.weights_m.copy()
        self.weights_c[0] += (1 - self.alpha**2 + self.beta)

        # Noise
        self.Q = np.diag([0.05, 0.05, 0.05, 0.1, 0.1, 0.05]) # Process noise
        self.R = np.eye(3) * 0.0005 # Measurement noise (x, y, z)

    def generate_sigma_points(self):
        sigmas = np.zeros((2 * self.n_x + 1, self.n_x))
        U = cholesky((self.n_x + self.lambda_) * self.P)
        sigmas[0] = self.x
        for k in range(self.n_x):
            sigmas[k + 1] = self.x + U[k]
            sigmas[k + 1 + self.n_x] = self.x - U[k]
        return sigmas

    def predict(self):
        sigmas = self.generate_sigma_points()
        sigmas_pred = np.zeros_like(sigmas)

        for i, s in enumerate(sigmas):
            x, y, z, v, yaw, yaw_d = s
            # CTRV Motion Equations
            if abs(yaw_d) > 0.001:
                x_n = x + (v/yaw_d) * (np.sin(yaw + yaw_d*self.dt) - np.sin(yaw))
                y_n = y + (v/yaw_d) * (np.cos(yaw) - np.cos(yaw + yaw_d*self.dt))
            else:
                x_n = x + v * self.dt * np.cos(yaw)
                y_n = y + v * self.dt * np.sin(yaw)
            
            sigmas_pred[i] = [x_n, y_n, z + 0, v, yaw + yaw_d*self.dt, yaw_d]

        # Predicted Mean
        self.x = np.dot(self.weights_m, sigmas_pred)
        # Predicted Covariance
        self.P = self.Q.copy()
        for i in range(2 * self.n_x + 1):
            y = sigmas_pred[i] - self.x
            self.P += self.weights_c[i] * np.outer(y, y)
        return sigmas_pred

    def update(self, z):
        sigmas_pred = self.predict()
        # Map sigmas to measurement space (x, y, z)
        z_sigmas = sigmas_pred[:, :3]
        z_pred = np.dot(self.weights_m, z_sigmas)
        
        S = self.R.copy()
        for i in range(2 * self.n_x + 1):
            res = z_sigmas[i] - z_pred
            S += self.weights_c[i] * np.outer(res, res)
            
        # Cross correlation
        T = np.zeros((self.n_x, 3))
        for i in range(2 * self.n_x + 1):
            T += self.weights_c[i] * np.outer(sigmas_pred[i] - self.x, z_sigmas[i] - z_pred)
            
        K = T @ np.linalg.inv(S)
        self.x += K @ (z - z_pred)
        self.P -= K @ S @ K.T

class TelloYawController(Node):
    def __init__(self):
        super().__init__('tello_yaw_controller')
        self.ukf = UKF_CTRV(dt=0.1)
        self.last_detection = time.time()
        
        # Publishers
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.takeoff_pub = self.create_publisher(Empty, '/tello/takeoff', 10)
        
        # Subscriber
        self.create_subscription(PoseStamped, '/tello/marker_pose', self.pose_cb, 10)
        self.create_timer(0.1, self.control_loop)
        
        # Autonomous Takeoff
        self.get_logger().info("SYSTEM: Initializing... Takeoff in 2 seconds.")
        time.sleep(2)
        self.takeoff_pub.publish(Empty())

    def pose_cb(self, msg):
        z = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        self.ukf.update(z)
        self.last_detection = time.time()

    def control_loop(self):
        # Safety: If marker lost, stop rotating
        #if time.time() - self.last_detection > 2.0:
         #   self.vel_pub.publish(Twist()) # Send zero velocity
          #  return

        self.ukf.predict()
        state = self.ukf.x # [x, y, z, v, yaw, yaw_rate]
        
        # ERROR: Marker's horizontal offset from camera center
        # If state[0] is positive, marker is to the right.
        marker_x_offset = state[0]
        
        twist = Twist()
        # Rotation Gain: High enough to be snappy, capped to stay slow
        # We use a negative gain because if marker is at +X (right), 
        # Tello needs positive angular.z to turn right.
        twist.angular.z = np.clip(marker_x_offset * 3.0, -0.4, 0.4)
        
        # Ensure all linear movement is zero (we only want rotation)
        twist.linear.x = 0.0
        twist.linear.y = 0.0
        twist.linear.z = 0.0

        # Deadzone: Stop vibrating if error is less than 2cm
        if abs(marker_x_offset) < 0.02:
            twist.angular.z = 0.0
            self.get_logger().info("STATUS: Target Centered", once=True)
        else:
            direction = "RIGHT" if marker_x_offset > 0 else "LEFT"
            self.get_logger().info(f"ACTION: Rotating {direction} to center marker")

        self.vel_pub.publish(twist)

def main():
    rclpy.init()
    node = TelloYawController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
