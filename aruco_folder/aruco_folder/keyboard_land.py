import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty
from pynput import keyboard

class KeyboardLandNode(Node):
    def __init__(self):
        super().__init__('keyboard_land_node')
        self.land_pub = self.create_publisher(Empty, '/tello/land', 10)
        self.get_logger().info("Emergency Land Node Active. PRESS [SPACE] TO LAND.")
        
        # Start the keyboard listener
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

    def on_press(self, key):
        try:
            if key == keyboard.Key.space:
                self.get_logger().warn("SPACE BAR PRESSED! Sending Land Command...")
                self.land_pub.publish(Empty())
        except AttributeError:
            pass

def main():
    rclpy.init()
    node = KeyboardLandNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
