import rclpy
from rclpy.node import Node

class NodeTemplate(Node): # Change the name of the class to your node's name
    def __init__(self):
        super().__init__('node_template') # Change the name of the node to your node's name
        self.get_logger().info('Node template has been started.') #Change the initial message to your node's message

def main(args=None):
    rclpy.init(args=args) # Initialize the ROS2 Python client library
    node = NodeTemplate() # Change the name of the class to your node's name
    rclpy.spin(node) # Keep the node running until it is shut down (crtl+c)
    node.destroy_node() # Destroy the node explicitly (optional)
    rclpy.shutdown() # Shutdown the ROS2 Python client library

if __name__ == '__main__':
    main()



