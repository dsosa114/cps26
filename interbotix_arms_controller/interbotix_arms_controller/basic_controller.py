import rclpy
from rclpy.node import Node
from interbotix_xs_msgs.msg import JointGroupCommand
from std_msgs.msg import Empty

class InterbotixBasicController(Node): 
    def __init__(self):
        super().__init__('interbotix_basic_controller_node')

        self.subscription = self.create_subscription(Empty, 'robot_home', self.listener_callback, 10)
        self.home_publisher = self.create_publisher(JointGroupCommand, 'commands/joint_group', 10)
        self.get_logger().info('Basic Controller node has been started.') 

    def listener_callback(self, msg):
        cmd = JointGroupCommand()
        cmd.name = 'arm'
        cmd.cmd = [0.0, 0.0, 0.0, 0.0, 0.0]  # Set the desired joint positions for the arm
        self.home_publisher.publish(cmd)

def main(args=None):
    rclpy.init(args=args) # Initialize the ROS2 Python client library
    node = InterbotixBasicController() # Change the name of the class to your node's name
    rclpy.spin(node) # Keep the node running until it is shut down (crtl+c)
    node.destroy_node() # Destroy the node explicitly (optional)
    rclpy.shutdown() # Shutdown the ROS2 Python client library

if __name__ == '__main__':
    main()

