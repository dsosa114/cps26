import rclpy
from nav2_simple_commander.robot_navigator import BasicNavigator
import tf_transformations
from geometry_msgs.msg import PoseStamped

def main():
   rclpy.init()
   navigator = BasicNavigator()
   
   q = tf_transformations.quaternion_from_euler(0.0, 0.0, 0.0)
   initial_pose = PoseStamped()
   initial_pose.header.frame_id = 'map'
   initial_pose.header.stamp = navigator.get_clock().now().to_msg()
   initial_pose.pose.position.x = 0.0
   initial_pose.pose.position.y = 0.0
   initial_pose.pose.position.z = 0.0
   initial_pose.pose.orientation.x = q[0]
   initial_pose.pose.orientation.y = q[1]
   initial_pose.pose.orientation.z = q[2]
   initial_pose.pose.orientation.w = q[3]

   navigator.setInitialPose(initial_pose)
   
   navigator.waitUntilNav2Active()

   navigator.get_logger().info('Waiting for 5 seconds before sending a goal')
   pose_1 = PoseStamped()
   pose_1.header.frame_id = 'map'
   pose_1.header.stamp = navigator.get_clock().now().to_msg()
   pose_1.pose.position.x = 2.0
   pose_1.pose.position.y = 0.0
   pose_1.pose.position.z = 0.0
   q = tf_transformations.quaternion_from_euler(0.0, 0.0, 1.57)
   pose_1.pose.orientation.x = q[0]
   pose_1.pose.orientation.y = q[1]
   pose_1.pose.orientation.z = q[2]
   pose_1.pose.orientation.w = q[3]

   navigator.goToPose(pose_1)
   while not navigator.isTaskComplete():
      feedback = navigator.getFeedback()
      if feedback:
         print('Distance remaining: ' + str(feedback.distance_remaining))  

#    navigator.followWaypoints([])
#    while not navigator.isTaskComplete():
#          feedback = navigator.getFeedback()
#          if feedback:
#             print('Distance remaining: ' + str(feedback.distance_remaining))

   navigator.destroy_node()
   rclpy.shutdown()

if __name__ == '__main__':
    main()
