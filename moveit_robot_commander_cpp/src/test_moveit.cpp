#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.hpp>

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("moveit_test_node");

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);

  auto spinner = std::thread([&executor]() { executor.spin(); });

  auto arm = moveit::planning_interface::MoveGroupInterface(node, "ur_manipulator");
  arm.setMaxVelocityScalingFactor(0.1);
  arm.setMaxAccelerationScalingFactor(0.1);

  // Named goal

  // arm.setStartStateToCurrentState();
  // arm.setNamedTarget("test_configuration");

  // moveit::planning_interface::MoveGroupInterface::Plan test_plan;
  // bool success = (arm.plan(test_plan) == moveit::core::MoveItErrorCode::SUCCESS);

  // if (success)
  // {
  //   RCLCPP_INFO(node->get_logger(), "Plan successful, executing...");
  //   arm.execute(test_plan);
  // }
  // else
  // {
  //   RCLCPP_ERROR(node->get_logger(), "Plan failed!");
  // }

  // arm.setStartStateToCurrentState();
  // arm.setNamedTarget("home");

  // moveit::planning_interface::MoveGroupInterface::Plan home_plan;
  // bool success2 = (arm.plan(home_plan) == moveit::core::MoveItErrorCode::SUCCESS);

  // if (success2)
  // {
  //   RCLCPP_INFO(node->get_logger(), "Plan successful, executing...");
  //   arm.execute(home_plan);
  // }
  // else
  // {
  //   RCLCPP_ERROR(node->get_logger(), "Plan failed!");
  // }

  /*Joint Goal*/
  // arm.setStartStateToCurrentState();
  // std::vector<double> joint_group_positions = {0.0, -1.5707, -1.5707, 0.0, 1.5707, 0.0};
  // arm.setJointValueTarget(joint_group_positions);

  // moveit::planning_interface::MoveGroupInterface::Plan joint_plan1;
  // bool success = (arm.plan(joint_plan1) == moveit::core::MoveItErrorCode::SUCCESS);

  // if (success)
  // {
  //   RCLCPP_INFO(node->get_logger(), "Plan successful, executing...");
  //   arm.execute(joint_plan1);
  // }
  // else
  // {
  //   RCLCPP_ERROR(node->get_logger(), "Plan failed!");
  // }

  /* Pose Goal*/
  tf2::Quaternion q;
  q.setRPY(-1.5707, 0.0, 1.5707);
  q = q.normalize();

  geometry_msgs::msg::PoseStamped target_pose;

  target_pose.header.frame_id = "base_link";
  target_pose.pose.position.x = -0.687885;
  target_pose.pose.position.y = 0.17415;
  target_pose.pose.position.z = 0.91315;
  target_pose.pose.orientation.x = q.getX();
  target_pose.pose.orientation.y = q.getY();
  target_pose.pose.orientation.z = q.getZ();
  target_pose.pose.orientation.w = q.getW();

  arm.setStartStateToCurrentState();
  arm.setPoseTarget(target_pose);

  moveit::planning_interface::MoveGroupInterface::Plan pose_plan;
  bool success = (arm.plan(pose_plan) == moveit::core::MoveItErrorCode::SUCCESS);

  if (success)
  {
    RCLCPP_INFO(node->get_logger(), "Plan successful, executing...");
    arm.execute(pose_plan);
  }
  else
  {
    RCLCPP_ERROR(node->get_logger(), "Plan failed!");
  }

  /* Cartesian Path*/
  std::vector<geometry_msgs::msg::Pose> waypoints;
  geometry_msgs::msg::Pose target_pose2 = arm.getCurrentPose().pose;
  target_pose2.position.z -= 0.1;
  target_pose2.position.y += 0.2;
  waypoints.push_back(target_pose2);

  geometry_msgs::msg::Pose target_pose3 = target_pose2;
  target_pose3.position.x += 0.1;
  waypoints.push_back(target_pose3);

  geometry_msgs::msg::Pose target_pose4 = target_pose3;
  target_pose4.position.z += 0.1;
  waypoints.push_back(target_pose4);

  moveit_msgs::msg::RobotTrajectory trajectory;
  double fraction = arm.computeCartesianPath(waypoints, 0.01, trajectory);

  if (fraction == 1)
  {
    RCLCPP_INFO(node->get_logger(), "Cartesian path computed successfully, executing...");
    arm.execute(trajectory);
  }
  else
  {
    RCLCPP_ERROR(node->get_logger(), "Cartesian path computation failed!");
  }


  rclcpp::shutdown();
  spinner.join();
  return 0;
}  