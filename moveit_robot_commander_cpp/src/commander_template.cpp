#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.hpp>
#include <example_interfaces/msg/bool.hpp>
#include <example_interfaces/msg/float64_multi_array.hpp>


using MoveGroupInterface = moveit::planning_interface::MoveGroupInterface;
using Bool = example_interfaces::msg::Bool;
using FloatArray = example_interfaces::msg::Float64MultiArray;

using namespace std::placeholders;

class MoveItCommander
{
public:
    MoveItCommander(std::shared_ptr<rclcpp::Node> node)
    {
        node_ = node;
        arm_ = std::make_shared<MoveGroupInterface>(node_, "manipulator");
        arm_->setMaxVelocityScalingFactor(0.1);
        arm_->setMaxAccelerationScalingFactor(0.1);
        gripper_ = std::make_shared<MoveGroupInterface>(node_, "gripper");

        open_gripper_sub_ = node_->create_subscription<Bool>("open_gripper", 10, std::bind(&MoveItCommander::OpenGripperCallback, this, _1));
        joint_cmd_sub_ = node_->create_subscription<FloatArray>("joint_command", 10, std::bind(&MoveItCommander::JointCommandCallback, this, _1));

    }
    
    void goToNamedTarget(const std::string& target_name)
    {
        arm_->setStartStateToCurrentState();
        arm_->setNamedTarget(target_name);
        planAndExecute(arm_);
    }

    void goToJointTarget(const std::vector<double>& joint_values)
    {
        arm_->setStartStateToCurrentState();
        arm_->setJointValueTarget(joint_values);
        planAndExecute(arm_);
    }

    void goToPoseTarget(double x, double y, double z, double roll, double pitch, double yaw, bool cartesian_path = false)
    {
        geometry_msgs::msg::PoseStamped target_pose;
        target_pose.pose.position.x = x;
        target_pose.pose.position.y = y;
        target_pose.pose.position.z = z;

        tf2::Quaternion q;
        q.setRPY(roll, pitch, yaw);
        q = q.normalize();
        target_pose.pose.orientation.x = q.x();
        target_pose.pose.orientation.y = q.y();
        target_pose.pose.orientation.z = q.z();
        target_pose.pose.orientation.w = q.w();

        arm_->setStartStateToCurrentState();

        if (!cartesian_path)
        {
            arm_->setPoseTarget(target_pose);
            planAndExecute(arm_);
        }
        else
        {
            std::vector<geometry_msgs::msg::Pose> waypoints;
            waypoints.push_back(target_pose.pose);

            moveit_msgs::msg::RobotTrajectory trajectory;
            const double eef_step = 0.01;
            double fraction = arm_->computeCartesianPath(waypoints, eef_step, trajectory);
            if (fraction == 1)
            {
                arm_->execute(trajectory);
            }
            else
            {
                RCLCPP_ERROR(node_->get_logger(), "Cartesian path planning failed");
            }
        }
        arm_->setPoseTarget(target_pose);
        planAndExecute(arm_);
    }

    void OpenGripper()
    {
        gripper_->setStartStateToCurrentState();
        gripper_->setNamedTarget("gripper_open");
        planAndExecute(gripper_);
    }  

    void CloseGripper()
    {
        gripper_->setStartStateToCurrentState();
        gripper_->setNamedTarget("gripper_closed");
        planAndExecute(gripper_);
    }

private:

    void planAndExecute(const std::shared_ptr<MoveGroupInterface>& interface)
    {
        MoveGroupInterface::Plan plan;
        bool success = interface->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS;
        if (success)
        {
            interface->execute(plan);
        }
        else
        {
            RCLCPP_ERROR(node_->get_logger(), "Planning failed");
        }
    }

    void OpenGripperCallback(const Bool::SharedPtr msg)
    {
        if (msg->data)
        {
            OpenGripper();
        }
        else
        {
            CloseGripper();
        }
    }

    void JointCommandCallback(const FloatArray::SharedPtr msg)
    {
        auto joint_values = msg->data;
        if (joint_values.size() != arm_->getJointNames().size())
        {
            RCLCPP_ERROR(node_->get_logger(), "Invalid joint command size");
            return;
        }
        goToJointTarget(joint_values);
    }

    rclcpp::Node::SharedPtr node_;
    std::shared_ptr<MoveGroupInterface> arm_;
    std::shared_ptr<MoveGroupInterface> gripper_;

    rclcpp::Subscription<Bool>::SharedPtr open_gripper_sub_;
    rclcpp::Subscription<FloatArray>::SharedPtr joint_cmd_sub_;
};

int main(int argc, char const *argv[])
{
    rclcpp::init(argc, argv);
    /* code */
    auto node = std::make_shared<rclcpp::Node>("moveit_commander_node");
    auto commander = MoveItCommander(node);
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
