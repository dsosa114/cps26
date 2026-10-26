from launch import LaunchDescription
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterValue
from launch.substitutions import Command, LaunchConfiguration
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, RegisterEventHandler, LogInfo
from launch.event_handlers import OnProcessStart
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os
from ament_index_python.packages import get_package_share_path, get_package_share_directory

def generate_launch_description():

    gz_launch_path = os.path.join(get_package_share_path("ros_gz_sim"),
                                  'launch')
    
    urdf_path = os.path.join(get_package_share_path("simple_diff_robot_description"),
                             'urdf', 'simple_diff_robot.urdf.xacro')

    rviz_config_path = os.path.join(get_package_share_path("simple_diff_robot_description"),
                                    'rviz', 'default_config.rviz')

    bridge_config_path = os.path.join(get_package_share_path("simple_diff_robot_gz_bringup"),
                                      'config', 'gz_bridge_config.yaml')

    robot_description = ParameterValue(Command(['xacro ', urdf_path]), value_type= str)

    robot_state_publisher = Node(
        name='simple_diff_robot_state_publisher',
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description}]
    )

    rviz_node = Node(
        name='rviz2',
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config_path]    
    )

    gz_sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            gz_launch_path,
            "/gz_sim.launch.py"
        ]), launch_arguments={'gz_args': 'my_first_home.sdf -r -v 4'}.items()
    )

    gz_model_create = Node(
        name='gz_model_create',
        package='ros_gz_sim',
        executable='create',
        arguments=['-topic', '/robot_description']
    )

    gz_bridge = Node(
        name='gz_bridge',
        package='ros_gz_bridge',
        executable='parameter_bridge',
        parameters=[{'config_file': bridge_config_path}],
    )

    return LaunchDescription([
        robot_state_publisher,
        rviz_node,
        gz_sim_launch,
        gz_model_create,
        gz_bridge
    ])