from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterValue
from launch.substitutions import Command
import os
from ament_index_python.packages import get_package_share_path

def generate_launch_description():

    gz_sim = LaunchConfiguration('gz_sim')
    rviz = LaunchConfiguration('rviz')
    activate_jsp_gui = LaunchConfiguration('activate_jsp_gui')

    urdf_path = os.path.join(get_package_share_path("custom_interbotix_xarms_description"),
                             'urdf', 'px150.urdf.xacro')

    rviz_config_path = os.path.join(get_package_share_path("custom_interbotix_xarms_description"),
                                    'rviz', 'rviz_config.rviz')

    robot_description = ParameterValue(Command(['xacro ', urdf_path, 
                                                ' ',
                                                'gz_sim:=',  
                                                gz_sim]), value_type= str)

    #xacro custom_interbotix_xarms_description/urdf/px150.urdf.xacro gz_sim:=true
    robot_state_publisher = Node(
        name='robot_state_publisher',
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description}]
    )

    # joint_state_publisher = Node(
    #     name='joint_state_publisher',
    #     package='joint_state_publisher',
    #     executable='joint_state_publisher',
    #     condition=UnlessCondition(activate_jsp_gui)
    # )
    
    joint_state_publisher_gui = Node(
        name='joint_state_publisher_gui',
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        condition=IfCondition(activate_jsp_gui)
    )

    rviz_node = Node(
        name='rviz2',
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config_path],
        condition=IfCondition(rviz)
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'gz_sim', 
            default_value='false', 
            description='Set to true to enable Gazebo simulation'
        ),
        DeclareLaunchArgument(
            'rviz', 
            default_value='true', 
            description='Set to true to enable RViz'
        ),
        DeclareLaunchArgument(
            'activate_jsp_gui', 
            default_value='true', 
            description='Set to true to enable the Joint State Publisher GUI'
        ),
        robot_state_publisher,
        # joint_state_publisher,
        joint_state_publisher_gui,
        rviz_node
    ])