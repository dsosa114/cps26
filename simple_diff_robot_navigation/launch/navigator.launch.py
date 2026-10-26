from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os
from ament_index_python.packages import get_package_share_path

def generate_launch_description():

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )

    map_path_arg = DeclareLaunchArgument(
            'map',
            default_value='',
            description='Path to map yaml file to load'
        )

    gz_launch_path = os.path.join(get_package_share_path("simple_diff_robot_gz_bringup"),
                                  'launch')

    nav2_bringup_path = os.path.join(get_package_share_path("nav2_bringup"),
                                      'launch')
    
    gz_sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            gz_launch_path,
            "/gz_sim.launch.py"
        ])
    )

    nav2_bringup_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                nav2_bringup_path,
                "/bringup_launch.py"
            ]), launch_arguments={'use_sim_time': LaunchConfiguration('use_sim_time'), 'map': LaunchConfiguration('map')}.items()
        )


    return LaunchDescription([
        use_sim_time_arg,
        map_path_arg,
        gz_sim_launch,
        nav2_bringup_launch
    ])