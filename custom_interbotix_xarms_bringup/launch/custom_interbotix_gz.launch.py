from ament_index_python.packages import get_package_share_path
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetParameter
from launch_ros.descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import (
    generate_move_group_launch,
    generate_moveit_rviz_launch,
)


def _launch_next_on_success(process_name, next_actions):
    def callback(event, context):
        if event.returncode != 0:
            raise RuntimeError(
                f'{process_name} exited with code {event.returncode}; '
                'stopping the bringup sequence'
            )
        return next_actions

    return callback


def generate_launch_description():
    use_sim_time = ParameterValue(
        LaunchConfiguration('use_sim_time'),
        value_type=bool,
    )

    description_package_path = get_package_share_path(
        'custom_interbotix_xarms_description'
    )
    robot_description_file = description_package_path / 'urdf' / 'px150.urdf.xacro'
    robot_description = ParameterValue(
        Command(['xacro ', str(robot_description_file), ' gz_sim:=true']),
        value_type=str,
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[
            {
                'robot_description': robot_description,
                'use_sim_time': use_sim_time,
            }
        ],
    )

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare('ros_gz_sim'), 'launch', 'gz_sim.launch.py']
            )
        ),
        launch_arguments={'gz_args': '-r -v 4'}.items(),
    )

    clock_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='clock_bridge',
        arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen',
    )

    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        name='spawn_px150',
        arguments=['-name', 'px150', '-topic', '/robot_description'],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen',
    )

    controller_spawners = [
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=[
                controller,
                '--controller-manager',
                '/controller_manager',
                '--controller-manager-timeout',
                '60',
            ],
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen',
        )
        for controller in (
            'joint_state_broadcaster',
            'arm_controller',
            'gripper_controller',
        )
    ]

    moveit_config = MoveItConfigsBuilder(
        'px150',
        package_name='custom_interbotix_xarms_moveit',
    ).to_moveit_configs()

    move_group_launch = GroupAction(
        scoped=True,
        actions=[
            SetParameter(name='use_sim_time', value=use_sim_time),
            *generate_move_group_launch(moveit_config).entities,
        ],
    )
    moveit_rviz_launch = GroupAction(
        scoped=True,
        actions=[
            SetParameter(name='use_sim_time', value=use_sim_time),
            *generate_moveit_rviz_launch(moveit_config).entities,
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'use_sim_time',
                default_value='true',
                description='Use Gazebo simulation time',
            ),
            RegisterEventHandler(
                OnProcessExit(
                    target_action=spawn_robot,
                    on_exit=_launch_next_on_success(
                        'PX150 spawn', [controller_spawners[0]]
                    ),
                )
            ),
            RegisterEventHandler(
                OnProcessExit(
                    target_action=controller_spawners[0],
                    on_exit=_launch_next_on_success(
                        'joint_state_broadcaster spawner',
                        [controller_spawners[1]],
                    ),
                )
            ),
            RegisterEventHandler(
                OnProcessExit(
                    target_action=controller_spawners[1],
                    on_exit=_launch_next_on_success(
                        'arm_controller spawner',
                        [controller_spawners[2]],
                    ),
                )
            ),
            RegisterEventHandler(
                OnProcessExit(
                    target_action=controller_spawners[2],
                    on_exit=_launch_next_on_success(
                        'gripper_controller spawner',
                        [move_group_launch, moveit_rviz_launch],
                    ),
                )
            ),
            robot_state_publisher,
            gz_sim,
            clock_bridge,
            spawn_robot,
        ]
    )
