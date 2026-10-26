# Custom Interbotix PX150 ROS 2 Packages

This repository contains ROS 2 packages for describing, visualizing, planning for, and controlling an Interbotix PX150 arm:

- `custom_interbotix_xarms_description` — PX150 URDF/Xacro model, meshes, controller configuration, and a visualization launch file.
- `custom_interbotix_xarms_moveit` — MoveIt 2 configuration and launch files for the PX150.
- `interbotix_arms_controller` — a Python ROS 2 node that publishes a joint-group command to move the arm to its home position.

## Requirements

- A ROS 2 installation and ROS 2 distribution compatible with the packages and their dependencies.
- `colcon` and `rosdep`.
- MoveIt 2 for the MoveIt configuration.
- Interbotix ROS message packages for the controller (`interbotix_xs_msgs`).

## Build

Place or clone this repository in the `src` directory of a ROS 2 workspace. From the workspace root, source your ROS 2 installation, install package dependencies, and build:

```bash
source /opt/ros/<ros_distro>/setup.bash
rosdep update
rosdep install --from-paths src --ignore-src -r -y
colcon build
source install/setup.bash
```

Replace `<ros_distro>` with the name of your installed ROS 2 distribution.

## Launch

Launch the robot description in RViz with the joint-state publisher GUI:

```bash
ros2 launch custom_interbotix_xarms_description display.launch.py
```

Launch the MoveIt demo:

```bash
ros2 launch custom_interbotix_xarms_moveit demo.launch.py
```
