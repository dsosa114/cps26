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

## Upload this folder to a new GitHub repository

First, create a new **empty** repository on GitHub. Do not initialize it with a README, license, or `.gitignore`. Then run these commands in a terminal:

```bash
cd /home/daniel/Workspaces/cps26_ws/src
git init
git branch -M main
git add README.md custom_interbotix_xarms_description custom_interbotix_xarms_moveit interbotix_arms_controller
git commit -m "Add custom Interbotix PX150 ROS 2 packages"
git remote add origin https://github.com/<your-username>/<your-repository>.git
git push -u origin main
```

Replace `<your-username>` and `<your-repository>` with the GitHub account and repository name. The explicit `git add` paths include the README and ROS packages without adding the local `.vscode` workspace files.
