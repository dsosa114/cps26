# ROS 2 Robotics Workspace

This workspace contains two robots under active development:

- **PX150 manipulator** — based on the Interbotix PX150 arm. It can be controlled with `ros2_control` and used with MoveIt 2 for motion planning.
- **Simple differential-drive robot** — can be navigated with Nav2 and visualized/simulated in Gazebo Harmonic.

The workspace is still under development, so its packages, setup instructions, and launch workflows may change.

## Packages

### PX150 manipulator

- `custom_interbotix_xarms_description` — PX150 URDF/Xacro model, meshes, controller configuration, and visualization launch files.
- `custom_interbotix_xarms_moveit` — MoveIt 2 configuration and launch files for the PX150.
- `interbotix_arms_controller` — a Python ROS 2 controller node for sending a home command to the arm.

### Differential-drive robot

- `simple_diff_robot_description` — robot URDF/Xacro description and RViz configuration.
- `simple_diff_robot_gz_bringup` — Gazebo Harmonic simulation bringup and ROS-Gazebo bridge configuration.
- `simple_diff_robot_navigation` — navigation and Cartographer launch files for the robot.

## Requirements

Install a ROS 2 distribution compatible with the packages, along with `colcon` and `rosdep`. The relevant robot workflows also require their ROS 2 dependencies, including MoveIt 2, Nav2, Gazebo Harmonic, and the Interbotix message packages used by the manipulator controller.

## Build

Place this repository in the `src` directory of a ROS 2 workspace. From the workspace root, source your ROS 2 installation, install package dependencies, and build:

```bash
source /opt/ros/<ros_distro>/setup.bash
rosdep update
rosdep install --from-paths src --ignore-src -r -y
colcon build
source install/setup.bash
```

Replace `<ros_distro>` with your installed ROS 2 distribution. Source `install/setup.bash` in each new terminal before using the packages.

## Fork and customize

Students are welcome to fork this repository and modify their copy as needed for coursework, experiments, or personal projects.

1. Open this repository on GitHub and select **Fork**. Choose your GitHub account as the destination.
2. Clone your fork into the `src` directory of your ROS 2 workspace:

```bash
cd ~/ros2_ws/src
git clone https://github.com/<your-username>/<repository-name>.git
cd <repository-name>
```

Replace `<your-username>` and `<repository-name>` with the owner and repository name of your fork. If you cloned the fork already, skip the `git clone` command and navigate to its directory.

Create a branch for your changes, then edit and build the packages as needed:

```bash
git switch -c <your-branch-name>
```

Commit and push your changes to your fork whenever you want to save or share them:

```bash
git add .
git commit -m "Describe your changes"
git push -u origin <your-branch-name>
```

You can customize your fork in any way. Changes to your fork do not affect the original repository.
