# my_robot_utils (`rbu_package`)

Collection of helper functions and classes for controlling Universal Robots (UR) manipulators, either in simulation (RoboDK) or on the real robot (RTDE), plus kinematics, trajectory, vision and motion-capture utilities.

## Installation

Requires Python >= 3.8 (3.10+ recommended).

```bash
cd /path/to/ur_manipulators_helpers/robot_utils
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install .            # or `pip install -e .` for development
```

Notes:
- `pyzed` (ZED camera) is not on PyPI; install the [ZED SDK](https://www.stereolabs.com/docs/app-development/python/install) and its Python API only if you use the ZED camera code.
- `openni`, `PyOpenGL`, `ultralytics` and `pyzed` are only needed by `camera_utils.py` and the viewers; `ur_rtde`/`robodk` are needed by `robot_utils.py`.
- `setup.py` currently has an empty `install_requires`, so `pip install .` does not pull dependencies; use `requirements.txt`.
- Several modules import via `ur_manipulators_helpers.robot_utils.rbu_package...`, while the installed package is `rbu_package`. Run from the workspace `src/` directory (so that path resolves), or change those imports to `rbu_package...`.

## Contents (`rbu_package/`)

| Module | Description |
|---|---|
| `robot_utils.py` | `RDK_Connection` (RoboDK simulation/real robot: connect, move linear/joint, FK/IK, gripper, cameras, targets), `RTDE_Connection` (real UR via RTDE: moveL/servoL, teach/record trajectories, TCP-relative moves, pose/joint reading, FK/IK, Swift visualization), `ExperimentBatchCollector` (collect and save experiment data) |
| `robo_functions.py` | Symbolic (`s_*`) and numeric rotation matrices, homogeneous transforms, rotation vector/quaternion/Euler conversions, DH and Product-of-Exponentials forward kinematics, Jacobians, interpolation/SLERP, UR-specific helpers |
| `robotiq_gripper.py` | `RobotiqGripper`: socket driver for Robotiq grippers (tested with Hand-E) |
| `math_utils.py` | Trajectory generation/interception, smoothing (polynomial, moving average, EMA filters), `TrajectoryComparator` (DTW, RMSE, etc.), numeric derivatives, Jacobian, Newton's method |
| `camera_utils.py` | `zed_camera`, `ps_camera`, `wrist_camera`, `ViewTransformer`, `YOLOv8_Detection` |
| `opencv_basic_functions.py` | Image I/O, contours, keypoints (SIFT/SURF), Hough transforms, color masks, template matching, `ObjectTracker`, `PositionChangeDetector` |
| `plot_utils.py` | `PlotHelper` (2D/3D plotting, robot/frames visualization), `Arrow3D` |
| `NatNetClient.py`, `MoCapData.py`, `DataDescriptions.py` | OptiTrack NatNet motion-capture client and data structures |
| `ogl_viewer/`, `cv_viewer/` | ZED 3D (OpenGL) and 2D tracking viewers |

## Usage

```python
from rbu_package import robo_functions as rf
from rbu_package.robot_utils import RTDE_Connection, RDK_Connection

rf.p_print(rf.rotz(rf.deg2rad(90)))   # pretty-print a rotation matrix
```

Consult the docstrings in each module (e.g. `RTDE_Connection.start_connection`, `robot_move_linear`) for exact arguments. See `../ur_rtde.ipynb` for a working example.

## Safety

Methods in `RTDE_Connection` move a real robot. Test in simulation first, keep the emergency stop within reach, and use low speeds/accelerations.