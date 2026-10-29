import ur_manipulators_helpers.robot_utils.rbu_package.robotiq_gripper
import ur_manipulators_helpers.robot_utils.rbu_package.robo_functions as rf
from robodk.robolink import *
from robodk.robomath import *
import numpy as np
from scipy.spatial.transform import Rotation as R
import warnings
import pandas as pd
import cv2

from rtde_control import RTDEControlInterface as RTDEControl
import rtde_receive
import rtde_io
#from roboticstoolbox.robot.Robot import Robot
#import roboticstoolbox as rtb
import swift
import asyncio
import platform
import datetime
import time
import pickle
import os


class ExperimentBatchCollector:
    """
    A class to collect data from multiple experimental iterations,
    process them in a batch, and provide options for persistent storage.
    """
    def __init__(self):
        self.batch = []
        print("ExperimentBatchCollector initialized.")

    def add_iteration(self, baseline, computed, executed, dt, exp_params):
        """Stores the complete set of parameters for a single iteration."""
        iteration_data = {
            'baseline': baseline,
            'computed': computed,
            'executed': executed,
            'dt': dt,
            'exp_params': exp_params
        }
        self.batch.append(iteration_data)
        print(f"  Iteration {exp_params.get('experiment_id', len(self.batch))} data added to batch.")

    def process_batch(self, analyzer):
        """
        Runs the comparison function on the analyzer for all stored iterations.
        
        Args:
            analyzer (TrajectoryComparator): The analyzer object with the run_comparison method.
        """
        if not self.batch:
            print("Batch is empty. Nothing to process.")
            return

        print(f"\n--- Starting to process a batch of {len(self.batch)} iterations... ---")
        for iteration in self.batch:
            analyzer.run_comparison(**iteration)
        
        print(f"--- Batch processing complete. ---")
    
    def clear_batch(self):
        """Clears the collected data from memory."""
        self.batch = []
        print("Batch cleared.")

    # NEW: Method to save the entire batch to a file
    def save_batch(self, filepath='experiment_batch.pkl'):
        """
        Saves the entire collected batch to a file using pickle.

        Args:
            filepath (str): The path to the file where the batch will be saved.
        """
        if not self.batch:
            print("Warning: Batch is empty. Nothing to save.")
            return
            
        with open(filepath, 'wb') as f:
            pickle.dump(self.batch, f)
        print(f"Successfully saved {len(self.batch)} iterations to '{filepath}'")

    # NEW: Class method to load a batch from a file
    @classmethod
    def from_file(cls, filepath='experiment_batch.pkl'):
        """
        Loads a batch from a pickle file and returns a new collector instance
        populated with the data.

        Args:
            filepath (str): The path to the file to load.
        
        Returns:
            ExperimentBatchCollector: A new instance containing the loaded data.
        """
        if not os.path.exists(filepath):
            print(f"Error: File not found at '{filepath}'")
            return None

        with open(filepath, 'rb') as f:
            loaded_batch_data = pickle.load(f)
        
        collector = cls()
        collector.batch = loaded_batch_data
        print(f"Successfully loaded {len(collector.batch)} iterations from '{filepath}'")
        return collector

class RDK_Connection:
    def __init__(self,
                 rdk_ip: str,
                 sim: bool = True) -> None:
        '''
        RDK Connection class implements multiple functionalities to make easier
        the programming of UR robots using RoboDK Python API. By default the
        construction of the object tries to establish the connection with RoboDK
        software.\n\n
        Arguments:\n
            \trdk_ip: IP address of the computer hosting the RoboDK interface.
            \tsim: A bool parameter to indicate if the movements will be done on
            simulation or in the real robot.

        '''
        self.rdk_ = rdk_ip
        self.robot = None
        self.sim_only = sim
        self.dt_gripper = None
        try:
            self.connect_to_rdk()
        except Exception as error:
            # handle the exception
            print("No installation of RoboDK was found in the system:", error) # An exception occurred: division by zero


    def sim_to_real(self):
        '''
        Allows movements on the real robot.\n
        Returns:\n
        \tbool: The status of "simulation only" flag
        '''
        self.sim_only = False
        if self.RDK.RunMode() == 1:
            self.RDK.setRunMode(6)
        warnings.warn("Warning. Activating real mode.")
        return not self.sim_only

    def real_to_sim(self):
        '''
        Denies movements on the real robot. The movements will be seen only in simulation\n
        Returns:\n
        \tbool: The status of "simulation only" flag
        '''
        self.sim_only = True
        if self.RDK.RunMode() > 1:
            self.RDK.setRunMode(1)
        warnings.warn("Warning. Deactivating real mode.")
        return self.sim_only

    def status(self):
        '''
        Used to consult the status of RoboDK RunMode.\n
        Returns:\n
        \tint: The status of RoboDK RunMode: 1: Simulation, 6: Run on robot
        '''
        return self.RDK.RunMode()

    def connect_to_rdk(self):
        '''
        Connect to the RoboDK API.\n
        Returns:\n
        \tbool: The status of the connection. True: succesful, False: otherwise.
        '''
        ###  ###
        self.RDK = Robolink(robodk_ip=self.rdk_, port=20500)
        return self.RDK.Connect()

    def get_rdk_item(self, name:str):
        '''
        Get an object of an item present in current RoboDK station.

        Returns:\n
        \trdk.Item object: This object is the link between an item in the station and Python.
        '''
        return self.RDK.Item(name)

    def check_rdk_item(self, name:str):
        '''
        Check if an item is present in current RoboDK station.

        Returns:\n
        \tbool: True for present, False if the item its not found.
        '''
        item = self.get_rdk_item(name)
        if item.Valid():
            return True
        else:
            return False

    def import_object_from_file(self, name:str, filename:str, parent:Robolink.Item = 0, position:list = [0, 0, 0], rgba:list = None):
        frame = self.RDK.AddFrame(name, parent)
        frame.setPose(transl(*position))
        new_object = self.RDK.AddFile(filename=filename, parent=frame)
        if rgba is not None:
            new_object.setColor(rgba)
        return frame
    
    def move_object_relative(self, object:Robolink.Item, position:list):
        original_pose = Pose_2_UR(object.Pose())
        new_position = np.add(original_pose[:3], position)
        object.setPose(transl(new_position.tolist()))

    def connect_to_robot(self, name:str, type:str, ip:str, attachements: list = [], **args):
        '''
        This function creates an instance of a robot object. If the sim_only flag is deactivated the
        function makes a connection with the real robot

        Arguments:\n
            name: A string identifier for our robot object.
            type: For now, only "UR" is allowed. Planning to increase the type of robots in future.\n
            ip: A string containing the ip of the physical robot.
            attachements: a list with all of the tools attached to the physical robot. Most used: camera and gripper.
            other arguments: like home_position or the flag for using modified D-H parameters for IK calculation
        '''
        ur_robots = {'UR10e':'ur10e', 'UR5e': 'ur5e'}
        success = False
        tries = 0
        max_tries = 3
        if not self.check_rdk_item(name):
            warnings.warn("Warning. There is no robot with that name in station")
            return False

       # Connect to the robot using default connetion parameters
        if self.sim_only != True:
            if type == 'UR':
                if 'grip_cam' in attachements:
                    self.robot = rf.UR_robot(name, ur_robots[name], linkto=self.get_rdk_item(name), attach=['grip_cam', 'wrist_cam'], **args)
                    self.Tgrip = rf.posVec2transMat(self.robot.attached['grip_cam'])
                    self.Twcam = rf.posVec2transMat(self.robot.attached['wrist_cam'])
                    self.gripper = ur_manipulators_helpers.robot_utils.rbu_package.robotiq_gripper.RobotiqGripper()
                    self.gripper.connect(ip, 63352)
                    if not self.gripper.is_active():
                        print("Activating gripper...")
                        self.gripper.activate()
                    else:
                        self.gripper.auto_calibrate()
                    self.gMax_Pos = self.gripper.get_max_position()
                    self.gMin_Pos = self.gripper.get_min_position()
                    self.dt_gripper = self.get_rdk_item('RobotiQ 2F85')
                    if self.dt_gripper.Type() != 2:
                        warnings.warn("Warning. There is no gripper in station.")

                elif 'gripper' in attachements:
                    self.robot = rf.UR_robot(name, ur_robots[name], linkto=self.get_rdk_item(name), attach=['gripper'])
                    self.Tgrip = rf.posVec2transMat(self.robot.attached['gripper'])
                    self.gripper = ur_manipulators_helpers.robot_utils.rbu_package.robotiq_gripper.RobotiqGripper()
                    self.gripper.connect(ip, 63352)
                    if not self.gripper.is_active():
                        print("Activating gripper...")
                        self.gripper.activate()
                    else:
                        self.gripper.auto_calibrate()
                    self.gMax_Pos = self.gripper.get_max_position()
                    self.gMin_Pos = self.gripper.get_min_position()
                    self.dt_gripper = self.get_rdk_item('RobotiQ 2F85')
                    if self.dt_gripper.Type() != 2:
                        warnings.warn("Warning. There is no gripper in station.")

                elif 'wrist_cam' in attachements:
                    self.robot = rf.UR_robot(name, ur_robots[name], linkto=self.get_rdk_item(name), attach=['wrist_cam'])
                    self.Twcam = rf.posVec2transMat(self.robot.attached['wrist_cam'])


                else:
                    self.robot = rf.UR_robot(name, ur_robots[name], linkto=self.get_rdk_item(name), attach=[])
                    self.Tgrip = rf.posVec2transMat(np.array([0., 0., 0., 0., 0., 0.]))

            else:
                warnings.warn("Warning. The desired robot has not been implemented")
                return False

            while not success or tries < max_tries:
                success = self.robot.Item.Connect(robot_ip=ip)
                tries += 1

            status, status_msg = self.robot.Item.ConnectedState()

            if status != ROBOTCOM_READY:
            # Stop if the connection did not succeed
                raise Exception("Failed to connect: " + status_msg)
            else:
                return success

        else:
            warnings.warn("Warning. For safety, it is necessary to activate real mode first to establish a connection with the robot.")
            self.robot = rf.UR_robot(name, ur_robots[name], linkto=self.get_rdk_item(name), attach=['grip_cam', 'wrist_cam'])
            self.Tgrip = rf.posVec2transMat(self.robot.attached['grip_cam'])
            self.Twcam = rf.posVec2transMat(self.robot.attached['wrist_cam'])
            return False

    def gripper_log_info(self):
        '''
        Function to print basic information about the gripper
        '''
        print(f"Pos: {str(self.gripper.get_current_position()): >3}  "
            f"Open: {self.gripper.is_open(): <2}  "
            f"Closed: {self.gripper.is_closed(): <2}  ")

    def dt_gripper_position(self, current_position) -> float:
        '''
        Function to control the position of the simulated gripper.

        Returns:
            List: with the current position of the simulated gripper.
        '''
        dt_pos = [0]
        dt_pos[0] = 85 - (current_position / self.gMax_Pos) * 85
        return dt_pos

    def select_gripper(self, name:str) -> bool:
        '''
        Function to enable the control of the simulated gripper.

        Returns:
            bool: True if the selected gripper exists, False otherwise.
        '''

        self.dt_gripper = self.get_rdk_item(name)
        if self.dt_gripper.Valid():
            self.gMax_Pos = int(np.array(self.dt_gripper.JointLimits()[1]).squeeze())
            self.gMin_Pos = int(np.array(self.dt_gripper.JointLimits()[0]).squeeze())
            return True
        else:
            warnings.warn("Warning. The gripper doesn't exists in station")
            return False


    def gripper_full_open(self):
        '''
        Function to fully open the gripper. Can open both grippers the physical and the simulated ones.


        '''
        if self.sim_only:
            if self.dt_gripper is not None:
                position = 85 if self.gMax_Pos > 85 else 85
                self.dt_gripper.MoveJ([position])
                # time.sleep(0.5)
            else:
               warnings.warn("Warning. There is no gripper defined.")
        else:
            if self.RDK.RunMode() > 1:
                self.RDK.setRunMode(1)
            time.sleep(0.05)
            self.gripper.move_and_wait_for_pos(0, 255, 255)
            self.gripper_log_info()
            dt_pos = self.dt_gripper_position(self.gripper.get_current_position())
            self.dt_gripper.MoveJ(dt_pos)
            # time.sleep(0.5)
            #tool_frame.DetachAll()
            if self.sim_only != True:
                self.RDK.setRunMode(6)
            time.sleep(0.05)

    def gripper_full_close(self):
        '''
        Function to fully close the gripper. Can close both grippers the physical and the simulated ones.
        '''
        if self.sim_only:
            if self.dt_gripper is not None:
                position = 0 if self.gMin_Pos > 0 else 0
                self.dt_gripper.MoveJ([position])
                # time.sleep(0.5)
            else:
               warnings.warn("Warning. There is no gripper defined.")
        else:
            if self.RDK.RunMode() > 1:
                self.RDK.setRunMode(1)
            time.sleep(0.05)
            # self.gripper.move_and_wait_for_pos(int(self.gMax_Pos/2), 255, 191)
            # self.gripper_log_info()
            # dt_pos = self.dt_gripper_position(self.gripper.get_current_position())
            # self.dt_gripper.MoveJ(dt_pos)
            #input("Press 'Enter' to continue...")
            self.gripper.move_and_wait_for_pos(self.gMax_Pos, 255, 191)
            self.gripper_log_info()
            dt_pos = self.dt_gripper_position(self.gripper.get_current_position())
            self.dt_gripper.MoveJ(dt_pos)
            #input("Press 'Enter' to continue...")
            # time.sleep(0.5)
            #tool_frame.AttachClosest()
            if self.sim_only != True:
                self.RDK.setRunMode(6)
            time.sleep(0.05)

    def robot_home(self, speed:int=75):
        '''
        The purpose of this function is to move the robot to a defined robot home position.
        '''
        self.robot_move_cartesian({'name': 'Home', 'joints': list(self.robot.home_pos)}, speed=speed)

    def robot_current_pose(self, **args) -> tuple:
        '''
        The purpose of this function is to obtain the current pose of the robot.

        Returns:
            tuple: With the transformation matrix of the pose, and the position and rotation vectors as well.
        '''
        manipulator = self.robot.Item
        joints = manipulator.Joints().list()

        T, tvec, rvec = self.robot_fkinematics(joints, **args)

        return (T, tvec, rvec)

    def robot_move_linear(self, target:dict, speed=200):
        '''
        Implements robot linear movements.

        Arguments:
            target: dictionary which contains the name of the position, and the desired joints list.
        '''
        manipulator = self.robot.Item
        manipulator.setSpeed(speed)
        # print('\nRobot joint movement to "{}"... '.format(target['name']))
        manipulator.MoveL(target['joints'])
        # T, _, _, _, _ = rf.ur_fkine(self.robot.parameters, np.asarray(target['joints']))
        # print(T)
        manipulator.Pause(100)

    def robot_move_cartesian(self, target:dict, speed=50):
        '''
        Implements robot joint movements.

        Arguments:
            target: dictionary which contains the name of the position, and the desired joints list
        '''
        manipulator = self.robot.Item
        manipulator.setSpeed(50, speed)
        # print('\nRobot joint movement to "{}"... '.format(target['name']))
        manipulator.MoveJ(target['joints'])
        # T, _, _, _, _ = rf.ur_fkine(self.robot.parameters, np.asarray(target['joints']))
        # print(T)
        manipulator.Pause(100)

    def robot_get_joints(self, objective:np.array, reference:str = 'gripper', trans_mat:np.array=np.eye(4),**args) -> list:
        '''
        Calculates the joint values of a reference position using geometrical IK.

        Arguments:
            objective: A numpy array containing the desired position and orientation of the target frame.
            reference: A string with the name of the reference frame used for IK calculations. There are
            two predifined frames: camera and gripper, if other value is needed or passed its necesary to
            indicate the transformation matrix using the argument 'trans_mat'.
            trans_mat: A numpy array containing the transformation matrix of the reference frame. It defaults
            to the unitary matrix so if ommited the IK calculation will be with respect to the robot 'flange'
            frame.
            other arguments: like the number of configuration returned by the IK or if you want the results in
            radians instead of degrees. Check robot_ikinematics function for more details.

        Returns:
            list: with the calculated joint positions.
        '''
        if objective.size == 16:
            T_ref = objective
        else:
            T_ref =  rf.posVec2transMat(objective)
        if reference == 'camera':
            T = T_ref@np.linalg.inv(self.Twcam)
        elif reference == 'gripper':
            T = T_ref@np.linalg.inv(self.Tgrip)
        else:
            T = T_ref@np.linalg.inv(trans_mat)

        return self.robot_ikinematics(T, **args)

    def robot_fkinematics(self, joints:np.array, reference:str='gripper', trans_mat:np.array=np.eye(4), **args) -> tuple:
        '''
        Calculates the foward kinematics with respect to a reference frame.

        Arguments:
            joints: A numpy array containing the joint values to be used in calculation.
            reference: A string with the name of the reference frame used for FK calculations. There are
            two predifined frames: camera and gripper, if other value is needed or passed its necesary to
            indicate the transformation matrix using the argument 'trans_mat'.
            trans_mat: A numpy array containing the transformation matrix of the reference frame. It defaults
            to the unitary matrix so if ommited the FK calculation will be with respect to the robot 'flange'
            frame.
            other arguments: a bool argument indicating if the calculation will be with respect to D-H parameters
            or with respect to the modified D-H parameters

        Returns:
            tuple: With the transformation matrix of the desired pose, and the position and rotation vectors as well.
        '''

        T, _, _, _, _ = rf.ur_fkine(self.robot.parameters, joints, **args)
        if reference == 'gripper':
            new_T = T @ self.Tgrip
        elif reference == 'camera':
            new_T = T @ self.Twcam
        else:
            new_T = T @ trans_mat

        R = new_T[0:3,0:3]
        tvec = new_T[0:3,3]
        rvec = rf.rotMat2rotVec_sp(R).flatten()
        return new_T, tvec, rvec

    def robot_ikinematics(self, trans_mat:np.ndarray, config=0, rad2deg=True):
        '''
        Calculates the joint values of a robot using geometrical IK.

        Arguments:
            trans_mat: A numpy array containing the transformation matrix of the desired robot pose.
            config: An integer number used to choose the configuration returned by the IK. Defaults to 0.
            Use the following as reference:
                    'elbow_L/R' | 'wrist_U/D' | 'elbow_U/D'
                0:      'left'     'up'            'up'
                1:      'left'     'up'            'down'
                2:      'left'     'down'          'up'
                3:      'left'     'down'          'down'
                4:      'right'    'up'            'up'
                5:      'right'    'up'            'down'
                6:      'right'    'down'          'up'
                7:      'right'    'down'          'down'
            rad2deg: A bool flag if you want the results in radians instead of degrees. Defaults to True.

        Returns:
            list: with the calculated joint positions.
        '''
        if trans_mat.size != 16:
            warnings.warn('The array has not the correct dimensions. Expected (4, 4) array.')
            return []
        T = trans_mat.squeeze()
        cols = ['t1', 't2', 't3', 't4', 't5', 't6']
        ikine = rf.ur_ikine(self.robot.parameters, T)
        ikine[cols] = ikine[cols].apply(pd.to_numeric)
        if rad2deg:
            ikine[cols] = ikine[cols].apply(np.rad2deg)

        return ikine.loc[config, cols].to_list()

    def robot_wristcam_target_approach(self, targets:list, gripper_orientation:np.array=np.array([0., np.pi, 0.]), scale=8):
        '''
        Deprecated function. Try to avoid using it.
        '''
        warnings.warn("Deprecation warning. This function will be removed soon, be careful when using it due to some errors.")
        joints = []
        orientation_base = gripper_orientation
        for target in targets:
            _, tvec, _ = self.robot_current_pose(tool_frame='camera')
            corrected_pos = (target[0] + tvec) * np.array([-1., 1., 1.*scale])
            P = np.hstack((corrected_pos, orientation_base))
            T = rf.posVec2transMat(P)
            T_p = T@np.linalg.inv(self.Tgrip)
            Tz = rf.trans_mat(rf.rotz(np.deg2rad(target[1])), np.array([0., 0., 0.], ndmin=2).T)
            T_des = T_p@Tz
            joints.append(self.robot_ikinematics(T_des))

        return joints

    def robot_target_approach(self, targets:list, safe_distance:float=0.15, object_orientation:np.array=np.array([0., 0., 0.]), **args):
        '''
        Deprecated function. Try to avoid using it.
        '''
        warnings.warn("Deprecation warning. This function will be removed soon, you can find a better solution in 'robot_target_and_approach' function")

        joints = []
        orientation = object_orientation
        for target in targets:
            B2F, _, _ = self.robot_current_pose(**args)
            T_o_c = rf.trans_mat(rf.rotVec2rotMat(orientation), target[0])
            # Pending to define how to allow the user to establish the approach distance
            Tapp = rf.trans_mat(np.eye(3), np.array([0., 0., -1.* safe_distance], ndmin=2).T)
            O = B2F@T_o_c
            T_p = O@np.linalg.inv(self.Tgrip)
            Tz = rf.trans_mat(rf.rotz(np.deg2rad(target[1])), np.array([0., 0., 0.], ndmin=2).T)
            T_des = T_p@Tz@Tapp
            joints.append(self.robot_ikinematics(T_des))

        return joints

    def robot_target(self, targets:list, **args):
        joints = []
        positions = []
        for target in targets:
            B2F, _, _ = self.robot_current_pose(**args)
            T_o_c = rf.posVec2transMat(target)
            O = B2F@T_o_c
            # T_p = O@np.linalg.inv(self.Tgrip)
            positions.append(rf.transMat2posVec(O))
            joints.append(self.robot_get_joints(O))

        return joints, positions

    def robot_target_and_approach(self, targets:list, safe_distance:float=0.15, **args):
        joints = []
        for target in targets:
            B2F, _, _ = self.robot_current_pose(**args)
            T_o_c = rf.posVec2transMat(target)
            Tapp = rf.trans_mat(np.eye(3), np.array([0., 0., -1.* safe_distance], ndmin=2).T)
            O = B2F@T_o_c
            T_p = O@np.linalg.inv(self.Tgrip)
            # Tz = rf.trans_mat(rf.rotz(np.deg2rad(target[1])), np.array([0., 0., 0.], ndmin=2).T)
            T_des = T_p@Tapp
            joints.append(self.robot_ikinematics(T_des))

        return joints

    def robot_reach_target(self, task:str, safe_distance:float=0.090):
        '''
            s
        '''
        if task == 'pick':
            if self.gripper.is_closed():
                self.gripper_full_open()
            T, tvec, _ = self.robot_current_pose(reference='gripper')
            Tz = rf.trans_mat(rf.rotz(0), np.array([0., 0., safe_distance], ndmin=2).T)
            T_des = T@Tz@np.linalg.inv(self.Tgrip)
            joints = self.robot_ikinematics(T_des)
            self.robot_move_linear({'name': 'to the object', 'joints': joints}, speed=50)

            self.gripper_full_close()

            T, tvec, _ = self.robot_current_pose(reference='gripper')
            Tz = rf.trans_mat(rf.rotz(0), np.array([0., 0., -1.*safe_distance], ndmin=2).T)
            T_des = T@Tz@np.linalg.inv(self.Tgrip)
            joints = self.robot_ikinematics(T_des)
            self.robot_move_linear({'name': 'to approach', 'joints': joints}, speed=50)

        elif task == 'place':
            T, tvec, _ = self.robot_current_pose(reference='gripper')
            Tz = rf.trans_mat(rf.rotz(0), np.array([0., 0., safe_distance], ndmin=2).T)
            T_des = T@Tz@np.linalg.inv(self.Tgrip)
            joints = self.robot_ikinematics(T_des)
            self.robot_move_linear({'name': 'to the object', 'joints': joints}, speed=50)

            self.gripper_full_open()

            T, tvec, _ = self.robot_current_pose(reference='gripper')
            Tz = rf.trans_mat(rf.rotz(0), np.array([0., 0., -1.*safe_distance], ndmin=2).T)
            T_des = T@Tz@np.linalg.inv(self.Tgrip)
            joints = self.robot_ikinematics(T_des)
            self.robot_move_linear({'name': 'to approach', 'joints': joints}, speed=50)

        else:
            warnings.warn('The operation selected is not recognized. It will be ommited')

        return True

    def robot_rotate_tcp(self, angle:float = 0., direction:str = 'z', reference:str = 'gripper')-> np.array:
        '''
        This function implements rotations in three different frames of the robot. Gripper (tool frame),
        Camera (camera_frame), or Flange (robot flange frame).

        Arguments:
            angle: Float value always in degrees
            direction: string value 'x', 'y', 'z' with respect to tcp frame.
                       Default direction is 'z'
            reference: string value indicating which tcp to use. Defaults to 'gripper'

        Returns:
            pose_vector = A np.array containing the calculated pose vector.
        '''
        J = np.asarray(self.robot.Item.Joints()).squeeze()
        T, _, _, _, _ = rf.ur_fkine(self.robot.parameters, J)
        if reference == 'camera':
            T = T@self.Twcam
        elif reference == 'gripper':
            T = T@self.Tgrip

        if direction == 'x':
            Tx = rf.trans_mat(rf.rotx(np.deg2rad(angle)), np.array([0., 0., 0.], ndmin=2).T)
            return rf.transMat2posVec(T@Tx)
        if direction == 'y':
            Ty = rf.trans_mat(rf.roty(np.deg2rad(angle)), np.array([0., 0., 0.], ndmin=2).T)
            return rf.transMat2posVec(T@Ty)
        if direction == 'z':
            Tz = rf.trans_mat(rf.rotz(np.deg2rad(angle)), np.array([0., 0., 0.], ndmin=2).T)
            return rf.transMat2posVec(T@Tz)

    def set_camera(self, name:str, params:str='SIZE=640x480') -> bool:
        '''
        This function allows to link and modify the parameters of a camera object in RoboDK.\n
        Arguments:\n
            \tname: Camera item name in current station.\n
            \tparams: Parameters to pass to camera item. # For more options, see https://robodk.com/doc/en/PythonAPI/robodk.html#robodk.robolink.Robolink.Cam2D_Add\n
        Returns:\n
            \tbool: True if the cam item is valid, or false if it is invalid.

        '''
        self.cam_item = self.get_rdk_item(name)
        if self.cam_item.Valid():
            self.cam_item.setParam(params)
            self.cam_item.setParam('Open', 1)

            return self.cam_item.Valid()
        else:
            warnings.warn("Warning. There is no camera item with that name defined.")
            return self.cam_item.Valid()

    def camera_get_image(self):
        #----------------------------------
        # Get the camera image, from disk
        # Insert your code here
        img_png = None
        import tempfile
        with tempfile.TemporaryDirectory(prefix='rdk_') as td:
            tf = td + '/temp.png'
            if self.RDK.Cam2D_Snapshot(tf, self.cam_item) == 1:
                img_png = cv2.imread(tf)
        return img_png



class RTDE_Connection:
    def __init__(self,
                 robot_ip: str = '172.23.254.231',
                 sim: bool = True) -> None:
        '''
        RDK Connection class implements multiple functionalities to make easier
        the programming of UR robots using RoboDK Python API. By default the
        construction of the object tries to establish the connection with RoboDK
        software.\n\n
        Arguments:\n
            \trdk_ip: IP address of the computer hosting the RoboDK interface.
            \tsim: A bool parameter to indicate if the movements will be done on
            simulation or in the real robot.

        '''
        self.ip = robot_ip
        self.robot = None
        self.sim_only = sim
        self.gripper = None
        self.reconnect = False
        ### Warning i wont be able to generalize this one should revist the constructor ###
        # self.robot_sim = None
        # self.env = swift.Swift()
        # try:
        #     self.env.launch(realtime=True, headless=True)
        # except Exception as e:
        #     print(e)
        ###################################################################################



    def start_connection(self, rtde_frequency:float = 500., port:int = 50002):
        '''
        Allows movements on the real robot.\n
        Returns:\n
        \tbool: The status of "simulation only" flag
        '''
        # self.sim_only = False
        if 'realtime' in platform.uname()[2]:
            # Parameters
            self.dt = 1.0/rtde_frequency  # 2ms
            flags = RTDEControl.FLAG_USE_EXT_UR_CAP
            ur_cap_port = 50002

            lookahead_time = 0.1
            gain = 600

            # ur_rtde realtime priorities
            rt_receive_priority = 90
            rt_control_priority = 85

            self.rtde_r = rtde_receive.RTDEReceiveInterface(self.ip, rtde_frequency, [], True, False, rt_receive_priority)
            self.rtde_c = RTDEControl(self.ip, rtde_frequency, flags, ur_cap_port, rt_control_priority)

        else:
            self.connection_frequency = rtde_frequency
            self.rtde_c = RTDEControl(self.ip, rtde_frequency, RTDEControl.FLAG_USE_EXT_UR_CAP, port)
            self.rtde_r = rtde_receive.RTDEReceiveInterface(self.ip)
            self.rtde_io = rtde_io.RTDEIOInterface(self.ip)

        self.reconnect = True
        warnings.warn("Warning. Activating real mode.")
        return self.rtde_c.isConnected()

    def stop_connection(self):
        if self.rtde_c.isConnected():
            self.rtde_c.disconnect()
        else:
            print("The connection to the robot control interface is closed.")

    def restart_connection(self, timeout:int = 60):
        timeout_ = timeout   # [seconds]
        if self.reconnect and not self.rtde_c.isConnected():
            self.rtde_c.reconnect()
            timeout_start = time.time()
            while time.time() < timeout_start + timeout_:
                if self.rtde_c.isConnected():
                    print("The robot has been reconnected successfully.")
                    break
                else:
                    time.sleep(0.1)
            print("The connection to the robot could not be established. The robot will remain disconnected.")
        elif not self.reconnect:
            print("The connection needs to be established first. Try start_connection method.")
        else:
            print("The connection has already been established. The robot is currently connected")

    def sim_to_real(self, status: bool =False):
        '''
        Denies movements on the real robot. The movements will be seen only in simulation\n
        Returns:\n
        \tbool: The status of "simulation only" flag
        '''
        self.sim_only = not status
        if self.gripper is None:
            self.gripper = ur_manipulators_helpers.robot_utils.rbu_package.robotiq_gripper.RobotiqGripper()
            self.gripper.connect(self.ip, 63352)
            if not self.gripper.is_active():
                print("Activating gripper...")
                self.gripper.activate()
            else:
                self.gripper.auto_calibrate()
            self.gMax_Pos = self.gripper.get_max_position()
            self.gMin_Pos = self.gripper.get_min_position()

        return self.rtde_c.isConnected()

    def status(self):
        '''
        Used to consult the status of RoboDK RunMode.\n
        Returns:\n
        \tint: The status of RoboDK RunMode: 1: Simulation, 6: Run on robot
        '''
        return self.rtde_c.getRobotStatus()

    def create_robot(self,
                         name:str,
                         attachements: list = [],
                         urdf_file:str = '/home/daniel/github/robodk/Rocket_assembly/urdf/robots/ur10e_2f85.urdf',
                         gripper_link:int = 10,
                         **args):
        '''
        This function creates an instance of a robot object. If the sim_only flag is deactivated the
        function makes a connection with the real robot

        Arguments:\n
            name: A string identifier for our robot object. Examples: 'UR10e', 'UR5e', etc.
            type: For now, only "UR" is allowed. Planning to increase the type of robots in future.\n
            attachements: a list with all of the tools attached to the physical robot. Most used: camera and gripper.
            urdf_file: A string with the path to the file of the robot urdf representation.
            other arguments: like home_position or the flag for using modified D-H parameters for IK calculation
        '''
        ur_robots = {'UR10e':'ur10e', 'UR5e': 'ur5e'}
        assert name in tuple(ur_robots.keys()), f"The robot: {name} hasn't been impĺemented in this class D:"

        # Connect to the robot using default connetion parameters
        if not self.sim_only:
            if 'grip_cam' in attachements:
                self.robot = rf.UR_robot(name, ur_robots[name], attach=['grip_cam', 'wrist_cam'], **args)
                self.Tgrip = rf.posVec2transMat(self.robot.attached['grip_cam'])
                self.Twcam = rf.posVec2transMat(self.robot.attached['wrist_cam'])
                self.gripper = ur_manipulators_helpers.robot_utils.rbu_package.robotiq_gripper.RobotiqGripper()
                self.gripper.connect(self.ip, 63352)
                if not self.gripper.is_active():
                    print("Activating gripper...")
                    self.gripper.activate()
                else:
                    self.gripper.auto_calibrate()
                self.gMax_Pos = self.gripper.get_max_position()
                self.gMin_Pos = self.gripper.get_min_position()

            elif 'gripper' in attachements:
                self.robot = rf.UR_robot(name, ur_robots[name], attach=['gripper', 'wrist_cam'], **args)
                self.Tgrip = rf.posVec2transMat(self.robot.attached['gripper'])
                self.Twcam = rf.posVec2transMat(self.robot.attached['wrist_cam']) #only used as reference point
                self.gripper = ur_manipulators_helpers.robot_utils.rbu_package.robotiq_gripper.RobotiqGripper()
                self.gripper.connect(self.ip, 63352)
                if not self.gripper.is_active():
                    print("Activating gripper...")
                    self.gripper.activate()
                else:
                    self.gripper.auto_calibrate()
                self.gMax_Pos = self.gripper.get_max_position()
                self.gMin_Pos = self.gripper.get_min_position()

            elif 'wrist_cam' in attachements:
                self.robot = rf.UR_robot(name, ur_robots[name], attach=['wrist_cam'], **args)
                self.Twcam = rf.posVec2transMat(self.robot.attached['wrist_cam'])


            else:
                self.robot = rf.UR_robot(name, ur_robots[name], **args)
                self.Tgrip = rf.posVec2transMat(np.array([0., 0., 0., 0., 0., 0.]))

        else:
            warnings.warn("Warning. For safety, it is necessary to activate real mode first to establish a connection with the robot.")
            self.robot = rf.UR_robot(name, ur_robots[name], attach=['grip_cam', 'wrist_cam'], **args)
            self.Tgrip = rf.posVec2transMat(self.robot.attached['grip_cam'])
            self.Twcam = rf.posVec2transMat(self.robot.attached['wrist_cam'])
            return False

        #self.robot_sim = Robot.URDF(urdf_file, gripper=gripper_link)
        # self.robot_sim.q = self.rtde_r.getActualQ()

    def gripper_log_info(self):
        '''
        Function to print basic information about the gripper
        '''
        print(f"Pos: {str(self.gripper.get_current_position()): >3}  "
            f"Open: {self.gripper.is_open(): <2}  "
            f"Closed: {self.gripper.is_closed(): <2}  ")

    def gripper_full_open(self):
        '''
        Function to fully open the gripper. Can open both grippers the physical and the simulated ones.


        '''
        if self.sim_only:
            warnings.warn("Warning. There is no gripper connected yet.")

        else:
            self.gripper.move_and_wait_for_pos(self.gMin_Pos, 255, 255)
            self.gripper_log_info()

    def gripper_full_close(self):
        '''
        Function to fully close the gripper. Can close both grippers the physical and the simulated ones.
        '''
        if self.sim_only:
            warnings.warn("Warning. There is no gripper connected yet.")

        else:
            self.gripper.move_and_wait_for_pos(self.gMax_Pos, 255, 255)
            self.gripper_log_info()

    def robot_teach_trajectory(self, xyzrpy:list= [1,1,1,0,0,0],wait_time:int=5, **kwargs):
        if not self.rtde_c.isConnected():
            raise Exception("An error occurred. The robot is currently disconnected.")

        init_speed = self.robot_read_TCP_speed()
        self.robot_teach(xyzrpy)
        while not self.robot_is_moving(init_speed):
            time.sleep(0.01)
        timestamp = np.array([self.get_timestamp()])
        x_robot = np.array([self.robot_read_pose()])
        dx_robot = np.array([self.robot_read_TCP_speed()])
        ddx_robot = np.array([self.robot_read_TCP_accelerometer()])
        idle_time = 0
        while idle_time < wait_time:
            if not self.robot_is_moving(dx_robot[-1]):
                idle_time = self.get_timestamp() - timestamp[-1]
                print(idle_time)
                time.sleep(0.01)
            else:
                timestamp = np.append(timestamp, np.array([self.get_timestamp()]), axis=0)
                x_robot = np.append(x_robot, np.array([self.robot_read_pose()]), axis=0)
                dx_robot = np.append(dx_robot, np.array([self.robot_read_TCP_speed()]), axis=0)
                ddx_robot = np.append(ddx_robot, np.array([self.robot_read_TCP_accelerometer()]), axis=0)
        self.robot_teach_end()
        timestamp = np.expand_dims(timestamp, axis=1)
        self.robot_save_trajectory(time=timestamp, pose=x_robot,
                                   velocity=dx_robot, acceleration=ddx_robot,
                                   **kwargs)

        T = (timestamp[-1] - timestamp[0]) * (self.connection_frequency/1000.)
        return [T, x_robot, dx_robot, ddx_robot]

    def robot_teach_shadow_trajectory(self, which_ur:str, ur_tool:str, xyzrpy:list= [1,1,1,0,0,0],wait_time:int=5, **kwargs):
        if not self.rtde_c.isConnected():
            raise Exception("An error occurred. The robot is currently disconnected.")

        init_speed = self.robot_read_TCP_speed()
        self.robot_teach(xyzrpy)
        while not self.robot_is_moving(init_speed):
            time.sleep(0.01)
        timestamp = np.array([self.get_timestamp()])
        q_robot = np.array([self.robot_read_joints()])
        x_robot = np.array([self.robot_read_pose()])
        dx_robot = np.array([self.robot_read_TCP_speed()])
        ddx_robot = np.array([self.robot_read_TCP_accelerometer()])
        x_shadow_robot = np.array([[self.robot_calculate_pose(which_ur=which_ur, ur_tool=ur_tool)]])
        idle_time = 0
        while idle_time < wait_time:
            if not self.robot_is_moving(dx_robot[-1]):
                idle_time = self.get_timestamp() - timestamp[-1]
                print(idle_time)
                time.sleep(0.01)
            else:
                timestamp = np.append(timestamp, np.array([self.get_timestamp()]), axis=0)
                q_robot = np.append(q_robot, np.array([self.robot_read_joints()]), axis=0)
                x_robot = np.append(x_robot, np.array([self.robot_read_pose()]), axis=0)
                dx_robot = np.append(dx_robot, np.array([self.robot_read_TCP_speed()]), axis=0)
                ddx_robot = np.append(ddx_robot, np.array([self.robot_read_TCP_accelerometer()]), axis=0)
                x_shadow_robot = np.append(x_shadow_robot, np.array([self.robot_calculate_pose(which_ur=which_ur, ur_tool=ur_tool)]))
        self.robot_teach_end()
        timestamp = np.expand_dims(timestamp, axis=1)
        self.robot_save_trajectory(time=timestamp, pose=x_robot,
                                   velocity=dx_robot, acceleration=ddx_robot, q=q_robot,
                                   shadow_pose=x_shadow_robot,
                                   **kwargs)

        T = (timestamp[-1] - timestamp[0]) * (self.connection_frequency/1000.)
        return [T, x_robot, dx_robot, ddx_robot, x_shadow_robot]

    def robot_save_trajectory(self, time, pose, velocity, acceleration, q:np.array= None, shadow_pose:np.array= None, obstacles:list = [], eul_rep:str='zyx', dir:str='', fname:str=''):
        M = np.hstack((time, pose, velocity, acceleration))
        df_columns = ['Time', 'TCP_pose_1', 'TCP_pose_2', 'TCP_pose_3',
                        'TCP_pose_4', 'TCP_pose_5', 'TCP_pose_6', 'TCP_speed_1',
                        'TCP_speed_2', 'TCP_speed_3', 'TCP_speed_4', 'TCP_spped_5',
                        'TCP_speed_6', 'TCP_accel_1', 'TCP_accel_2', 'TCP_accel_3']
        if q is not None:
            M = np.hstack((M, q))
            df_columns.extend(['ActualQ_1', 'ActualQ_2', 'ActualQ_3', 'ActualQ_4', 'ActualQ_5', 'ActualQ_6'])

            # M = np.hstack((M, np.hstack((q, shadow_pose))))
            # df_columns.extend(['ActualQ_1', 'ActualQ_2', 'ActualQ_3', 'ActualQ_4', 'ActualQ_5', 'ActualQ_6',
                            #    'TCP_shadow_1', 'TCP_shadow_2', 'TCP_shadow_3', 'TCP_shadow_4', 'TCP_shadow_5', 'TCP_shadow_6'])
            M = np.hstack((M, shadow_pose.reshape((int(shadow_pose.size/6), 6))))
            df_columns.extend(['TCP_shadow_1', 'TCP_shadow_2', 'TCP_shadow_3', 'TCP_shadow_4', 'TCP_shadow_5', 'TCP_shadow_6'])

        if len(obstacles) > 0:
            for n, obst in enumerate(obstacles):
                o_rot = np.full_like(acceleration, R.from_matrix(obst.R).as_euler(eul_rep, degrees=True))
                o_center = np.full_like(acceleration, obst.center)
                o_axes = np.full_like(acceleration, obst.axis)
                o_shape = np.full((len(time), 2), obst.coeffs)
                M = np.hstack((M, np.hstack((o_center, o_rot, o_axes, o_shape))))
                df_columns.extend(['obst_'+str(n)+'_pose_1', 'obst_'+str(n)+'_pose_2',
                                    'obst_'+str(n)+'_pose_3', 'obst_'+str(n)+'_pose_4',
                                    'obst_'+str(n)+'_pose_5', 'obst_'+str(n)+'_pose_6',
                                    'obst_'+str(n)+'_axis_1', 'obst_'+str(n)+'_axis_2',
                                    'obst_'+str(n)+'_axis_3', 'obst_'+str(n)+'_coeff_1',
                                    'obst_'+str(n)+'_coeff_2'])
            # obst.center, obst.axis

        print(len(df_columns), df_columns)
        df = pd.DataFrame(M, columns=df_columns)
        datestring = datetime.datetime.now().strftime("_%Y-%m-%d_%H%M%S")
        home = os.path.expanduser("~")
        # path = os.path.join(home, dir, fname + datestring +'.csv')
        path = os.path.join(home, dir, fname + datestring +'.csv')
        df.to_csv(path, index=False)
        return df.head(5)

    def robot_home(self, speed:float=1.05):
        '''
        The purpose of this function is to move the robot to a defined robot home position.
        '''
        if not self.rtde_c.isConnected():
            raise Exception("An error occurred. The robot is currently disconnected.")
        self.robot_move_cartesian(target_joints=np.deg2rad(self.robot.home_pos), speed=speed)

    def robot_read_joints(self, rad2deg:bool =False) -> list:
        '''
        The purpose of this function is to obtain the current pose of the robot.

        Returns:
            tuple: With the transformation matrix of the pose, and the position and rotation vectors as well.
        '''
        if not rad2deg:
            return self.rtde_r.getActualQ()
        else:
            return np.rad2deg(self.rtde_r.getActualQ())

    def robot_read_pose(self) -> list:
        return self.rtde_r.getActualTCPPose()

    def robot_read_TCP_speed(self) -> list:
        return self.rtde_r.getActualTCPSpeed()

    def robot_read_TCP_accelerometer(self) -> list:
        return self.rtde_r.getActualToolAccelerometer()

    def robot_calculate_pose(self, which_ur:str='current', ur_tool:str='grip_cam', **args) -> np.ndarray:
        '''
        The purpose of this function is to obtain the current pose of the robot.

        Returns:
            tuple: With the transformation matrix of the pose, and the position and rotation vectors as well.
        '''
        joints = self.robot_read_joints(rad2deg=True)

        return self.robot_fkinematics(joints, which_ur=which_ur, ur_tool=ur_tool, **args)


    def robot_teach(self, constraints: list = [0,0,0,0,0,0]) -> bool:
        if not self.rtde_c.isConnected():
            raise Exception("An error occurred. The robot is currently disconnected.")
        return self.rtde_c.freedriveMode(constraints)

    def start_recording_file(self, file):
        return self.rtde_r.startFileRecording(file)

    def stop_recording_file(self):
        return self.rtde_r.stopFileRecording()

    def get_timestamp(self):
        return self.rtde_r.getTimestamp()

    def robot_is_moving(self, prev_speed):
        current_speed = self.rtde_r.getActualTCPSpeed()#[:3]
        return not np.allclose(current_speed, prev_speed, rtol=1e-2, atol=1e-3)

    def robot_teach_end(self) -> bool:
        if not self.rtde_c.isConnected():
            raise Exception("An error occurred. The robot is currently disconnected.")
        return self.rtde_c.endFreedriveMode()


    def robot_move_linear(self, target_pose:np.array, speed:float =0.2, acceleration:float =1.2):
        '''
        Implements robot linear movements.

        Arguments:
            target: dictionary which contains the name of the position, and the desired joints list.
        '''
        if not self.rtde_c.isConnected():
            raise Exception("An error occurred. The robot is currently disconnected.")
        if self.status() > 3:
            self.robot_teach_end()
        # print('\nRobot joint movement to "{}"... '.format(target['name']))
        self.rtde_c.moveL(target_pose, speed, acceleration)
        # T, _, _, _, _ = rf.ur_fkine(self.robot.parameters, np.asarray(target['joints']))
        # print(T)

    def robot_moveL_joints(self, target_pose:np.array, speed:float =0.2, acceleration:float =1.2):
        '''
        Implements robot linear movements.

        Arguments:
            target: dictionary which contains the name of the position, and the desired joints list.
        '''
        if not self.rtde_c.isConnected():
            raise Exception("An error occurred. The robot is currently disconnected.")
        if self.status() > 3:
            self.robot_teach_end()
        # print('\nRobot joint movement to "{}"... '.format(target['name']))
        M = rf.posVec2transMat(target_pose)
        joints = self.robot_calculate_joints(M)

        self.robot_move_cartesian(target_joints=joints, speed=speed)
        # T, _, _, _, _ = rf.ur_fkine(self.robot.parameters, np.asarray(target['joints']))
        # print(T)

    def robot_move_linear_safe(self, target_pose:np.array, speed:float =0.2, acceleration:float =1.2):
        '''
        Implements robot linear movements. Still unsafe.

        Arguments:
            target: dictionary which contains the name of the position, and the desired joints list.
        '''
        warnings.warn("Still an unsafe function. Be careful")
        if not self.rtde_c.isConnected():
            raise Exception("An error occurred. The robot is currently disconnected.")
        if self.status() > 3:
            self.robot_teach_end()

        self.rtde_c.moveL(target_pose, speed, acceleration, True)
        self.rtde_c.startContactDetection()
        contact_detected = self.rtde_c.readContactDetection()
        if contact_detected:
            return contact_detected
        else:
            return self.rtde_c.stopContactDetection()

    def robot_move_servoL(self, target_pose:np.array, speed:float =0.2, acceleration:float =1.2,
                           dt = 0.001, lookahead_time = 0.1, gain = 300.):
        '''
        Implements robot linear movements. Still unsafe.

        Arguments:
            target: dictionary which contains the name of the position, and the desired joints list.
        '''
        if not self.rtde_c.isConnected():
            raise Exception("An error occurred. The robot is currently disconnected.")
        if self.status() > 3:
            self.robot_teach_end()

        t_start = self.rtde_c.initPeriod()
        self.rtde_c.servoL(target_pose, speed, acceleration, dt, lookahead_time, gain)
        self.rtde_c.waitPeriod(t_start)

    def robot_move_servoL_joints(self, target_pose:np.array, speed:float =0.2, acceleration:float =1.2,
                           dt = 0.001, lookahead_time = 0.1, gain = 300.):
        '''
        Implements robot linear movements. Still unsafe.

        Arguments:
            target: dictionary which contains the name of the position, and the desired joints list.
        '''
        if not self.rtde_c.isConnected():
            raise Exception("An error occurred. The robot is currently disconnected.")
        if self.status() > 3:
            self.robot_teach_end()

        M = rf.posVec2transMat(target_pose)
        joints = self.robot_calculate_joints(M)

        t_start = self.rtde_c.initPeriod()
        self.rtde_c.servoJ(joints, speed, acceleration, dt, lookahead_time, gain)
        self.rtde_c.waitPeriod(t_start)

    def robot_servo_stop(self):
        if not self.rtde_c.isConnected():
            raise Exception("An error occurred. The robot is currently disconnected.")
        self.rtde_c.servoStop()

    def robot_move_cartesian(self, target_joints:np.array, speed: float=1.05, acceleration:float =1.2):
        '''
        Implements robot joint movements.

        Arguments:
            target: dictionary which contains the name of the position, and the desired joints list
        '''
        if not self.rtde_c.isConnected():
            raise Exception("An error occurred. The robot is currently disconnected.")
        if self.status() > 3:
            self.robot_teach_end()
        # print('\nRobot joint movement to "{}"... '.format(target['name']))
        self.rtde_c.moveJ(target_joints, speed, acceleration)
        # T, _, _, _, _ = rf.ur_fkine(self.robot.parameters, np.asarray(target['joints']))
        # print(T)
        # manipulator.Pause(100)

    def robot_calculate_joints(self, objective:np.array, reference:str = 'gripper', trans_mat:np.array=np.eye(4),**args) -> list:
        '''
        Calculates the joint values of a reference position using geometrical IK.

        Arguments:
            objective: A numpy array containing the desired position and orientation of the target frame.
            reference: A string with the name of the reference frame used for IK calculations. There are
            two predifined frames: camera and gripper, if other value is needed or passed its necesary to
            indicate the transformation matrix using the argument 'trans_mat'.
            trans_mat: A numpy array containing the transformation matrix of the reference frame. It defaults
            to the unitary matrix so if ommited the IK calculation will be with respect to the robot 'flange'
            frame.
            other arguments: like the number of configuration returned by the IK or if you want the results in
            radians instead of degrees. Check robot_ikinematics function for more details.

        Returns:
            list: with the calculated joint positions.
        '''
        if objective.size == 16:
            T_ref = objective
        else:
            T_ref =  rf.posVec2transMat(objective)
        if reference == 'camera':
            T = T_ref@np.linalg.inv(self.Twcam)
        elif reference == 'gripper':
            T = T_ref@np.linalg.inv(self.Tgrip)
        else:
            T = T_ref@np.linalg.inv(trans_mat)

        return self.robot_ikinematics(T, **args)

    def robot_fkinematics(self, joints:np.array, reference:str='gripper', trans_mat:np.array=np.eye(4), which_ur:str='current', ur_tool:str='grip_cam',**args) -> tuple:
        '''
        Calculates the foward kinematics with respect to a reference frame.

        Arguments:
            joints: A numpy array containing the joint values to be used in calculation. Need to be in degrees.
            reference: A string with the name of the reference frame used for FK calculations. There are
            two predifined frames: camera and gripper, if other value is needed or passed its necesary to
            indicate the transformation matrix using the argument 'trans_mat'.
            trans_mat: A numpy array containing the transformation matrix of the reference frame. It defaults
            to the unitary matrix so if ommited the FK calculation will be with respect to the robot 'flange'
            frame.
            other arguments: a bool argument indicating if the calculation will be with respect to D-H parameters
            or with respect to the modified D-H parameters

        Returns:
            tuple: With the transformation matrix of the desired pose, and the position and rotation vectors as well.
        '''
        if which_ur == 'current':
            T, _, _, _, _ = rf.ur_fkine(self.robot.parameters, joints, **args)
            if reference == 'gripper':
                new_T = T @ self.Tgrip
            elif reference == 'camera':
                new_T = T @ self.Twcam
            else:
                new_T = T @ trans_mat

            # R = new_T[0:3,0:3]
            # tvec = new_T[0:3,3]
            # rvec = rf.rotMat2rotVec_sp(R).flatten()
        else:
            T, _, _, _, _ = rf.ur_fkine(rf.UR_robot.dhp[which_ur], joints, **args)
            new_T = T @ rf.posVec2transMat(rf.UR_robot.fix_frame[ur_tool])

        return rf.transMat2posVec(new_T)

    def robot_ikinematics(self, trans_mat:np.ndarray, config=0, rad2deg=False):
        '''
        Calculates the joint values of a robot using geometrical IK.

        Arguments:
            trans_mat: A numpy array containing the transformation matrix of the desired robot pose.
            config: An integer number used to choose the configuration returned by the IK. Defaults to 0.
            Use the following as reference:
                    'elbow_L/R' | 'wrist_U/D' | 'elbow_U/D'
                0:      'left'     'up'            'up'
                1:      'left'     'up'            'down'
                2:      'left'     'down'          'up'
                3:      'left'     'down'          'down'
                4:      'right'    'up'            'up'
                5:      'right'    'up'            'down'
                6:      'right'    'down'          'up'
                7:      'right'    'down'          'down'
            rad2deg: A bool flag if you want the results in radians instead of degrees. Defaults to True.

        Returns:
            list: with the calculated joint positions.
        '''
        if trans_mat.size != 16:
            warnings.warn('The array has not the correct dimensions. Expected (4, 4) array.')
            return []
        T = trans_mat.squeeze()
        cols = ['t1', 't2', 't3', 't4', 't5', 't6']
        ikine = rf.ur_ikine(self.robot.parameters, T)
        ikine[cols] = ikine[cols].apply(pd.to_numeric)
        if rad2deg:
            ikine[cols] = ikine[cols].apply(np.rad2deg)

        return ikine.loc[config, cols].to_list()

    def robot_target(self, targets:list, **args):
        joints = []
        positions = []
        for target in targets:
            vB2F = self.robot_calculate_pose(**args)
            T_B2F = rf.posVec2transMat(vB2F)
            T_o_c = rf.posVec2transMat(target)
            O = T_B2F@T_o_c
            # T_p = O@np.linalg.inv(self.Tgrip)
            positions.append(rf.transMat2posVec(O))
            joints.append(self.robot_ikinematics(O))

        return joints, positions

    def robot_target_and_approach(self, targets:list, safe_distance:float=0.15, pose:np.array = None,**args):
        joints = []
        positions = []
        for target in targets:
            if pose is None:
                vB2F = self.robot_calculate_pose(**args)
            else:
                vB2F = pose
            T_B2F = rf.posVec2transMat(vB2F)
            T_o_c = rf.posVec2transMat(target)
            Tapp = rf.trans_mat(np.eye(3), np.array([0., 0., -1.* safe_distance], ndmin=2).T)
            O = T_B2F@T_o_c
            T_p = O@np.linalg.inv(self.Tgrip)
            # Tz = rf.trans_mat(rf.rotz(np.deg2rad(target[1])), np.array([0., 0., 0.], ndmin=2).T)
            T_des = T_p@Tapp
            T_des_pos = O@Tapp

            positions.append(rf.transMat2posVec(T_des_pos))
            joints.append(self.robot_ikinematics(T_des))

        return joints, positions

    def robot_approach(self, targets:list, safe_distance:float=0.15, **args):
        joints = []
        positions = []
        for target in targets:
            T_B2O = rf.posVec2transMat(target)
            Tapp = rf.trans_mat(np.eye(3), np.array([0., 0., -1.* (safe_distance-target[2])], ndmin=2).T)
            # T_p = O@np.linalg.inv(self.Tgrip)
            # # Tz = rf.trans_mat(rf.rotz(np.deg2rad(target[1])), np.array([0., 0., 0.], ndmin=2).T)
            # T_des = T_p@Tapp
            T_des_pos = T_B2O@Tapp

            positions.append(rf.transMat2posVec(T_des_pos))
            joints.append(self.robot_calculate_joints(rf.transMat2posVec(T_des_pos)))

        return joints, positions

    def robot_reach_target(self, task:str, safe_distance:float=0.090, speed:float=0.05, **args):
        '''
            s
        '''
        if not self.rtde_c.isConnected():
            raise Exception("An error occurred. The robot is currently disconnected.")
        if task == 'pick':
            if not self.sim_only:
                if self.gripper.is_closed():
                    self.gripper_full_open()
            tvec = self.robot_calculate_pose(**args)
            T = rf.posVec2transMat(tvec)
            Tz = rf.trans_mat(rf.rotz(0), np.array([0., 0., 1.*safe_distance], ndmin=2).T)
            T_des = T@Tz#@np.linalg.inv(self.Tgrip)
            # joints = self.robot_ikinematics(T_des, **args)
            pV = rf.transMat2posVec(T_des)
            self.robot_move_linear(target_pose=pV, speed=speed)

            if not self.sim_only:
                self.gripper_full_close()

            tvec = self.robot_calculate_pose(**args)
            T = rf.posVec2transMat(tvec)
            Tz = rf.trans_mat(rf.rotz(0), np.array([0., 0., -1.*safe_distance], ndmin=2).T)
            T_des = T@Tz#@np.linalg.inv(self.Tgrip)
            pV = rf.transMat2posVec(T_des)
            self.robot_move_linear(target_pose=pV, speed=speed)

        elif task == 'place':
            tvec = self.robot_calculate_pose(reference='gripper')
            T = rf.posVec2transMat(tvec)
            Tz = rf.trans_mat(rf.rotz(0), np.array([0., 0., safe_distance], ndmin=2).T)
            T_des = T@Tz#@np.linalg.inv(self.Tgrip)
            pV = rf.transMat2posVec(T_des)
            self.robot_move_linear(target_pose=pV, speed=speed)

            if not self.sim_only:
                self.gripper_full_open()

            tvec = self.robot_calculate_pose(reference='gripper')
            T = rf.posVec2transMat(tvec)
            Tz = rf.trans_mat(rf.rotz(0), np.array([0., 0., -1.*safe_distance], ndmin=2).T)
            T_des = T@Tz#@np.linalg.inv(self.Tgrip)
            pV = rf.transMat2posVec(T_des)
            self.robot_move_linear(target_pose=pV, speed=0.05)

        else:
            warnings.warn('The operation selected is not recognized. It will be ommited')

        return True

    def robot_rotateL_wrt_tcp(self, angles:list = [0., 0., 0.], reference:str = 'gripper', trans_mat:np.array = np.eye(4))-> np.array:
        '''
        This function implements rotations in three different frames of the robot. Gripper (tool frame),
        Camera (camera_frame), or Flange (robot flange frame).

        Arguments:
            angle: Float value always in degrees
            direction: string value 'x', 'y', 'z' with respect to tcp frame.
                       Default direction is 'z'
            reference: string value indicating which tcp to use. Defaults to 'gripper'

        Returns:
            pose_vector = A np.array containing the calculated pose vector.
        '''
        if not self.rtde_c.isConnected():
            raise Exception("An error occurred. The robot is currently disconnected.")
        tvec = self.robot_calculate_pose(reference=reference, trans_mat=trans_mat)
        T = rf.posVec2transMat(tvec)

        Tx = rf.trans_mat(rf.rotx(np.deg2rad(angles[0])), np.array([0., 0., 0.], ndmin=2).T)
        Ty = rf.trans_mat(rf.roty(np.deg2rad(angles[1])), np.array([0., 0., 0.], ndmin=2).T)
        Tz = rf.trans_mat(rf.rotz(np.deg2rad(angles[2])), np.array([0., 0., 0.], ndmin=2).T)

        if np.isclose(np.abs(np.radians(angles[1])), np.pi/2, atol=1e-3):
            warnings.warn("The input angles are close to gimball lock. No rotation movement is provided")
            return rf.transMat2posVec(T)

        M = T@Tx@Ty@Tz

        self.robot_move_linear(rf.transMat2posVec(M), speed=0.1)

        return rf.transMat2posVec(M)

    def robot_rotateJ_wrt_tcp(self, angles:list = [0., 0., 0.], reference:str = 'gripper', trans_mat:np.array = np.eye(4))-> np.array:
        '''
        This function implements rotations in three different frames of the robot. Gripper (tool frame),
        Camera (camera_frame), or Flange (robot flange frame).

        Arguments:
            angle: Float value always in degrees
            direction: string value 'x', 'y', 'z' with respect to tcp frame.
                       Default direction is 'z'
            reference: string value indicating which tcp to use. Defaults to 'gripper'

        Returns:
            pose_vector = A np.array containing the calculated pose vector.
        '''
        if not self.rtde_c.isConnected():
            raise Exception("An error occurred. The robot is currently disconnected.")
        tvec = self.robot_calculate_pose(reference=reference, trans_mat=trans_mat)
        T = rf.posVec2transMat(tvec)

        Tx = rf.trans_mat(rf.rotx(np.deg2rad(angles[0])), np.array([0., 0., 0.], ndmin=2).T)
        Ty = rf.trans_mat(rf.roty(np.deg2rad(angles[1])), np.array([0., 0., 0.], ndmin=2).T)
        Tz = rf.trans_mat(rf.rotz(np.deg2rad(angles[2])), np.array([0., 0., 0.], ndmin=2).T)

        if np.isclose(np.abs(np.radians(angles[1])), np.pi/2, atol=1e-3):
            warnings.warn("The input angles are close to gimball lock. No rotation movement is provided")
            return rf.transMat2posVec(T)

        M = T@Tx@Ty@Tz
        joints = self.robot_calculate_joints(M, reference=reference, trans_mat=trans_mat)

        self.robot_move_cartesian(target_joints=joints, speed=0.1)

        return rf.transMat2posVec(M)

    def robot_moveL_wrt_tcp(self, displace:list = [0., 0., 0.], reference:str = 'gripper', trans_mat:np.array = np.eye(1))-> np.array:
        '''
        This function implements rotations in three different frames of the robot. Gripper (tool frame),
        Camera (camera_frame), or Flange (robot flange frame).

        Arguments:
            angle: Float value always in degrees
            direction: string value 'x', 'y', 'z' with respect to tcp frame.
                       Default direction is 'z'
            reference: string value indicating which tcp to use. Defaults to 'gripper'

        Returns:
            pose_vector = A np.array containing the calculated pose vector.
        '''
        if not self.rtde_c.isConnected():
            raise Exception("An error occurred. The robot is currently disconnected.")
        tvec = self.robot_calculate_pose(reference=reference, trans_mat=trans_mat)

        Td = rf.trans_mat(np.eye(3), np.array(displace, ndmin=2).T)
        M = rf.posVec2transMat(tvec)@Td
        if reference == 'gripper':
            self.rtde_c.setTcp(rf.transMat2posVec(self.Tgrip))
        elif reference == 'camera':
            self.rtde_c.setTcp(rf.transMat2posVec(self.Twcam))
        else:
            self.rtde_c.setTcp(rf.transMat2posVec(trans_mat))

        self.robot_move_linear(rf.transMat2posVec(M), speed=0.1)

        self.rtde_c.setTcp(rf.transMat2posVec(self.Tgrip))

        return rf.transMat2posVec(M)

    def robot_moveJ_wrt_tcp(self, displace:list = [0., 0., 0.], reference:str = 'gripper', trans_mat:np.array = np.eye(1))-> np.array:
        '''
        This function implements rotations in three different frames of the robot. Gripper (tool frame),
        Camera (camera_frame), or Flange (robot flange frame).

        Arguments:
            angle: Float value always in degrees
            direction: string value 'x', 'y', 'z' with respect to tcp frame.
                       Default direction is 'z'
            reference: string value indicating which tcp to use. Defaults to 'gripper'

        Returns:
            pose_vector = A np.array containing the calculated pose vector.
        '''
        if not self.rtde_c.isConnected():
            raise Exception("An error occurred. The robot is currently disconnected.")
        tvec = self.robot_calculate_pose(reference=reference, trans_mat=trans_mat)
        Td = rf.trans_mat(np.eye(3), np.array(displace, ndmin=2).T)
        M = rf.posVec2transMat(tvec)@Td
        joints = self.robot_calculate_joints(M, reference=reference, trans_mat=trans_mat)

        self.robot_move_cartesian(target_joints=joints, speed=0.1)

        return joints


    def visualize(self):
        self.visual = True
        self.robot_sim.q = self.rtde_r.getActualQ()

        self.env.launch(realtime=True)
        self.env.add(self.robot_sim, robot_alpha=True, collision_alpha=False)