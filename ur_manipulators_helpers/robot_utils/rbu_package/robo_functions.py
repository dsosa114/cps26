import numpy as np
from numpy import linalg as LA
import sympy as sp
import pandas as pd
from scipy.spatial.transform import Rotation
from typing import List, Tuple, Dict
from scipy.spatial.transform import Rotation as R
import random
# from spatialmath.base import trotz, transl
# from roboticstoolbox import DHRobot, RevoluteMDH, RevoluteDH
# from roboticstoolbox.robot.Robot import Robot

import copy

#from sympy import *
import os
# import warnings
# warnings.filterwarnings('ignore')

def p_print(matrix:np.ndarray, round:int = 4):
    try:
        from sympy import init_printing
        init_printing(use_unicode=True, wrap_line=False)
    except ImportError:
        print("Sympy is not installed. Pretty printing will not be available.")

    return sp.Matrix(matrix.round(round))

def deg2rad(angle):
    angle = np.pi * angle / 180.
    return angle

def rad2deg(angle):
    angle = 180. * angle  / np.pi
    return angle

def s_skew_symmetric_matrix_from_vector(vector):
  """
  This function creates a skew-symmetric matrix in SymPy 
  that corresponds to the cross product with a given 3-component vector.

  Args:
    vector: A SymPy Matrix object representing a 3-component vector.

  Returns:
    A SymPy Matrix object representing the skew-symmetric matrix.
  """

  x, y, z = vector
  matrix = sp.Matrix([
      [0, -z, y],
      [z, 0, -x],
      [-y, x, 0]
  ])
  return matrix

# %%
def s_skew_symmetric_matrix_to_vector(matrix):
  """
  Converts a 3x3 skew-symmetric matrix to its corresponding 3-component vector in SymPy .

  Args:
    matrix: The 3x3 skew-symmetric matrix as a SymPy  array.

  Returns:
    A SymPy  array representing the 3-component vector.

  Raises:
    ValueError: If the input matrix is not 3x3 or not skew-symmetric.
  """

  # Check if the matrix is 3x3
  if matrix.shape!= (3, 3):
    raise ValueError("Matrix must be 3x3")

  # Check if the matrix is skew-symmetric
  if not matrix.equals(-matrix.T):
    raise ValueError("Matrix must be skew-symmetric")

  # Extract the vector components
  vector = sp.Matrix([matrix[2,1], matrix[0, 2], matrix[1, 0]])
  return vector

# %%
def s_rotx(angle)-> sp.Matrix:
    ''' Calculate symbolic Rot x matrix\n
        Arguments:
            angle: should be in radians or could be an sp.Symbol
        Return:
            Rotation matrix around x-axis. Type sp.Matrix
    '''
    return sp.Matrix([[1, 0, 0], [0, sp.cos(angle), -sp.sin(angle)], [0, sp.sin(angle), sp.cos(angle)]])

# %%
def s_roty(angle)-> sp.Matrix:
    ''' Calculate symbolic Rot y matrix\n
        Arguments:
            angle: should be in radians or could be an sp.Symbol
        Return:
            Rotation matrix around y-axis. Type sp.Matrix
    '''
    return sp.Matrix([[sp.cos(angle), 0, sp.sin(angle)], [0, 1, 0], [-sp.sin(angle), 0, sp.cos(angle)]])

# %%
def s_rotz(angle)-> sp.Matrix:
    ''' Calculate symbolic Rot z matrix\n
        Arguments:
            angle: should be in radians or could be an sp.Symbol
        Return:
            Rotation matrix around z-axis. Type sp.Matrix
    '''
    return sp.Matrix([[sp.cos(angle), -sp.sin(angle), 0], [sp.sin(angle), sp.cos(angle), 0], [0, 0, 1]])

# %%
def s_general_rot(parameters:Tuple, representation:str='exp'):
    Rot = sp.eye(3)
    if representation == 'exp':
        if len(parameters) < 4:
            omega = str(parameters[0])
            vector = (sp.Symbol(omega + '_x'), sp.Symbol(omega + '_y'), sp.Symbol(omega + '_z'))
        else:
            vector = parameters[0:3]
        skew = s_skew_symmetric_matrix_from_vector(vector)
        Rot = sp.eye(3)+sp.sin(parameters[-1])*skew + (1-sp.cos(parameters[-1]))*skew**2
    if representation == 'zyx':
        Rot = Rot@s_rotz(parameters[0])@s_roty(parameters[1])@s_rotx(parameters[2])
    if representation == 'zyz':
        Rot = Rot@s_rotz(parameters[0])@s_roty(parameters[1])@s_rotz(parameters[2])
    if representation == 'rpy':
        Rot = s_rotz(parameters[0])@s_roty(parameters[1])@s_rotx(parameters[2])@Rot
    return Rot

# %%
def s_rot2vec(R:sp.Matrix, representation:str='exp'):
    trR = R[0,0] + R[1,1] + R[2,2]
    if representation == 'exp':
        angle = sp.acos(0.5*(trR-1))
        skew = 1/(2*sp.sin(angle))*(R-R.T)
        vec_w = s_skew_symmetric_matrix_to_vector(skew)
        vector = sp.Matrix.hstack(vec_w, angle)
    return vector

# %%
def s_ht_mat(rot:sp.Matrix, pos:sp.Matrix) -> sp.Matrix:
    ''' Calculate the symbolic homogeneous transformation matrix\n
        Arguments:
            rot: should be a 3x3 rotation matrix of type sp.Matrix.
            pos: sould be a 3x1 position vector of type sp.Matrix.
        Return:
            Homogeneous transformation matix
    '''
    homogeneous = sp.Matrix.vstack(sp.Matrix.hstack(rot, pos), sp.Matrix([[0, 0, 0, 1]]))
    return homogeneous

# %%
def s_ht2rot_trans(T:sp.Matrix) -> Tuple[sp.Matrix, sp.Matrix]:
    ''' Get the rotation matrix and position vector from a symbolic homogeneous transformation matrix\n
        Arguments:
            T : Homogeneous transformation matix
        Return:
            rot: should be a 3x3 rotation matrix of type sp.Matrix.
            pos: sould be a 3x1 position vector of type sp.Matrix.
    '''
    rot = T[0:3, 0:3]
    pos = T[0:3, 3]
    return rot, pos

# %%
def s_inverse_ht_mat(T:sp.Matrix)-> sp.Matrix:
    ''' Calculate the symbolic inverse of an homogeneous transformation matrix\n
        Arguments:
            T: should be a 4x4 homogeneous transformation matrix of type sp.Matrix.
        Return:
            Inverse of an homogeneous transformation matix
    '''
    T_inv = sp.Matrix(T)
    T_inv[0:3, 3] = -T[0:3, 0:3].T * T[0:3, 3]
    T_inv[0:3, 0:3] = T[0:3, 0:3].T
    return sp.trigsimp(T_inv)

# %% [markdown]
# ### Numerical functions

# %%
def skew_symmetric_matrix_from_vector(vector:tuple):
  """
  This function creates a skew-symmetric matrix that corresponds to the cross product with a given 3-component vector.

  Args:
    vector: A tuple object representing a 3-component vector.

  Returns:
    A numpy array object representing the skew-symmetric matrix.
  """

  x, y, z = vector
  matrix = np.array([
      [0, -z, y],
      [z, 0, -x],
      [-y, x, 0]
  ])
  return matrix

# %%
def skew_symmetric_matrix_to_vector(matrix):
  """
  Converts a 3x3 skew-symmetric matrix to its corresponding 3-component vector in NumPy.

  Args:
    matrix: The 3x3 skew-symmetric matrix as a NumPy array.

  Returns:
    A NumPy array representing the 3-component vector.

  Raises:
    ValueError: If the input matrix is not 3x3 or not skew-symmetric.
  """

  # Check if the matrix is 3x3
  if matrix.shape != (3, 3):
    raise ValueError("Matrix must be 3x3")

  # Check if the matrix is skew-symmetric
  if not np.allclose(matrix, -matrix.T):
    raise ValueError("Matrix must be skew-symmetric")

  # Extract the vector components
  vector = np.array([matrix[2, 1], matrix[0, 2], matrix[1, 0]])
  return vector

# %%
def get_random_nonzero_index(array_list):
  """
  Returns the index of a randomly selected non-zero array 
  from a list of NumPy arrays.

  Args:
    array_list: A list of NumPy arrays.

  Returns:
    The index of a randomly selected non-zero array, 
    or None if there are no non-zero arrays.
  """

  nonzero_indices = [i for i, arr in enumerate(array_list) if np.isclose(np.linalg.norm(arr), 1.)]
  if nonzero_indices:
    return random.choice(nonzero_indices)
  else:
    return None

# %%
def is_valid_rotation_matrix(matrix):
  """
  Checks if a NumPy array is a valid rotation matrix.

  Args:
    matrix: The NumPy array to check.

  Returns:
    True if the matrix is a valid rotation matrix, False otherwise.
  """

  # Check if the matrix is 3x3
  if matrix.shape != (3, 3):
    return False

  # Check if the matrix is orthogonal (transpose is equal to inverse)
  if not np.allclose(matrix.T, np.linalg.inv(matrix), atol=1e-4):
    print(f"The matrix is no orthogonal: {matrix.T} and {np.linalg.inv(matrix)}")
    return False

  # Check if the determinant is 1
  if not np.isclose(np.linalg.det(matrix), 1, atol=1e-4):
    print(f"Determinant of matrix is {np.linalg.det(matrix)}, not 1")
    return False

  return True

# %%
def rotx(angle:float, units:str='rad', as_T:bool = False)-> np.array:
    ''' Calculate numerical rotation matrix around x-axis\n
        Arguments:
            angle: Numerical value of the angle of rotation
            units: Radians('rad') or degrees('deg'). Default: 'rad'
            as_T: If True, returns a 4x4 transformation matrix. Default: False
        Return:
            Rotation matrix around x-axis. Type np.array
    '''
    if units == 'deg':
        angle = np.deg2rad(angle)
    
    rot_x = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
    if as_T:
        T = np.zeros((4,4))
        T[:3, :3] = rot_x
        T[3, 3] = 1.

        return T
    else: 
        return rot_x

# %%
def roty(angle:float, units:str='rad', as_T:bool = False)-> np.array:
    ''' Calculate numerical rotation matrix around y-axis\n
        Arguments:
            angle: Numerical value of the angle of rotation
            units: Radians('rad') or degrees('deg'). Default: 'rad'
            as_T: If True, returns a 4x4 transformation matrix. Default: False
        Return:
            Rotation matrix around y-axis. Type np.array
    '''
    if units == 'deg':
        angle = np.deg2rad(angle)
        
    rot_y =  np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])

    if as_T:
        T = np.zeros((4,4))
        T[:3, :3] = rot_y
        T[3, 3] = 1.

        return T
    else: 
        return rot_y
# %%
def rotz(angle:float, units:str='rad', as_T:bool = False)-> np.array:
    ''' Calculate numerical rotation matrix around z-axis\n
        Arguments:
            angle: Numerical value of the angle of rotation
            units: Radians('rad') or degrees('deg'). Default: 'rad'
            as_T: If True, returns a 4x4 transformation matrix. Default: False
        Return:
            Rotation matrix around z-axis. Type np.array
    '''
    if units == 'deg':
        angle = np.deg2rad(angle)

    rot_z = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])

    if as_T:
        T = np.zeros((4,4))
        T[:3, :3] = rot_z
        T[3, 3] = 1.

        return T
    else: 
        return rot_z
# %%
def general_rot(parameters:tuple, representation:str='exp'):
    Rot = np.eye(3)
    vector = np.array(parameters)
    if representation == 'exp':
        if len(vector) < 4:
            raise ValueError(f"expected (4,) array, got ({len(vector)},) array.")
        else:
            if not np.isclose(np.linalg.norm(vector[0:3]), 1, rtol=1e-4):
                if np.linalg.norm(vector[0:3]) == 0.:
                    return Rot
                else:
                    raise ValueError(f"norm of the input vector is: {np.linalg.norm(vector[0:3])}, and it should be one.")
        skew = skew_symmetric_matrix_from_vector(vector[0:3])
        Rot = np.eye(3)+np.sin(vector[-1])*skew + (1-np.cos(vector[-1]))*skew@skew
    if representation == 'zyx':
        if len(vector) < 3:
            raise ValueError(f"expected (3,) array, got ({len(vector)},) array.")
        Rot = Rot@rotz(vector[0])@roty(vector[1])@rotx(vector[2])
    if representation == 'zyz':
        if len(vector) < 3:
            raise ValueError(f"expected (3,) array, got ({len(vector)},) array.")
        Rot = Rot@rotz(vector[0])@roty(vector[1])@rotz(vector[2])
    if representation == 'rpy':
        if len(vector) < 3:
            raise ValueError(f"expected (3,) array, got ({len(vector)},) array.")
        Rot = rotz(vector[0])@roty(vector[1])@rotx(vector[2])@Rot
    if representation == 'quaternion':
        if len(vector) < 4:
            raise ValueError(f"expected (4,) array, got ({len(vector)},) array.")
        q = vector
        # Rot = np.array([q[0]**2+q[1]**2-q[2]**2-q[3]**2, 2*])
    return Rot

# %%
def rot2vec(R:np.array, representation:str='exp'):
    if not is_valid_rotation_matrix(R):
        raise ValueError(f"The input matrix is not a valid rotation matrix")
    if representation == 'exp':
        trR = np.trace(R)
        if np.allclose(R, np.eye(3), atol=1e-5):
            vector = np.zeros((4,))
        elif np.isclose(trR, -1, atol=1e-3):
            angle = np.pi
            vec_w = [1/(np.sqrt(2*(1+R[2,2])))*np.array([R[0,2], R[1,2], 1 + R[2,2]]), 
                     1/(np.sqrt(2*(1+R[1,1])))*np.array([R[0,1], 1 + R[1,1], R[2,1]]),
                     1/(np.sqrt(2*(1+R[0,0])))*np.array([1 + R[0,0], R[1,0], R[2,0]])]
            get_random_nonzero_index(vec_w)
            vector = np.hstack((vec_w[get_random_nonzero_index(vec_w)], angle))
        else:
            angle = np.arccos(0.5*(trR-1))
            skew = 1/(2*np.sin(angle))*(R-R.T)
            vec_w = skew_symmetric_matrix_to_vector(skew)
            vector = np.hstack((vec_w, angle))
    if representation == 'zyx' or representation == 'rpy':
        if np.isclose(R[2,0], -1, atol=1e-4):
            beta = np.pi/2
            alpha = 0
            gamma = np.arctan2(R[0,1], R[1,1])
        elif np.isclose(R[2,0], 1, atol=1e-4):
            beta = -np.pi/2
            alpha = 0
            gamma = -np.arctan2(R[0,1], R[1,1])
        else:
            beta = np.arctan2(-R[2,0], np.sqrt(R[0,0]**2+R[1,0]**2))
            alpha = np.arctan2(R[1,0], R[0,0])
            gamma = np.arctan2(R[2,1], R[2,2])
        vector = np.hstack((alpha, beta, gamma))
    if representation == 'zyz':
        if np.isclose(R[2,2], -1, atol=1e-4):
            beta = np.pi
            alpha = np.arctan2(-R[1,0], -R[0,0])
            gamma = 0
        elif np.isclose(R[2,0], 1, atol=1e-4):
            beta = 0
            alpha = np.arctan2(R[1,0], R[0,0])
            gamma = 0
        else:
            beta = np.arctan2(np.sqrt(R[0,2]**2+R[1,2]**2), R[2,2])
            alpha = np.arctan2(R[1,2], R[0,2])
            gamma = np.arctan2(R[2,1], -R[2,0])
        vector = np.hstack((alpha, beta, gamma))
    return vector


def rot(ux, uy, uz):
    rot = np.vstack((np.vstack((ux, uy)), uz))
    return(rot.T)

def sym_rot(ux, uy, uz):
    rot = np.vstack((np.vstack((ux, uy)), uz))
    return(rot.T)

def skew3(x):
    x = x.flatten()
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])

def rotMat2rotVec(R):
    A = (R - R.T)/2
    rho = np.array([A[2,1], A[0,2], A[1,0]],ndmin=2).T
    s = LA.norm(rho)
    c = (np.sum(R.diagonal()) - 1 )/ 2
    if s == 0:
        if c == 1:
            r = np.zeros((3, 1))
        if c == -1:
            test = R + np.eye(3)
            nz_col = np.nonzero(np.sum(test, axis=0))
            ith_col = np.random.choice(nz_col[0])
            v = test[:,ith_col].T
            u = v / LA.norm(v)
            r = u*np.pi
            if ((LA.norm(r)==np.pi) and
                ((np.abs(r[0])==np.abs(r[1])==0.)and np.abs(r[2]) == 0.) or
                ((np.abs(r[0]) == 0.) and (np.abs(r[1]) == 0.)) or
                (np.abs(r[0]) == 0.)):
                r = -r
    else:
        u = rho/s
        theta = np.arctan2(s, c)
        r = u*theta

    return r

def rotVec2rotMat(r):
    theta = LA.norm(r)
    if theta == 0.:
        R = np.eye(3)
    else:
        u = r/theta
        R = np.eye(3)*np.cos(theta)+(1-np.cos(theta))*u@u.T+skew3(u)*np.sin(theta)
    return R

def rotMat2rotVec_sp(R):
    return Rotation.from_matrix(R).as_rotvec()

def rotMat2quaternion(R):
    return Rotation.from_matrix(R).as_quat()

def rotVec2quaternion(r):
    if r.ndim > 1:
        r = r.flatten()
    return Rotation.from_rotvec(r).as_quat()

def rotVec2eulerAng(r, sequence="zyx"):
    if r.ndim > 1:
        r = r.flatten()
    print("Euler angle representation:" + sequence)
    return Rotation.from_rotvec(r).as_euler(sequence)

def eulerAng2rotVec(rpy):
    return Rotation.from_euler("zyx", rpy).as_rotvec()


def ht_mat(rot:np.array, pos:np.array) -> np.array:
    ''' Calculate the numerical homogeneous transformation matrix\n
        Arguments:
            rot: should be a 3x3 rotation matrix of type np.array.
            pos: sould be a 3x1 position vector of type np.array.
        Return:
            Homogeneous transformation matix. Type np.array
    '''
    if pos.ndim <= 1:
        pos = np.expand_dims(pos, axis=1)
    homogeneous = np.vstack((np.hstack((rot, pos)), [0., 0., 0., 1.]))
    return homogeneous

# %%
def ht2rot_trans(T:np.array) -> Tuple[np.array, np.array]:
    ''' Get the rotation matrix and position vector from a homogeneous transformation matrix\n
        Arguments:
            T : Homogeneous transformation matix
        Return:
            rot: should be a 3x3 rotation matrix of type np.array.
            pos: sould be a 3x1 position vector of type np.array.
    '''
    rot = T[0:3, 0:3]
    pos = T[0:3, 3]
    return rot, pos

# %%
def inverse_ht_mat(T:np.array)-> np.array:
    ''' Calculate the inverse of an homogeneous transformation matrix\n
        Arguments:
            T: should be a 4x4 homogeneous transformation matrix of type np.array.
        Return:
            Inverse of an homogeneous transformation matix
    '''
    T_inv = np.empty_like(T)
    T_inv[0:3, 3] = -T[0:3, 0:3].T @ T[0:3, 3]
    T_inv[0:3, 0:3] = T[0:3, 0:3].T
    return T_inv

def s_DH_ht_mat(a, alpha, d, phi, modified:bool = True):
    r''' Calculate the Denavit-Hartenberg (DH) homogeneous transformation matrix

        Arguments:
            a: DH 'link length' parameter. (Mod) Length of the mutually perpendicular line between $z_{i-1}$ and $z_i$ along $x_{i-1}$-axis.
            alpha: DH 'link twist' parameter. (Mod) Angle from $z_{i-1}$ to $z_{i}$ measured about $x_{i-1}$-axis.
            d: DH 'link offset' parameter. (Mod) Distance from the intersection of $x_{i-1}$ and  $z_{i}$ to the origin of the link i-frame along the  $z_{i}$-axis.
            phi: DH 'joint angle' parameter. (Mod) Angle from $x_{i-1}$ to $x_{i}$ measured about $z_{i}$-axis.
            modified: Select between modified or standard DH convention. Default: True (modified version)
        Return:
            An homogeneous transformation matix.
    '''
    tvec_a = sp.Matrix([[a, 0., 0.]]).T
    tvec_d = sp.Matrix([[0., 0., d]]).T
    tvec = sp.zeros(1, 3).T
    if modified:
        T = s_ht_mat(s_rotx(alpha), tvec)@s_ht_mat(s_rotx(0), tvec_a)@s_ht_mat(s_rotz(0), tvec_d)@s_ht_mat(s_rotz(phi), tvec)
    else:
        T = s_ht_mat(s_rotz(phi), tvec)@s_ht_mat(s_rotz(0), tvec_d)@s_ht_mat(s_rotx(0), tvec_a)@s_ht_mat(s_rotx(alpha), tvec)
    return T

# %%
def s_exp_ht_mat(S:sp.Matrix, theta):
    r''' Calculate the exponential coordinate homogeneous transformation matrix

        Arguments:
            S: the normalized twist (screw axis)
            theta: angle of rotation
        Return:
            An homogeneous transformation matix.
    '''
    angular = S[0:3]
    linear = S[3:]
    skew = s_skew_symmetric_matrix_from_vector(angular)
    rot = s_general_rot((*angular, theta))
    T = sp.eye(4)
    T[0:3, 0:3] = rot
    T[0:3, 3] = (sp.eye(3)*theta + (1-sp.cos(theta))*skew + (theta-sp.sin(theta))*skew**2)@sp.Matrix(linear)
    return T

# %%
def s_DH_table(a:list, alpha:list, d:list, phi:list, joints:list, start:int=0)->pd.DataFrame:
  r''' Create a Denavit-Hartenberg (DH) parameters table.

        Arguments:
            a: List of DH 'link length' parameters. 
                (Mod) Length of the mutually perpendicular line between $z_{i-1}$ and $z_i$ along $x_{i-1}$-axis.

            alpha: List of DH 'link twist' parameters. 
                (Mod) Angle from $z_{i-1}$ to $z_{i}$ measured about $x_{i-1}$-axis.

            d: List of DH 'link offset' parameters. 
                (Mod) Distance from the intersection of $x_{i-1}$ and  $z_{i}$ to the origin of the link i-frame along the  $z_{i}$-axis.

            phi: List of DH 'joint angle' parameters. 
                (Mod) Angle from $x_{i-1}$ to $x_{i}$ measured about $z_{i}$-axis.

            joints: List of joint types, where 0: revolute joint, 1: prismatic joint, -1: no joint. 
        Return:
            An homogeneous transformation matix.
    '''
  sym_d = []
  sym_phi=[]
  sym_alpha=list(map(sp.rad, alpha))
  for i, mask_value in enumerate(joints):
    if mask_value == 1:
        if isinstance(d[i], int):
            sym_d.append(sp.Symbol('d_'+ str(i+start+1)) + d[i])  # Add to group_zeros if mask value is 0
        else:
            sym_d.append(sp.Symbol('d_'+ str(i+start+1)) + d[i])  # Add to group_zeros if mask value is 0
        sym_phi.append(sp.rad(phi[i]))
    elif mask_value == 0:
        sym_d.append(d[i])
        if isinstance(phi[i], float) or isinstance(phi[i], int):
            sym_phi.append(sp.Symbol(r'\theta_'+ str(i+start+1)) + sp.rad(phi[i]))  # Add to group_ones if mask value is
        else: 
            sym_phi.append(sp.Symbol(r'\theta_'+ str(i+start+1)) + phi[i])
    else:
        sym_d.append(d[i])
        sym_phi.append(sp.rad(phi[i]))

  dh_table = pd.DataFrame({'a': a, 'alpha': sym_alpha, 'd': sym_d, 'phi': sym_phi})
  return dh_table

# %%
def s_fkine_DH(dh_table:pd.DataFrame, modified:bool=True) -> Tuple[sp.Matrix, List[sp.Matrix]]:
    r''' Calculate the forward kinematics (FK) of a robot from a DH-table.

        Arguments:
            dh_table: DH parameter table. Type: pandas.DataFrame
            modified: Select between modified or standard DH convention. Default: True (modified version)
        Return:
            T_acc: An homogeneous transformation matix that represents the robot FK.
            ind_T: A list of each intermediate transformation matrix in the FK calculation.
    '''
    ind_T = []
    T_acc = sp.Matrix.eye(4)
    for i in range(len(dh_table)):
        T = s_DH_ht_mat(*dh_table.iloc[i], modified)
        T_acc = T_acc @ T
        T_acc = sp.trigsimp(T_acc)
        ind_T.append(T)

    return T_acc, ind_T

# %%
def s_twist_list(w:List[List], q:List[List])-> List[sp.Matrix]:
    S = []
    
    for i, w_i in enumerate(w):
        v_i = list(-1 * sp.Matrix(w_i).cross(sp.Matrix(q[i])))
        S_i = w_i + v_i
        S.append(S_i)
    return S

# %%
def s_twist_table(wp_list:List[List])-> List[sp.Matrix]:
    S = []
    
    for wp in wp_list:
        v_i = list(-1 * sp.Matrix(wp[:3]).cross(sp.Matrix(wp[3:])))
        S_i = wp[:3] + v_i
        S.append(S_i)
    return pd.DataFrame(S)
# %%
def s_fkine_PoE(S_table:pd.DataFrame, M:sp.Matrix, space:bool=True) -> Tuple[sp.Matrix, List[sp.Matrix]]:
    r''' Calculate the forward kinematics (FK) of a robot from a DH-table.

        Arguments:
            dh_table: DH parameter table. Type: pandas.DataFrame
            modified: Select between modified or standard DH convention. Default: True (modified version)
        Return:
            T_acc: An homogeneous transformation matix that represents the robot FK.
            ind_T: A list of each intermediate transformation matrix in the FK calculation.
    '''
    ind_T = []
    T_acc = sp.Matrix.eye(4)
    for i in range(len(S_table)):
        T = s_exp_ht_mat(S_table.iloc[i], sp.Symbol(r'\theta_'+ str(i+1)))
        T_acc = T_acc @ T
        T_acc = sp.trigsimp(T_acc)
        ind_T.append(sp.trigsimp(T))

    if space:
        T_acc = sp.trigsimp(T_acc@M)
    else:
        T_acc = sp.trigsimp(M@T_acc)
    return T_acc, ind_T

def trans_mat(rot, tras):
    if tras.ndim <= 1:
        tras = np.expand_dims(tras, axis=1)
    homogeneous = np.vstack((np.hstack((rot, tras)), [0., 0., 0., 1.]))
    return homogeneous

def sym_trans_mat(rot, tras):
    #homogeneous = sp.Matrix.hstack(rot, tras)
    if tras.ndim <= 1:
        tras = np.expand_dims(tras, axis=1)
    homogeneous = sp.Matrix.vstack(sp.Matrix.hstack(rot, tras), sp.Matrix([[0, 0, 0, 1]]))
    return homogeneous

def dh_trans(theta, a, d, alpha, modified = False):
    tvec_a = np.array([[a, 0., 0.]]).T
    tvec_d = np.array([[0., 0., d]]).T
    tvec = np.zeros((1, 3)).T
    if modified:
        T = trans_mat(rotx(alpha), tvec)@trans_mat(rotx(0), tvec_a)@trans_mat(rotz(0), tvec_d)@trans_mat(rotz(theta), tvec)
    else:
        T = trans_mat(rotz(theta), tvec)@trans_mat(rotz(0), tvec_d)@trans_mat(rotx(0), tvec_a)@trans_mat(rotx(alpha), tvec)
    return(T)

def sym_dh_trans(theta, a, d, alpha, modified = False):
    tvec_a = sp.Matrix([[a, 0., 0.]]).T
    tvec_d = sp.Matrix([[0., 0., d]]).T
    tvec = sp.zeros(1, 3).T
    if modified:
        T = sym_trans_mat(s_rotx(alpha), tvec)@sym_trans_mat(s_rotx(0), tvec_a)@sym_trans_mat(s_rotz(0), tvec_d)@sym_trans_mat(s_rotz(theta), tvec)
    else:
        T = sym_trans_mat(s_rotz(theta), tvec)@sym_trans_mat(s_rotz(0), tvec_d)@sym_trans_mat(s_rotx(0), tvec_a)@sym_trans_mat(s_rotx(alpha), tvec)
    return(T)

def s_fkine_ht_list(ht_list:List[sp.Matrix]):
    r''' Calculate the forward kinematics (FK) of a robot from a list of homogeneous transformations.

        Arguments:
            ht_list: List of homogeneous matrices to use.
        Return:
            T_acc: An homogeneous transformation matix that represents the robot FK.
    '''
    T_acc = sp.Matrix.eye(4)
    for T in ht_list:
        T_acc = T_acc @ T
        T_acc = sp.trigsimp(T_acc)

    return T_acc

# %%
def s_invert_ht_list(T:list):
    T_rev = T[::-1]
    T_inv = []
    T_acc = sp.Matrix.eye(4)
    for T_i in T_rev:
        T_acc = s_inverse_ht_mat(T_i) @ T_acc
        T_acc = sp.trigsimp(T_acc)
        T_inv.append(s_inverse_ht_mat(T_i))
    T_inv.reverse()
    return T_acc, T_inv

def DH_ht_mat(a, alpha, d, phi, modified:bool = True):
    r''' Calculate the Denavit-Hartenberg (DH) homogeneous transformation matrix

        Arguments:
            a: DH 'link length' parameter. (Mod) Length of the mutually perpendicular line between $z_{i-1}$ and $z_i$ along $x_{i-1}$-axis.
            alpha: DH 'link twist' parameter. (Mod) Angle from $z_{i-1}$ to $z_{i}$ measured about $x_{i-1}$-axis.
            d: DH 'link offset' parameter. (Mod) Distance from the intersection of $x_{i-1}$ and  $z_{i}$ to the origin of the link i-frame along the  $z_{i}$-axis.
            phi: DH 'joint angle' parameter. (Mod) Angle from $x_{i-1}$ to $x_{i}$ measured about $z_{i}$-axis.
            modified: Select between modified or standard DH convention. Default: True (modified version)
        Return:
            An homogeneous transformation matix.
    '''
    tvec_a = np.array([[a, 0., 0.]]).T
    tvec_d = np.array([[0., 0., d]]).T
    tvec = np.zeros((1, 3)).T
    if modified:
        T = ht_mat(rotx(alpha), tvec)@ht_mat(rotx(0), tvec_a)@ht_mat(rotz(0), tvec_d)@ht_mat(rotz(phi), tvec)
    else:
        T = ht_mat(rotz(phi), tvec)@ht_mat(rotz(0), tvec_d)@ht_mat(rotx(0), tvec_a)@ht_mat(rotx(alpha), tvec)
    return T

# %%
def exp_ht_mat(S:np.array, theta):
    r''' Calculate the exponential coordinate homogeneous transformation matrix

        Arguments:
            S: the normalized twist (screw axis)
            theta: angle of rotation
        Return:
            An homogeneous transformation matix.
    '''
    angular = S[0:3]
    # print(angular)
    linear = S[3:]
    # print(linear)
    skew = skew_symmetric_matrix_from_vector(angular)
    # print(skew)
    rot = general_rot((*angular, theta))
    # print(rot)
    T = np.eye(4)
    T[0:3, 0:3] = rot
    T[0:3, 3] = (np.eye(3)*theta + (1-np.cos(theta))*skew + (theta-np.sin(theta))*skew@skew)@linear
    return T
# %%
def twist_table(wp_list:list, space:bool=True) -> Tuple[np.array, List[np.array]]:
    r''' Calculate the forward kinematics (FK) of a robot from a DH-table.

        Arguments:
            dh_table: DH parameter table. Type: pandas.DataFrame
            modified: Select between modified or standard DH convention. Default: True (modified version)
        Return:
            T_acc: An homogeneous transformation matix that represents the robot FK.
            ind_T: A list of each intermediate transformation matrix in the FK calculation.
    '''
    S_list = []
    for wp in wp_list:
        v = -1*np.cross(wp[:3], wp[3:])
        S_list.append(np.hstack((wp[:3], v)))
    return pd.DataFrame(S_list)

# %%
def fkine_PoE(S_table:pd.DataFrame, joints:list, M:np.array, space:bool=True) -> Tuple[np.array, List[np.array]]:
    r''' Calculate the forward kinematics (FK) of a robot from a DH-table.

        Arguments:
            dh_table: DH parameter table. Type: pandas.DataFrame
            modified: Select between modified or standard DH convention. Default: True (modified version)
        Return:
            T_acc: An homogeneous transformation matix that represents the robot FK.
            ind_T: A list of each intermediate transformation matrix in the FK calculation.
    '''
    ind_T = []
    T_acc = np.eye(4)
    for i in range(len(S_table)):
        T = exp_ht_mat(S_table.iloc[i], joints[i])
        T_acc = T_acc @ T
        ind_T.append(T)

    if space:
        T_acc = T_acc@M
    else:
        T_acc = M@T_acc
    return T_acc, ind_T


# %%
def DH_table(a:list, alpha:list, d:list, phi:list, joints:list)->pd.DataFrame:
  r''' Create a Denavit-Hartenberg (DH) parameters table.

        Arguments:
            a: List of DH 'link length' parameters. 
                (Mod) Length of the mutually perpendicular line between $z_{i-1}$ and $z_i$ along $x_{i-1}$-axis.

            alpha: List of DH 'link twist' parameters. 
                (Mod) Angle from $z_{i-1}$ to $z_{i}$ measured about $x_{i-1}$-axis.

            d: List of DH 'link offset' parameters. 
                (Mod) Distance from the intersection of $x_{i-1}$ and  $z_{i}$ to the origin of the link i-frame along the  $z_{i}$-axis.

            phi: List of DH 'joint angle' parameters. 
                (Mod) Angle from $x_{i-1}$ to $x_{i}$ measured about $z_{i}$-axis.

            joints: List of joint types, where 0: revolute joint, 1: prismatic joint, -1: no joint. 
        Return:
            An homogeneous transformation matix.
    '''
  dh_table = pd.DataFrame({'a': a, 'alpha': alpha, 'd': d, 'phi': phi, 'joint_type': joints})
  return dh_table

# %%
def fkine_DH(dh_table:pd.DataFrame, joints:np.array, modified:bool=True) -> Tuple[sp.Matrix, List[sp.Matrix]]:
    r''' Calculate the forward kinematics (FK) of a robot from a DH-table.

        Arguments:
            dh_table: DH parameter table. Type: pandas.DataFrame
            joints: joint values. Angles should be in radians.
            modified: Select between modified or standard DH convention. Default: True (modified version)
        Return:
            T_acc: An homogeneous transformation matix that represents the robot FK.
            ind_T: A list of each intermediate transformation matrix in the FK calculation.
    '''
    ind_T = []
    T_acc = np.eye(4)
    for i in range(len(dh_table)):
        rowi = dh_table.iloc[i].to_numpy()
        if rowi[-1] == 0:
            rowi[-2] = rowi[-2] + joints[i]
        elif rowi[-1] == 1:
            rowi[-3] = rowi[-3] + joints[i]
        T = DH_ht_mat(*rowi[:-1], modified)
        T_acc = T_acc @ T
        ind_T.append(T)

    return T_acc, ind_T

# %%
def fkine_ht_list(ht_list:List[np.array]):
    r''' Calculate the forward kinematics (FK) of a robot from a list of homogeneous transformations.

        Arguments:
            ht_list: List of homogeneous matrices to use.
        Return:
            T_acc: An homogeneous transformation matix that represents the robot FK.
    '''
    T_acc = np.eye(4)
    for T in ht_list:
        T_acc = T_acc @ T

    return T_acc

# %%
def random_point_in_articulated_robot_workspace(radius, center=(0, 0, 0)):
  """
  Generates a random point within a sphere.

  Args:
    radius: The radius of the sphere.
    center: The coordinates of the sphere's center (default: (0, 0, 0)).

  Returns:
    A NumPy array representing the (x, y, z) coordinates of the random point.
  """

  # Generate random spherical coordinates
  phi = np.random.uniform(0, 2 * np.pi)  # Azimuthal angle
  costheta = np.random.uniform(-1, 1)     # Cosine of polar angle
  u = np.random.uniform(0, 1)           # Uniform random variable for radius

  # Convert to Cartesian coordinates
  theta = np.arccos(costheta)
  r = radius * u**(1/3)  #  Ensures uniform distribution within the sphere
  x = r * np.sin(theta) * np.cos(phi) + center[0]
  y = r * np.sin(theta) * np.sin(phi) + center[1]
  z = r * np.cos(theta) + center[2]

  return np.array([x, y, z])

# %%
def random_rotation_matrix(num_matrices=1):
  """
  Generates random rotation matrices.

  Args:
    num_matrices: The number of rotation matrices to generate (default: 1).

  Returns:
    A NumPy array of shape (num_matrices, 3, 3) containing the random 
    rotation matrices, or a single 3x3 matrix if num_matrices is 1.
  """

  # Generate random quaternions
  quaternions = np.random.rand(num_matrices, 4)

  # Normalize quaternions to ensure they represent valid rotations
  quaternions = quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)

  # Create Rotation objects from quaternions
  rotations = R.from_quat(quaternions)

  # Convert to rotation matrices
  rotation_matrices = rotations.as_matrix()

  if num_matrices == 1:
    return rotation_matrices[0]  # Return a single matrix
  else:
    return rotation_matrices
  
def linear_interpolation(initial_position, target_position, num_points=101):
    """
    Performs linear interpolation between two position vectors.

    Args:
        initial_position (np.array): The starting position vector (e.g., 3D).
        target_position (np.array): The ending position vector (e.g., 3D).
        num_points (int): The number of interpolated points to generate.

    Returns:
        np.array: An array of interpolated position vectors.
                  Shape: (num_points, len(initial_position))
    """
    s = np.linspace(0, 1, num_points)
    # Ensure s is a column vector for correct broadcasting with position vectors
    s_col = s[:, np.newaxis]
    p = (1 - s_col) * initial_position + s_col * target_position
    return p

def slerp(initial_orientation, target_orientation, num_points=101):
    """
    Performs spherical linear interpolation (SLERP) between two orientation vectors (quaternions).

    Args:
        initial_orientation (np.array): The starting orientation vector (quaternion, 4D, [x, y, z, w]).
        target_orientation (np.array): The ending orientation vector (quaternion, 4D, [x, y, z, w]).
        num_points (int): The number of interpolated points to generate.

    Returns:
        np.array: An array of interpolated orientation vectors (quaternions).
                  Shape: (num_points, 4)
    """
    s = np.linspace(0, 1, num_points)

    # Normalize quaternions to ensure they are unit quaternions
    # This is crucial for SLERP to work correctly
    initial_orientation = initial_orientation / LA.norm(initial_orientation)
    target_orientation = target_orientation / LA.norm(target_orientation)

    # Calculate the dot product (cosine of the angle between quaternions)
    dot_product = np.dot(initial_orientation, target_orientation)

    # If the dot product is negative, negate one quaternion to take the shortest path
    # This ensures interpolation always goes the "short way around" the sphere
    if dot_product < 0.0:
        target_orientation = -target_orientation
        dot_product = -dot_product

    # Clamp dot_product to handle potential floating point inaccuracies
    # This prevents arccos from receiving values outside [-1, 1]
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # Calculate the angle between the quaternions
    alpha = np.arccos(dot_product)

    # Handle the case where quaternions are very close (to avoid division by zero or instability)
    if alpha < 1e-6: # If the angle is very small, perform linear interpolation and re-normalize
        q = (1 - s[:, np.newaxis]) * initial_orientation + s[:, np.newaxis] * target_orientation
        # Re-normalize the interpolated quaternions to ensure they remain unit quaternions
        q = q / LA.norm(q, axis=1)[:, np.newaxis]
        return q

    # Perform SLERP using the provided formula
    sin_alpha = np.sin(alpha)
    q = (np.sin(alpha * (1 - s[:, np.newaxis])) / sin_alpha) * initial_orientation + \
        (np.sin(alpha * s[:, np.newaxis]) / sin_alpha) * target_orientation

    return q

def interpolate_robot_trajectory(initial_transform_matrix, target_transform_matrix, num_points=101):
    """
    Interpolates a linear trajectory for a robot given initial and target transformation matrices.
    The position (translation) is interpolated linearly, and the orientation (rotation) is
    interpolated using Spherical Linear Interpolation (SLERP) on quaternions.

    Args:
        initial_transform_matrix (np.array): A 4x4 numpy array representing the initial
                                             transformation matrix.
        target_transform_matrix (np.array): A 4x4 numpy array representing the target
                                            transformation matrix.
        num_points (int): The total number of interpolated transformation matrices to generate,
                          including the initial and target matrices.

    Returns:
        list: A list of 4x4 numpy arrays, each representing an interpolated
              transformation matrix along the trajectory.
    """
    # 1. Extract initial and target position vectors (last column, first 3 rows)
    initial_position = initial_transform_matrix[:3, 3]
    target_position = target_transform_matrix[:3, 3]

    # 2. Extract initial and target 3x3 rotation matrices (top-left 3x3 block)
    initial_rotation_matrix = initial_transform_matrix[:3, :3]
    target_rotation_matrix = target_transform_matrix[:3, :3]

    # 3. Convert rotation matrices to quaternions for SLERP
    # scipy.spatial.transform.Rotation.from_matrix expects a 3x3 rotation matrix
    # .as_quat() converts the Rotation object to a (x, y, z, w) quaternion array
    initial_quaternion = R.from_matrix(initial_rotation_matrix).as_quat()
    target_quaternion = R.from_matrix(target_rotation_matrix).as_quat()

    # 4. Interpolate positions using the provided linear_interpolation function
    interpolated_positions = linear_interpolation(initial_position, target_position, num_points)

    # 5. Interpolate orientations (quaternions) using the provided slerp function
    interpolated_quaternions = slerp(initial_quaternion, target_quaternion, num_points)

    # 6. Combine interpolated positions and orientations into a list of transformation matrices
    trajectory_matrices = []
    for i in range(num_points):
        # Convert each interpolated quaternion back to a 3x3 rotation matrix
        # R.from_quat expects a (x, y, z, w) quaternion array
        interpolated_rotation_matrix = R.from_quat(interpolated_quaternions[i]).as_matrix()

        # Create a new 4x4 identity matrix to build the interpolated transform
        interpolated_transform_matrix = np.identity(4)

        # Place the interpolated 3x3 rotation matrix into the top-left block
        interpolated_transform_matrix[:3, :3] = interpolated_rotation_matrix

        # Place the interpolated 3x1 position vector into the last column (translation part)
        interpolated_transform_matrix[:3, 3] = interpolated_positions[i]

        trajectory_matrices.append(interpolated_transform_matrix)

    return trajectory_matrices


def s_adjoint_mat(T):
    A = sp.zeros(6, 6)
    R = T[:3, :3]
    p = T[:3, 3]
    skew_p = s_skew_symmetric_matrix_from_vector(p)
    A[:3, :3] = R
    A[3:, :3] = skew_p@R
    A[3:, 3:] = R

    return sp.simplify(A)

# %%
def s_Jacobian(ht_list:List[sp.Matrix], joints:List, space:bool=True, standard:bool=True):
    R_acc = []
    p_acc = []
    J_col = []
    for i in range(len(ht_list)):
        rot, pos = s_ht2rot_trans(s_fkine_ht_list(ht_list[:(i+1)]))
        R_acc.append(rot)
        p_acc.append(pos)

    # print(p_acc[-1])
    for i in range(len(ht_list)):
        if space:
            R_current=R_acc[i]
            p_current=p_acc[i]
        else:
            if standard:
                if i == 0:
                    R_current=sp.eye(3)
                    p_current=sp.zeros(3,1)
                else:
                    R_current=R_acc[(i-1)]
                    p_current=p_acc[(i-1)]
            else:
                R_current=R_acc[i]
                p_current=p_acc[i]
        
        
        if joints[i] == 0:
            Ji_a = R_current@sp.Matrix([[0], [0], [1]])
            
            if space:
                Ji_l = -Ji_a.cross(p_current)
            else:
                Ji_l = Ji_a.cross((p_acc[-1]-p_current))
            
            J_col.append(sp.simplify(sp.Matrix.vstack(Ji_a, Ji_l)))
        elif joints[i] == 1:
            Ji_a = sp.zeros(3,1)
            Ji_l = R_current@sp.Matrix([[0], [0], [1]])
        
            J_col.append(sp.simplify(sp.Matrix.vstack(Ji_a, Ji_l)))
    
    J = sp.Matrix.hstack(*J_col)
        #J_col.append(Ji_a)
    return J, J_col

def n_Jacobian(parameter_table:pd.DataFrame, joints:List, space:bool=True, modified:bool=True):
    R_acc = []
    p_acc = []
    J_col = []
    _, ht_list = fkine_DH(parameter_table, joints, modified)
    for i in range(len(ht_list)):
        rot, pos = ht2rot_trans(fkine_ht_list(ht_list[:(i+1)]))
        R_acc.append(rot)
        p_acc.append(pos)

    print(p_acc)
    for i in range(len(ht_list)):
        if space or modified:
            R_current=R_acc[i]
            p_current=p_acc[i]
        else:
            if i == 0:
                R_current=np.eye(3)
                p_current=np.zeros((3,1))
            else:
                R_current=R_acc[i-1]
                p_current=p_acc[i-1]
        
        
        if parameter_table['joint_type'][i] == 0:
            Ji_a = R_current@np.array([[0], [0], [1]])
            
            if space:
                Ji_l = np.cross(-Ji_a, np.expand_dims(p_current, 1),axis=0)
                # Ji_l = -Ji_a.cross(p_current)
            else:
                p = np.subtract(p_acc[-1],p_current.squeeze())
               
                Ji_l = np.cross(Ji_a, p,axis=0)
                # Ji_l = Ji_a.cross((p_acc[-1]-p_current))
            
            J_col.append(np.vstack((Ji_a, Ji_l)))
        elif parameter_table['joint_type'][i] == 1:
            Ji_a = np.zeros((3,1))
            Ji_l = R_current@np.array([[0], [0], [1]])
        
            J_col.append(np.vstack((Ji_a, Ji_l)))
    
    J = np.hstack((J_col))
        #J_col.append(Ji_a)
    return J, J_col


def ur_fkine(df_robot, joints=np.array([0, 0, 0, 0, 0, 0]), modified=False):
    # joints = joints * np.pi / 180
    joints = np.radians(joints)
    dh_robot = df_robot.copy(deep=True)
    dh_robot['theta'] = dh_robot['theta'] + joints#.tolist()
    T_a = []
    T = np.eye(4)
    T_a.append(T)
    for i in range(0, dh_robot.shape[0]):
        rowi = dh_robot.iloc[i].to_numpy()
        Ti = dh_trans(rowi.item(0), rowi.item(1), rowi.item(2), rowi.item(3), modified=modified)
        T = T@Ti 
        T_a.append(T) 
    R = T[0:3,0:3]
    t_vec = T[0:3,3]
    r_vec = rotMat2rotVec_sp(R).flatten()
    return T, R, t_vec, r_vec, T_a

def ur_angular_geometric_jacobian(Ti):
    z0 = np.array([[0],[0],[1]])
    Jo = []
    for i in range(1,len(Ti)+1):
        z = np.array([[0],[0],[1]])
        for j in range(i-1):
            z = np.hstack((z, Ti[j][0:3,0:3]@z0))
        jo = np.hstack((z, np.repeat([[0],[0],[0]],len(Ti)-i, axis=1)))
        # print(z)
        Jo.append(jo)
    return np.array(Jo)

def ur_linear_geometric_jacobian(cm, Ti, Jo: np.array):
    p_cm = cm.copy(deep=True)
    z = pd.DataFrame(Jo[5])
    Jv = []
    jv = np.zeros((3, p_cm.shape[0]))
    for i in range(0, p_cm.shape[0]):
        l_cm = p_cm.iloc[i].to_numpy()
        P = Ti[i]@trans_mat(np.eye(3), l_cm)@np.array([0, 0, 0, 1])
        # jv_c = skew3(z[i].to_numpy())@np.delete(P, -1, axis=0)
        jv[:,i] = skew3(z[i].to_numpy())@np.delete(P, -1, axis=0)
        Jv.append(copy.deepcopy(jv))
    return np.array(Jv)

def ur_link_inertia(I, Ti):
    I_cm = I.copy(deep=True)
    Il = []
    for i in range(0, I_cm.shape[0]):
        i_cm = I_cm.iloc[i].to_numpy()
        il = Ti[i][:3,:3]@i_cm.reshape((3,3))@Ti[i][:3,:3].T 
        Il.append(il)
    return np.array(Il)

## Experimental but cant figure it out how its supposed to work
def ur_diff_lin_geometric_jacobian(cm, Ti, q: np.array):
    p_cm = cm.copy(deep=True)
    Jv = []
    for i in range(0, p_cm.shape[0]):
        l_cm = p_cm.iloc[i].to_numpy()
        P = Ti[i]@trans_mat(np.eye(3), l_cm)@np.array([0, 0, 0, 1])
        x = P[0]
        y = P[1]
        z = P[2]
        for j in range(p_cm.shape[0]):
            Jv = [Jv, [np.diff(x,q[j]), np.diff(y,q[j]), np.diff(z,q[j])]]
        # Jv.append(jv)
    return Jv

### Spatial coordinates arithmetic
### Echeandia paper and Featherstone book 
def cross_motion(v):
    """ 
    Cross_motion:
        spatial/planar cross-product operator (motion).
        
        crm(v)  calculates the 6x6 (or 3x3) matrix such that the expression
        crm(v)*m is the cross product of the motion vectors v and m.  If
        length(v)==6 then it is taken to be a spatial vector, and the return
        value is a 6x6 matrix.  Otherwise, v is taken to be a planar vector, and
        the return value is 3x3.
        
    """
    if v.size == 6 and v.shape[1] == 1:
        vcross = np.block([
            [skew3(v[:3]), np.eye(3)*0],
            [skew3(v[3:]), skew3(v[:3])]
        ])
    elif v.size == 3 and v.shape[1] == 1:
        vcross = np.array([
            [0, 0, 0],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
    return vcross

### Echeandia paper
def cross_force(v):
    """ 
    Cross product force:
        spatial/planar cross-product operator (force).
        
        crf(v)  calculates the 6x6 (or 3x3) matrix such that the expression
        crf(v)*m is the cross product of the motion vectors v with the force vector f.
        If length(v)==6 then it is taken to be a spatial vector, and the return
        value is a 6x6 matrix.  Otherwise, v is taken to be a planar vector, and
        the return value is 3x3.
        
    """
    vcross = -cross_motion(v).T
    return vcross

### Echeandia paper
def icross_force(v):
    """ 
    iCross_force:
        spatial/planar cross-product operator (motion).
        
        crm(v)  calculates the 6x6 (or 3x3) matrix such that the expression
        crm(v)*m is the cross product of the motion vectors v and m.  If
        length(v)==6 then it is taken to be a spatial vector, and the return
        value is a 6x6 matrix.  Otherwise, v is taken to be a planar vector, and
        the return value is 3x3.
        
    """
    if v.size == 6 and v.shape[1] == 1:
        vcross = np.block([
            [-skew3(v[:3]), skew3(v[:3])],
            [-skew3(v[3:]), np.eye(3)*0]
        ])
    return vcross

### Echeandia paper
def body_factorFunctions(I, v, type=3):
    if type == 1:
        B = cross_force(v)@I
    elif type == 2:
        B = icross_force(I@v) - I @ cross_motion(v)
    else:
        B = 1/2 * (cross_force(v)@I + icross_force(I@v) - I@cross_motion(v))
    return B
########################################################################################
########################################################################################
def plane_eq(p1, p2, p3):
    # These two vectors are in the plane
    v1 = p3 - p1
    v2 = p2 - p1

    # the cross product is a vector normal to the plane
    cp = np.cross(v1, v2)
    a, b, c = cp

    # This evaluates a * x3 + b * y3 + c * z3 which equals d
    d = np.dot(cp, p3)

    print('The equation is {0}x + {1}y + {2}z = {3}'.format(a, b, c, d))

    #Normalize vectors
    uz = cp/LA.norm(cp)
    ux = v2/LA.norm(v2)
    uy = np.cross(uz, ux)
    print("Unit vectors x = {}, y = {}, and z = {}".format(ux, uy, uz))
    return ([a, b, c, d], [ux, uy, uz])

def get_joints(model, objective, gripper = 'none', config=0, rad2deg=True):
    if gripper == 'none':
        Tgrip = np.eye(4)
    elif gripper == 'gripper':
        Tgrip = posVec2transMat(UR_robot.fix_frame['gripper'])
    elif gripper == 'grip_cam':
        Tgrip = posVec2transMat(UR_robot.fix_frame['grip_cam'])
    dhp = UR_robot.dhp[model]
    T_objective = posVec2transMat(objective) @ np.linalg.inv(Tgrip)
    cols = ['t1', 't2', 't3', 't4', 't5', 't6']
    ikine = ur_ikine(dhp, T_objective)
    ikine[cols] = ikine[cols].apply(pd.to_numeric)
    if rad2deg:
        ikine[cols] = ikine[cols].apply(np.rad2deg)

    return ikine.loc[config, cols].to_list()

def get_theta1(df_robot, T):
    theta_1 = {'left': 0., 'right': 0.}
    row_4 = df_robot.iloc[3].to_numpy()
    row_6 = df_robot.iloc[5].to_numpy()
    d6Vec = np.array([0., 0., -row_6.item(2), 1.], ndmin=2).T
    P_5 = T@d6Vec
    phi_1 = np.arctan2(P_5.item(1), P_5.item(0))
    if -1. <= (row_4.item(2)/LA.norm(P_5[0:2])) <= 1.:
        phi_2 = np.arccos(row_4.item(2)/LA.norm(P_5[0:2]))
    else:
        phi_2 = np.nan
    theta_1['right'] = phi_1 + phi_2 + np.pi/2
    theta_1['left'] = phi_1 - phi_2 + np.pi/2
    return theta_1

def get_theta5(df_robot, T, t_vec):
    theta_1, _, _, _, _, _ = tuple(t_vec)
    theta_5 = {'up': 0., 'down': 0.}
    row_4 = df_robot.iloc[3].to_numpy()
    row_6 = df_robot.iloc[5].to_numpy()
    uVec = np.array([0., 0., 0., 1.], ndmin=2).T
    P_6 = T@uVec
    num = P_6.item(0)*np.sin(theta_1)-P_6.item(1)*np.cos(theta_1)-row_4.item(2)
    #P_16y = P_6.item(0)*-np.sin(theta_1)+P_6.item(1)*np.cos(theta_1)
    if np.abs(num) <= np.abs(row_6.item(2)):
        phi_ = np.arccos(num/row_6.item(2))
    else:
        phi_ = 0.
    theta_5['up'] = +phi_
    theta_5['down'] = -phi_
    return theta_5

def get_theta6(df_robot, T, t_vec, modified=False):
    theta_1, _, _, _, theta_5, _ = tuple(t_vec)
    uXVec = np.array([1., 0., 0., 0.], ndmin=2).T
    uYVec = np.array([0., 1., 0., 0.], ndmin=2).T
    X_6 = T@uXVec
    Y_6 = T@uYVec
    if modified:
        #num_y = -T[1,0]*np.sin(theta_1)+T[1,1]*np.cos(theta_1)
        #num_x = T[0,0]*np.sin(theta_1)-T[0,1]*np.cos(theta_1)
        #num_y = -X_6.item(1)*np.sin(theta_1)+Y_6.item(1)*np.cos(theta_1)
        #num_x = X_6.item(0)*np.sin(theta_1)-Y_6.item(0)*np.cos(theta_1)
        ###Kelsey Hawkins solution
        num_y = -Y_6.item(0)*np.sin(theta_1)+Y_6.item(1)*np.cos(theta_1)
        num_x = X_6.item(0)*np.sin(theta_1)-X_6.item(1)*np.cos(theta_1)
    else:
        ###Kelsey Hawkins solution
        num_y = -Y_6.item(0)*np.sin(theta_1)+Y_6.item(1)*np.cos(theta_1)
        num_x = X_6.item(0)*np.sin(theta_1)-X_6.item(1)*np.cos(theta_1)
    den = np.sin(theta_5)
    if np.abs(den) < 1e-15:
        theta_6 = 0.0
        # print("denominator = {}".format(den))
    elif (np.abs(num_y) < 1e-15) and (np.abs(num_x) < 1e-15) :
        theta_6 = 0.0
        print("num_y = {} and num_x = {}".format(num_y, num_x))
    else:
        theta_6 = np.arctan2(num_y/den, num_x/den)

    return theta_6

def get_theta3(df_robot, T, t_vec, modified=False):
    theta_1, _, _, _, theta_5, theta_6 = tuple(t_vec)
    theta_3 = {'up': 0., 'down': 0.}
    if modified:
        T_1 = dh_trans(theta_1, df_robot['a_i-1'][0], df_robot['d'][0], df_robot['alpha_i-1'][0], modified=modified)
        T_5 = dh_trans(theta_5, df_robot['a_i-1'][4], df_robot['d'][4], df_robot['alpha_i-1'][4], modified=modified)
        T_6 = dh_trans(theta_6, df_robot['a_i-1'][5], df_robot['d'][5], df_robot['alpha_i-1'][5], modified=modified)
        T_14 = LA.inv(T_1)@T@LA.inv(T_5@T_6)
        uVec = np.array([0., 0., 0., 1.], ndmin=2).T
        #uVec = np.array([0., -df_robot['d'][3], 0., 1.], ndmin=2).T
        P_4 = T_14@uVec
        P_4xz = np.delete(P_4, np.s_[1::2], 0)
        num_t3 = LA.norm(P_4xz)**2 - (df_robot['a_i-1'][2])**2 - (df_robot['a_i-1'][3])**2
        #num_t3 = (LA.norm(P_4[0:3]))**2 - (df_robot['a_i-1'][2])**2 - (df_robot['a_i-1'][3])**2
        den_t3 = 2*df_robot['a_i-1'][2]*df_robot['a_i-1'][3]
        if -1. <= (num_t3/den_t3) <= 1.:
            theta_3['up'] = -np.arccos(num_t3/den_t3)
            theta_3['down'] = np.arccos(num_t3/den_t3)
        else:
            theta_3['up'] = 0.0
            theta_3['down'] = 0.0
    else:
        T_1 = dh_trans(theta_1, df_robot['a'][0], df_robot['d'][0], df_robot['alpha'][0])
        T_5 = dh_trans(theta_5, df_robot['a'][4], df_robot['d'][4], df_robot['alpha'][4])
        T_6 = dh_trans(theta_6, df_robot['a'][5], df_robot['d'][5], df_robot['alpha'][5])
        T_14 = LA.inv(T_1)@T@LA.inv(T_5@T_6)
        uVec = np.array([0., -df_robot['d'][3], 0., 1.], ndmin=2).T
        P_4 = T_14@uVec
        #P_4xz = np.delete(P_4, np.s_[1::2], 0)
        #num_t3 = LA.norm(P_4xz)**2 - (df_robot['a'][1])**2 - (df_robot['a'][2])**2
        num_t3 = LA.norm(P_4[0:3])**2 - (df_robot['a'][1])**2 - (df_robot['a'][2])**2
        den_t3 = 2*df_robot['a'][1]*df_robot['a'][2]
        if -1. <= (num_t3/den_t3) <= 1.:
            theta_3['up'] = -np.arccos(num_t3/den_t3)
            theta_3['down'] = np.arccos(num_t3/den_t3)
        else:
            theta_3['up'] = 0.0
            theta_3['down'] = 0.0
    return theta_3

def get_theta2(df_robot, T, t_vec, modified=False):
    theta_1, _, theta_3, _, theta_5, theta_6 = tuple(t_vec)
    if modified:
        T_1 = dh_trans(theta_1, df_robot['a_i-1'][0], df_robot['d'][0], df_robot['alpha_i-1'][0], modified=modified)
        T_5 = dh_trans(theta_5, df_robot['a_i-1'][4], df_robot['d'][4], df_robot['alpha_i-1'][4], modified=modified)
        T_6 = dh_trans(theta_6, df_robot['a_i-1'][5], df_robot['d'][5], df_robot['alpha_i-1'][5], modified=modified)
        T_14 = LA.inv(T_1)@T@LA.inv(T_5@T_6)
        uVec = np.array([0., 0., 0., 1.], ndmin=2).T
        #uVec = np.array([0., -df_robot['d'][3], 0., 1.], ndmin=2).T
        P_4 = T_14@uVec
        P_4xz = np.delete(P_4, np.s_[1::2], 0)
        phi_1 = np.arctan2(-P_4.item(2), -P_4.item(0))
        phi_2 = np.arcsin(-df_robot['a_i-1'][3]* np.sin(theta_3)/LA.norm(P_4xz))
        #phi_2 = np.arcsin(-df_robot['a_i-1'][3]* np.sin(theta_3)/LA.norm(P_4[0:3]))
        theta_2 = phi_1-phi_2
    else:
        T_1 = dh_trans(theta_1, df_robot['a'][0], df_robot['d'][0], df_robot['alpha'][0])
        T_5 = dh_trans(theta_5, df_robot['a'][4], df_robot['d'][4], df_robot['alpha'][4])
        T_6 = dh_trans(theta_6, df_robot['a'][5], df_robot['d'][5], df_robot['alpha'][5])
        T_14 = LA.inv(T_1)@T@LA.inv(T_5@T_6)
        uVec = np.array([0., -df_robot['d'][3], 0., 1.], ndmin=2).T
        P_4 = T_14@uVec
        phi_1 = np.arctan2(-P_4.item(1), -P_4.item(0))
        phi_2 = np.arcsin(-df_robot['a'][2]* np.sin(theta_3)/LA.norm(P_4[0:3]))
        theta_2 = phi_1-phi_2
    return theta_2

def get_theta4(df_robot, T, t_vec, modified=False):
    theta_1, theta_2, theta_3, _, theta_5, theta_6 = tuple(t_vec)
    if modified:
        T_1 = dh_trans(theta_1, df_robot['a_i-1'][0], df_robot['d'][0], df_robot['alpha_i-1'][0], modified=modified)
        T_2 = dh_trans(theta_2, df_robot['a_i-1'][1], df_robot['d'][1], df_robot['alpha_i-1'][1], modified=modified)
        T_3 = dh_trans(theta_3, df_robot['a_i-1'][2], df_robot['d'][2], df_robot['alpha_i-1'][2], modified=modified)
        T_5 = dh_trans(theta_5, df_robot['a_i-1'][4], df_robot['d'][4], df_robot['alpha_i-1'][4], modified=modified)
        T_6 = dh_trans(theta_6, df_robot['a_i-1'][5], df_robot['d'][5], df_robot['alpha_i-1'][5], modified=modified)
        T_34 = LA.inv(T_2@T_3)@(LA.inv(T_1)@T@LA.inv(T_5@T_6))
        uVec = np.array([1., 0., 0., 0.], ndmin=2).T
        X_34 = T_34@uVec
        theta_4 = np.arctan2(X_34.item(1), X_34.item(0))
    else:
        T_1 = dh_trans(theta_1, df_robot['a'][0], df_robot['d'][0], df_robot['alpha'][0])
        T_2 = dh_trans(theta_2, df_robot['a'][1], df_robot['d'][1], df_robot['alpha'][1])
        T_3 = dh_trans(theta_3, df_robot['a'][2], df_robot['d'][2], df_robot['alpha'][2])
        T_5 = dh_trans(theta_5, df_robot['a'][4], df_robot['d'][4], df_robot['alpha'][4])
        T_6 = dh_trans(theta_6, df_robot['a'][5], df_robot['d'][5], df_robot['alpha'][5])
        T_34 = LA.inv(T_2@T_3)@(LA.inv(T_1)@T@LA.inv(T_5@T_6))
        uVec = np.array([1., 0., 0., 0.], ndmin=2).T
        X_34 = T_34@uVec
        theta_4 = np.arctan2(X_34.item(1), X_34.item(0))

    return theta_4

def ur_ikine(df_robot, T, modified=False):
    results = pd.DataFrame(np.array([[0, 0, 0, 0, 0, 0, 'left', 'up', 'up'],
                                   [0, 0, 0, 0, 0, 0, 'left', 'up', 'down'],
                                   [0, 0, 0, 0, 0, 0, 'left', 'down', 'up'],
                                   [0, 0, 0, 0, 0, 0, 'left', 'down', 'down'],
                                   [0, 0, 0, 0, 0, 0, 'right', 'up', 'up'],
                                   [0, 0, 0, 0, 0, 0, 'right', 'up', 'down'],
                                   [0, 0, 0, 0, 0, 0, 'right', 'down', 'up'],
                                   [0, 0, 0, 0, 0, 0, 'right', 'down', 'down']]),
                   columns=['t1', 't2', 't3', 't4', 't5', 't6', 'elbow_L/R', 'wrist_U/D', 'elbow_U/D'])
    joints = [0., 0., 0., 0., 0., 0.]
    cols = ['t1', 't2', 't3', 't4', 't5', 't6']
    t_1 = get_theta1(df_robot, T)
    for key_1, value_1 in t_1.items():
        if 'left' in key_1:
            joints[0] = value_1
            #print("elbow left = {}".format(rad2deg(value_1)))
            t_5 = get_theta5(df_robot, T, joints)
            for key_5, value_5 in t_5.items():
                if 'up' in key_5:
                    joints[4] = value_5
                    #print("wrist up = {}".format(rad2deg(value_5)))
                    joints[5] = get_theta6(df_robot, T, joints, modified=modified)
                    #print("t_6 = {}".format(rad2deg(joints[5])))
                    t_3 = get_theta3(df_robot, T, joints, modified=modified)
                    for key_3, value_3 in t_3.items():
                        if 'up' in key_3:
                            joints[2] = value_3
                            #print("elbow up = {}".format(rad2deg(value_3)))
                            joints[1] = get_theta2(df_robot, T, joints, modified=modified)
                            #print("t_2 = {}".format(rad2deg(joints[1])))
                            joints[3] = get_theta4(df_robot, T, joints, modified=modified)
                            #print("t_4 = {}".format(rad2deg(joints[3])))
                            results.loc[0, cols] = joints
                        if 'down' in key_3:
                            joints[2] = value_3
                            #print("elbow down = {}".format(rad2deg(value_3)))
                            joints[1] = get_theta2(df_robot, T, joints, modified=modified)
                            #print("t_2 = {}".format(rad2deg(joints[1])))
                            joints[3] = get_theta4(df_robot, T, joints, modified=modified)
                            #print("t_4 = {}".format(rad2deg(joints[3])))
                            results.loc[1, cols] = joints
                if 'down' in key_5:
                    joints[4] = value_5
                    #print("wrist down = {}".format(rad2deg(value_5)))
                    joints[5] = get_theta6(df_robot, T, joints, modified=modified)
                    #print("t_6 = {}".format(rad2deg(joints[5])))
                    t_3 = get_theta3(df_robot, T, joints, modified=modified)
                    for key_3, value_3 in t_3.items():
                        if 'up' in key_3:
                            joints[2] = value_3
                            #print("elbow up = {}".format(rad2deg(value_3)))
                            joints[1] = get_theta2(df_robot, T, joints, modified=modified)
                            #print("t_2 = {}".format(rad2deg(joints[1])))
                            joints[3] = get_theta4(df_robot, T, joints, modified=modified)
                            #print("t_4 = {}".format(rad2deg(joints[3])))
                            results.loc[2, cols] = joints
                        if 'down' in key_3:
                            joints[2] = value_3
                            #print("elbow down = {}".format(rad2deg(value_3)))
                            joints[1] = get_theta2(df_robot, T, joints, modified=modified)
                            #print("t_2 = {}".format(rad2deg(joints[1])))
                            joints[3] = get_theta4(df_robot, T, joints, modified=modified)
                            #print("t_4 = {}".format(rad2deg(joints[3])))
                            results.loc[3, cols] = joints
        #print("++++++++++++++++++++++++++++++++")
        if 'right' in key_1:
            joints[0] = value_1
            #print("elbow right = {}".format(rad2deg(value_1)))
            t_5 = get_theta5(df_robot, T, joints)
            for key_5, value_5 in t_5.items():
                if 'up' in key_5:
                    joints[4] = value_5
                    #print("wrist up = {}".format(rad2deg(value_5)))
                    joints[5] = get_theta6(df_robot, T, joints, modified=modified)
                    #print("t_6 = {}".format(rad2deg(joints[5])))
                    t_3 = get_theta3(df_robot, T, joints, modified=modified)
                    for key_3, value_3 in t_3.items():
                        if 'up' in key_3:
                            joints[2] = value_3
                            #print("elbow up = {}".format(rad2deg(value_3)))
                            joints[1] = get_theta2(df_robot, T, joints, modified=modified)
                            #print("t_2 = {}".format(rad2deg(joints[1])))
                            joints[3] = get_theta4(df_robot, T, joints, modified=modified)
                            #print("t_4 = {}".format(rad2deg(joints[3])))
                            results.loc[4, cols] = joints
                        if 'down' in key_3:
                            joints[2] = value_3
                            #print("elbow down = {}".format(rad2deg(value_3)))
                            joints[1] = get_theta2(df_robot, T, joints, modified=modified)
                            #print("t_2 = {}".format(rad2deg(joints[1])))
                            joints[3] = get_theta4(df_robot, T, joints, modified=modified)
                            #print("t_4 = {}".format(rad2deg(joints[3])))
                            results.loc[5, cols] = joints
                if 'down' in key_5:
                    joints[4] = value_5
                    #print("wrist down = {}".format(rad2deg(value_5)))
                    joints[5] = get_theta6(df_robot, T, joints, modified=modified)
                    #print("t_6 = {}".format(rad2deg(joints[5])))
                    t_3 = get_theta3(df_robot, T, joints, modified=modified)
                    for key_3, value_3 in t_3.items():
                        if 'up' in key_3:
                            joints[2] = value_3
                            #print("elbow up = {}".format(rad2deg(value_3)))
                            joints[1] = get_theta2(df_robot, T, joints, modified=modified)
                            #print("t_2 = {}".format(rad2deg(joints[1])))
                            joints[3] = get_theta4(df_robot, T, joints, modified=modified)
                            #print("t_4 = {}".format(rad2deg(joints[3])))
                            results.loc[6, cols] = joints
                        if 'down' in key_3:
                            joints[2] = value_3
                            #print("elbow down = {}".format(rad2deg(value_3)))
                            joints[1] = get_theta2(df_robot, T, joints, modified=modified)
                            #print("t_2 = {}".format(rad2deg(joints[1])))
                            joints[3] = get_theta4(df_robot, T, joints, modified=modified)
                            #print("t_4 = {}".format(rad2deg(joints[3])))
                            results.loc[7, cols] = joints

    return results

def transMat2posVec(T, degrees:bool=False):
    Rmat = T[0:3,0:3]
    pos = T[0:3,3]
    rot = rotMat2rotVec_sp(Rmat).T.flatten()
    if degrees:
        rot = np.rad2deg(rot)
    #print(rotMat2rotVec(Rmat))
    return np.hstack((pos, rot))

def posVec2transMat(posVec):
    pos = posVec[0:3].reshape(1,3)
    rot = posVec[3:6].reshape(1,3)
    Rmat = rotVec2rotMat(rot.T)
    return trans_mat(Rmat, pos.T)

def posVecDecomposition(posVec):
    pos = posVec[0:3].reshape(1,3)
    rot = posVec[3:6].reshape(1,3)
    return pos, rot

def FromXYZRPWToArray(xyzrpw):
    if len(xyzrpw) < 6:
        return None
    x,y,z,r,p,w = xyzrpw
    a = r * np.pi / 180.0
    b = p * np.pi / 180.0
    c = w * np.pi / 180.0
    ca = np.cos(a)
    sa = np.sin(a)
    cb = np.cos(b)
    sb = np.sin(b)
    cc = np.cos(c)
    sc = np.sin(c)
    return np.array([[cb * cc, cc * sa * sb - ca * sc, sa * sc + ca * cc * sb, x],
                    [cb * sc, ca * cc + sa * sb * sc, ca * sb * sc - cc * sa, y],
                    [-sb, cb * sa, ca * cb, z],
                    [0., 0., 0., 1.]]);

def FromURToArray(xyzwpr):
    if len(xyzwpr) < 6:
        return None
    x,y,z,w,p,r = xyzwpr
    angle = np.sqrt(w * w + p * p + r * r)
    if (angle < 1e-6):
        return np.identity(4)
    c = np.cos(angle)
    s = np.sin(angle)
    ux = w / angle
    uy = p / angle
    uz = r / angle
    return np.array([[ux * ux + c * (1 - ux * ux), ux * uy * (1 - c) - uz * s,  ux * uz * (1 - c) + uy * s, x],
                [ux * uy * (1 - c) + uz * s, uy * uy + (1 - uy * uy) * c, uy * uz * (1 - c) - ux * s, y],
                [ux * uz * (1 - c) - uy * s, uy * uz * (1 - c) + ux * s, uz * uz + (1 - uz * uz) * c, z],
                [0., 0., 0., 1.]]);

class UR_robot:
    dhp_ur3e = pd.DataFrame(np.array([[0, 0, 0.15185, np.pi/2],
                                   [0, -0.24355, 0, 0],
                                   [0, -0.2132, 0, 0],
                                   [0, 0, 0.13105, np.pi/2],
                                   [0, 0, 0.08535, -np.pi/2],
                                   [0, 0, 0.0921, 0]]),
                                   columns=['theta', 'a', 'd', 'alpha'])
    dhp_ur5 = pd.DataFrame(np.array([[0, 0, 0.089159, np.pi/2],
                                   [0, -0.425, 0, 0],
                                   [0, -0.3922, 0, 0],
                                   [0, 0, 0.10915, np.pi/2],
                                   [0, 0, 0.09465, -np.pi/2],
                                   [0, 0, 0.0823, 0]]),
                                   columns=['theta', 'a', 'd', 'alpha'])
    dhp_ur5e = pd.DataFrame(np.array([[0, 0, 0.1625, np.pi/2],
                                   [0, -0.425, 0, 0],
                                   [0, -0.3922, 0, 0],
                                   [0, 0, 0.1333, np.pi/2],
                                   [0, 0, 0.0997, -np.pi/2],
                                   [0, 0, 0.0996, 0]]),
                                   columns=['theta', 'a', 'd', 'alpha'])
    dhp_ur10e = pd.DataFrame(np.array([[0, 0, 0.1807, np.pi/2],
                                   [0, -0.6127, 0, 0],
                                   [0, -0.57155, 0, 0],
                                   [0, 0, 0.17415, np.pi/2],
                                   [0, 0, 0.11985, -np.pi/2],
                                   [0, 0, 0.11655, 0]]),
                                   columns=['theta', 'a', 'd', 'alpha'])
    dhp_ur16e = pd.DataFrame(np.array([[0, 0, 0.1807, np.pi/2],
                                   [0, -0.4784, 0, 0],
                                   [0, -0.36, 0, 0],
                                   [0, 0, 0.17415, np.pi/2],
                                   [0, 0, 0.11985, -np.pi/2],
                                   [0, 0, 0.11655, 0]]),
                                   columns=['theta', 'a', 'd', 'alpha'])
    mdhp_ur10e = pd.DataFrame(np.array([[0, 0, 0.1807, 0],
                                   [0, 0, 0, np.pi/2],
                                   [0, -0.6127, 0, 0],
                                   [0, -0.57155, 0.17415, 0],
                                   [0, 0, 0.11985, np.pi/2],
                                   [0, 0, 0.11655, -np.pi/2]]),
                                   columns=['theta', 'a_i-1', 'd', 'alpha_i-1'])
    m_ur10e = pd.DataFrame(np.array([7.369, 13.051, 3.989, 2.100, 1.980, 0.615]), 
                           index=["l1", "l2", "l3", "l4", "l5", "l6"], 
                           columns=['mass (kg)'])
    cm_ur10e = pd.DataFrame(np.array([[0.021, 0.000, 0.027], 
                         [0.38, 0.000, 0.158], 
                         [0.24, 0.000, 0.068], 
                         [0.000, 0.007, 0.018], 
                         [0.000, 0.007, 0.018], 
                         [0, 0, -0.026]]), 
                         index=["l1", "l2", "l3", "l4", "l5", "l6"], 
                         columns=["cm_x", "cm_y", "cm_z"])
    I_ur10e = pd.DataFrame(np.array([[0.03408, 0.00002, -0.00425, 0.00002, 0.03529, 0.00008, 0.00425, 0.00008, 0.02156], 
                                    [0.02814, 0.00005, -0.01561, 0.00005, 0.77068, 0.00002, -0.01561, 0.00002, 0.76943], 
                                    [0.01014, 0.00008, 0.00916, 0.00008, 0.30928, 0.00000, 0.00916, 0.00000, 0.30646], 
                                    [0.00296, -0.00001, -0.00000, -0.00001, 0.00222, -0.00024, -0.00000, -0.00024, 0.00258], 
                                    [0.00296, -0.00001, -0.00000, -0.00001, 0.00222, -0.00024, -0.00000, -0.00024, 0.00258], 
                                    [0.00040, 0.00000,  -0.00000, 0.00000, 0.00041, 0.00000, -0.00000, 0.00000, 0.00034]]), 
                         index=["l1", "l2", "l3", "l4", "l5", "l6"], 
                         columns=["Ixx", "Ixy", "Ixz", "Iyx", "Iyy", "Iyz", "Izx", "Izy", "Izz"])

    dhp = {'ur3e': dhp_ur3e, 'ur5': dhp_ur5,'ur5e': dhp_ur5e, 'ur10e': dhp_ur10e, 'ur16e': dhp_ur16e}
    mdhp = {'ur10e': mdhp_ur10e}
    mass = {'ur10e': m_ur10e}
    cm = {'ur10e': cm_ur10e}
    I = {'ur10e': I_ur10e}
    wc2flange = np.array([0., 35.7e-3, 0.1e-3, -np.deg2rad(30), 0., 0.])
    wsc2flange = np.array([0., 43.2e-3, 13.09e-3, -np.deg2rad(30), 0., 0.])
    # My original was:
    # tool2flange = np.array([0., 0., 50e-03, 0., 0., 0.])
    # Modified after reading the documentation at:
    # https://assets.robotiq.com/website-assets/support_documents/document/TCP_20and_20Center_20of_20Mass_20of_20Robotiq_20Products_20V1.1_20210406.pdf
    gripcam_tool2flange = np.array([0., 0., 175.5e-03, 0., 0., 0.])
    # My original was:
    # grip_tool2flange = np.array([0., 0., 58.08e-03, 0., 0., 0.])
    # Modified after reading the documentation at:
    # https://assets.robotiq.com/website-assets/support_documents/document/TCP_20and_20Center_20of_20Mass_20of_20Robotiq_20Products_20V1.1_20210406.pdf
    grip_tool2flange = np.array([0., 0., 174.0e-03, 0., 0., 0.])
    fix_frame = {'grip_cam': gripcam_tool2flange, 'gripper': grip_tool2flange, 'wrist_cam': wc2flange, 'wrist_sim_cam':wsc2flange}
    K1_wrist = np.array([[1.07067640e+03, 0.00000000e+00, 2.83730877e+02],
    [0.00000000e+00, 1.06788000e+03, 2.00106034e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    K2_wrist = np.array([[1.08386084e+03, 0.00000000e+00, 2.73007283e+02],
    [0.00000000e+00, 1.07276257e+03, 1.97336809e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    K3_wrist = np.array([[710.09092509,   0.        , 322.75370467],
    [0.        , 709.92654071, 206.88743337],
    [0.          , 0.          , 1.        ]])
    K4_wrist = np.array( [[718.90118408, 0., 327.12475112],
    [0., 713.69824219, 213.88042599],
    [0., 0., 1.]])
    K5_wrist = np.array([[706.01318626, 0., 328.16868712],
    [0., 706.28100808, 213.66257482],
    [0., 0., 1.]])

    H = np.array([[-1.4563106e-03, 0., 4.660194e-01],
                  [0., 1.4603175e-03, -9.4124603e-01],
                  [0., 0., 1.]])

    def __init__(self, name, type, linkto=None, modified_parameters=False, attach=[], home_pos = np.array([-69.49986954, -85.5911859 , -61.30664418, -94.68861433, 79.90856193,  17.94833866])):
        assert type in tuple(UR_robot.dhp.keys()), f"The robot: {type} hasn't been impĺemented in this library D:"
        self.name = name
        self.type = type
        if modified_parameters:
            self.parameters = UR_robot.mdhp[type]
        else:
            self.parameters = UR_robot.dhp[type]
        if self.type != 'ur10e':
            print("The dynamics of this robot hasn't been implemented yet :(")
        else:
            self.m = UR_robot.mass[type]
            self.cm = UR_robot.cm[type]
            self.I = UR_robot.I[type]
        self.Item = linkto
        self.attached = {key:UR_robot.fix_frame[key] for key in attach}
        self.home_pos = home_pos

    def is_attached(self, tool):
        if tool in self.attached.keys():
            return True
        else:
            return False
        
    def set_tools(self, tools: list = []):
        self.attached = {key:UR_robot.fix_frame[key] for key in tools}

# class UR10(Robot):
#     """
#     Class that imports a UR10 URDF model

#     ``UR3()`` is a class which imports a Universal Robotics UR310 robot
#     definition from a URDF file.  The model describes its kinematic and
#     graphical characteristics.

#     .. runblock:: pycon

#         >>> import roboticstoolbox as rtb
#         >>> robot = rtb.models.URDF.UR10()
#         >>> print(robot)

#     Defined joint configurations are:

#     - qz, zero joint angle configuration, 'L' shaped configuration
#     - qr, vertical 'READY' configuration

#     .. codeauthor:: Jesse Haviland
#     .. sectionauthor:: Peter Corke
#     """

#     def __init__(self, urdf: str='/home/daniel/github/robodk/Rocket_assembly/urdf/robots/ur10e_2f85.urdf'):

#         links, name, urdf_string, urdf_filepath = self.URDF_read(urdf)

#         super().__init__(
#             links,
#             name=name.upper(),
#             manufacturer="Universal Robotics",
#             gripper_links=links[10],
#             urdf_string=urdf_string,
#             urdf_filepath=urdf_filepath,
#         )

#         self.qr = np.array([np.pi, 0, 0, 0, np.pi / 2, 0])
#         self.qz = np.zeros(6)

#         self.addconfiguration("qr", self.qr)
#         self.addconfiguration("qz", self.qz)

# class rtb_UR_robot(DHRobot):
#     dhp_ur3e = pd.DataFrame(np.array([[0, 0, 0.15185, np.pi/2],
#                                    [0, -0.24355, 0, 0],
#                                    [0, -0.2132, 0, 0],
#                                    [0, 0, 0.13105, np.pi/2],
#                                    [0, 0, 0.08535, -np.pi/2],
#                                    [0, 0, 0.0921, 0]]),
#                                    columns=['theta', 'a', 'd', 'alpha'])
#     dhp_ur5 = pd.DataFrame(np.array([[0, 0, 0.089159, np.pi/2],
#                                    [0, -0.425, 0, 0],
#                                    [0, -0.3922, 0, 0],
#                                    [0, 0, 0.10915, np.pi/2],
#                                    [0, 0, 0.09465, -np.pi/2],
#                                    [0, 0, 0.0823, 0]]),
#                                    columns=['theta', 'a', 'd', 'alpha'])
#     dhp_ur5e = pd.DataFrame(np.array([[0, 0, 0.1625, np.pi/2],
#                                    [0, -0.425, 0, 0],
#                                    [0, -0.3922, 0, 0],
#                                    [0, 0, 0.1333, np.pi/2],
#                                    [0, 0, 0.0997, -np.pi/2],
#                                    [0, 0, 0.0996, 0]]),
#                                    columns=['theta', 'a', 'd', 'alpha'])
#     dhp_ur10e = pd.DataFrame(np.array([[0, 0, 0.1807, np.pi/2],
#                                    [0, -0.6127, 0, 0],
#                                    [0, -0.57155, 0, 0],
#                                    [0, 0, 0.17415, np.pi/2],
#                                    [0, 0, 0.11985, -np.pi/2],
#                                    [0, 0, 0.11655, 0]]),
#                                    columns=['theta', 'a', 'd', 'alpha'])
#     dhp_ur16e = pd.DataFrame(np.array([[0, 0, 0.1807, np.pi/2],
#                                    [0, -0.4784, 0, 0],
#                                    [0, -0.36, 0, 0],
#                                    [0, 0, 0.17415, np.pi/2],
#                                    [0, 0, 0.11985, -np.pi/2],
#                                    [0, 0, 0.11655, 0]]),
#                                    columns=['theta', 'a', 'd', 'alpha'])
#     mdhp_ur10e = pd.DataFrame(np.array([[0, 0, 0.1807, 0],
#                                    [0, 0, 0, np.pi/2],
#                                    [0, -0.6127, 0, 0],
#                                    [0, -0.57155, 0.17415, 0],
#                                    [0, 0, 0.11985, np.pi/2],
#                                    [0, 0, 0.11655, -np.pi/2]]),
#                                    columns=['theta', 'a_i-1', 'd', 'alpha_i-1'])
#     m_ur10e = pd.DataFrame(np.array([7.369, 13.051, 3.989, 2.100, 1.980, 0.615]), 
#                            index=["l1", "l2", "l3", "l4", "l5", "l6"], 
#                            columns=['mass (kg)'])
#     cm_ur10e = pd.DataFrame(np.array([[0.021, 0.000, 0.027], 
#                          [0.38, 0.000, 0.158], 
#                          [0.24, 0.000, 0.068], 
#                          [0.000, 0.007, 0.018], 
#                          [0.000, 0.007, 0.018], 
#                          [0, 0, -0.026]]), 
#                          index=["l1", "l2", "l3", "l4", "l5", "l6"], 
#                          columns=["cm_x", "cm_y", "cm_z"])
#     I_ur10e = pd.DataFrame(np.array([[0.03408, 0.00002, -0.00425, 0.00002, 0.03529, 0.00008, -0.00425, 0.00008, 0.02156], 
#                                     [0.02814, 0.00005, -0.01561, 0.00005, 0.77068, 0.00002, -0.01561, 0.00002, 0.76943], 
#                                     [0.01014, 0.00008, 0.00916, 0.00008, 0.30928, 0.00000, 0.00916, 0.00000, 0.30646], 
#                                     [0.00296, -0.00001, -0.00000, -0.00001, 0.00222, -0.00024, -0.00000, -0.00024, 0.00258], 
#                                     [0.00296, -0.00001, -0.00000, -0.00001, 0.00222, -0.00024, -0.00000, -0.00024, 0.00258], 
#                                     [0.00040, 0.00000,  -0.00000, 0.00000, 0.00041, 0.00000, -0.00000, 0.00000, 0.00034]]), 
#                          index=["l1", "l2", "l3", "l4", "l5", "l6"], 
#                          columns=["Ixx", "Ixy", "Ixz", "Iyx", "Iyy", "Iyz", "Izx", "Izy", "Izz"])

#     dhp = {'ur3e': dhp_ur3e, 'ur5': dhp_ur5,'ur5e': dhp_ur5e, 'ur10e': dhp_ur10e, 'ur16e': dhp_ur16e}
#     mdhp = {'ur10e': mdhp_ur10e}
#     mass = {'ur10e': m_ur10e}
#     cm = {'ur10e': cm_ur10e}
#     I = {'ur10e': I_ur10e}
#     dyna = {'ur10e': pd.concat([m_ur10e, cm_ur10e, I_ur10e], axis=1)}
#     wc2flange = np.array([0., 35.7e-3, 0.1e-3, -np.deg2rad(30), 0., 0.])
#     wsc2flange = np.array([0., 43.2e-3, 13.09e-3, -np.deg2rad(30), 0., 0.])
#     # My original was:
#     # tool2flange = np.array([0., 0., 50e-03, 0., 0., 0.])
#     # Modified after reading the documentation at:
#     # https://assets.robotiq.com/website-assets/support_documents/document/TCP_20and_20Center_20of_20Mass_20of_20Robotiq_20Products_20V1.1_20210406.pdf
#     gripcam_tool2flange = np.array([0., 0., 175.5e-03, 0., 0., 0.]) #TCP distance from robot flange to gripper fingertips in closed position (gripper+camera conf)
#     gripper_tool2flange = np.array([0., 0., 162.e-03, 0., 0., 0.]) #TCP distance from robot flange to gripper fingertips in closed position (only gripper conf)
#     grip_tool2flange = np.array([0., 0., 58.08e-03, 0., 0., 0.]) #TCP distance from gripper center of mass to gripper fingertips in closed position
#     fix_frame = {'grip_cam': gripcam_tool2flange, 'gripper': grip_tool2flange, 'wrist_cam': wc2flange, 'wrist_sim_cam':wsc2flange}
#     K1_wrist = np.array([[1.07067640e+03, 0.00000000e+00, 2.83730877e+02],
#     [0.00000000e+00, 1.06788000e+03, 2.00106034e+02],
#     [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
#     K2_wrist = np.array([[1.08386084e+03, 0.00000000e+00, 2.73007283e+02],
#     [0.00000000e+00, 1.07276257e+03, 1.97336809e+02],
#     [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
#     K3_wrist = np.array([[710.09092509,   0.        , 322.75370467],
#     [0.        , 709.92654071, 206.88743337],
#     [0.          , 0.          , 1.        ]])
#     K4_wrist = np.array( [[718.90118408, 0., 327.12475112],
#     [0., 713.69824219, 213.88042599],
#     [0., 0., 1.]])
#     K5_wrist = np.array([[706.01318626, 0., 328.16868712],
#     [0., 706.28100808, 213.66257482],
#     [0., 0., 1.]])

#     H = np.array([[-1.4563106e-03, 0., 4.660194e-01],
#                   [0., 1.4603175e-03, -9.4124603e-01],
#                   [0., 0., 1.]])

#     def __init__(self, name, type, linkto=None, modified_parameters=False, current_tool='grip_cam' ,attach=['grip_cam', 'gripper', 'wrist_cam']):
#         assert type in tuple(rtb_UR_robot.dhp.keys()), f"The robot: {type} hasn't been impĺemented in this library D:"

#         if modified_parameters:
#             parameters = rtb_UR_robot.mdhp[type]
#         else:
#             parameters = rtb_UR_robot.dhp[type]
#         self.Item = linkto
#         dynamic_parameters = rtb_UR_robot.dyna[type]
#         self.attached = {key:rtb_UR_robot.fix_frame[key] for key in attach}

#         a = parameters['a']
#         d = parameters['d']
#         alpha = parameters['alpha']
#         offset = parameters['theta']

#         mass = dynamic_parameters['mass (kg)']
#         center_of_mass = dynamic_parameters[dynamic_parameters.columns[[x.startswith("cm_") for x in dynamic_parameters.columns]]]
#         I_cm = dynamic_parameters[dynamic_parameters.columns[[x.startswith("I") for x in dynamic_parameters.columns]]]
#         # qlimits = dynamic_parameters['qlim']

#         links = []
#         tool = posVec2transMat(self.attached[current_tool]) if attach else None

#         if modified_parameters:
#             for j in range(6):
#                 link = RevoluteMDH(
#                     d=d[j], 
#                     a=a[j], 
#                     alpha=alpha[j], 
#                     offset=offset[j], 
#                     I=I_cm.iloc[j].to_numpy().reshape(3,3),
#                     m=mass[j], 
#                     r=center_of_mass.iloc[j].to_numpy(),  
#                     G=1
#                 )
#                 links.append(link)
#         else:
#             for j in range(6):
#                 link = RevoluteDH(
#                     d=d[j], 
#                     a=a[j], 
#                     alpha=alpha[j], 
#                     offset=offset[j],
#                     I=I_cm.iloc[j].to_numpy().reshape(3,3),
#                     m=mass[j], 
#                     r=center_of_mass.iloc[j].to_numpy(), 
#                     G=1
#                 )
#                 links.append(link)

#         super().__init__(
#             links,
#             name=name,
#             manufacturer="Universal Robots",
#             gravity=[0., 0., -9.81],
#             tool=tool
#         )

#         self.qr = np.deg2rad(np.array([-72.017519, -83.917726, -66.369657, -90.939348, 81.120141, 15.700460]))
#         self.qz = np.zeros(6)

#         self.addconfiguration("qr", self.qr)
#         self.addconfiguration("qz", self.qz)

#     def is_attached(self, tool):
#         if tool in self.attached:
#             return True
#         else:
#             print(self.tool)
#             return False
        
# class XArm_robot(DHRobot):
#     dhp_lite6 = pd.DataFrame(np.array([[0, 0, 0.2433, -np.pi/2],
#                                    [-np.pi/2, 0.200, 0, np.pi],
#                                    [-np.pi/2, 0.087, 0, np.pi/2],
#                                    [0, 0, 0.2276, np.pi/2],
#                                    [0, 0, 0, -np.pi/2],
#                                    [0, 0, 0.0615, 0]]),
#                                    columns=['theta', 'a', 'd', 'alpha'])

#     mdhp_lite6 = pd.DataFrame(np.array([[0, 0, 0.2433, 0],
#                                    [-np.pi/2, 0, 0, -np.pi/2],
#                                    [-np.pi/2, 0.200, 0, np.pi],
#                                    [0, 0.087, 0.2276, np.pi/2],
#                                    [0, 0, 0, np.pi/2],
#                                    [0, 0, 0.0615, -np.pi/2]]),
#                                    columns=['theta', 'a', 'd', 'alpha'])
    
#     dyna_lite6 = pd.DataFrame([[1.411, np.array([-0.36e-03, 41.95e-03, -2.5e-03]), np.array([-6.283, 6.283])],
#                                [1.340, np.array([179.0e-03, 0., 58.4e-03]), np.array([-2.618, 2.618])],
#                                [0.953, np.array([72.0e-03, -35.7e-03, -1.0e-03]), np.array([-0.061, 5.235])],
#                                [1.284, np.array([-2.0e-03, -28.5e-03, -81.3e-03]), np.array([-6.283, 6.283])],
#                                [0.804, np.array([0., 10.e-03, 1.9e-03]), np.array([-2.164, 2.164])],
#                                [0.130, np.array([0., -1.94e-03, -10.2e-03]), np.array([-6.283, 6.283])]],
#                                columns=['m', 'cm', 'qlim'])

#     dhp = {'lite6': dhp_lite6}
#     mdhp = {'lite6': mdhp_lite6}
#     dof = {'lite6': 6}
#     dynamic = {'lite6': dyna_lite6}
#     wc2flange = np.array([0., 35.7e-3, 0.1e-3, -np.deg2rad(30), 0., 0.])
#     wsc2flange = np.array([0., 43.2e-3, 13.09e-3, -np.deg2rad(30), 0., 0.])
#     # My original was:
#     # tool2flange = np.array([0., 0., 50e-03, 0., 0., 0.])
#     # Modified after reading the documentation at:
#     # https://assets.robotiq.com/website-assets/support_documents/document/TCP_20and_20Center_20of_20Mass_20of_20Robotiq_20Products_20V1.1_20210406.pdf
#     gripcam_tool2flange = np.array([0., 0., 175.5e-03, 0., 0., 0.])
#     grip_tool2flange = np.array([0., 0., 58.08e-03, 0., 0., 0.])
#     fix_frame = {'grip_cam': gripcam_tool2flange, 'gripper': grip_tool2flange, 'wrist_cam': wc2flange, 'wrist_sim_cam':wsc2flange}
    
#     def __init__(self, name, type, linkto=None, modified_parameters=False, attach=[]):
#         if modified_parameters:
#             parameters = XArm_robot.mdhp[type]
#         else:
#             parameters = XArm_robot.dhp[type]
#         self.Item = linkto
#         dynamic_parameters = XArm_robot.dynamic[type]
#         attached = {key:XArm_robot.fix_frame[key] for key in attach}

#         a = parameters['a']
#         d = parameters['d']
#         alpha = parameters['alpha']
#         offset = parameters['theta']

#         mass = dynamic_parameters['m']
#         center_of_mass = dynamic_parameters['cm']
#         qlimits = dynamic_parameters['qlim']

#         links = []

#         if modified_parameters:
#             for j in range(6):
#                 link = RevoluteMDH(
#                     d=d[j], 
#                     a=a[j], 
#                     alpha=alpha[j], 
#                     offset=offset[j], 
#                     qlim=qlimits[j],
#                     m=mass[j], 
#                     r=center_of_mass[j], 
#                     G=1
#                 )
#                 links.append(link)
#         else:
#             for j in range(6):
#                 link = RevoluteDH(
#                     d=d[j], 
#                     a=a[j], 
#                     alpha=alpha[j], 
#                     offset=offset[j],
#                     qlim=qlimits[j], 
#                     m=mass[j], 
#                     r=center_of_mass[j], 
#                     G=1
#                 )
#                 links.append(link)

#         super().__init__(
#             links,
#             name=name,
#             manufacturer="UFactory",
#         )

#         self.qr = np.array([0, -0.3, 0, -2.2, 0, 2.0])
#         self.qz = np.zeros(6)

#         self.addconfiguration("qr", self.qr)
#         self.addconfiguration("qz", self.qz)

#     def is_attached(self, tool):
#         if tool in self.attached:
#             return True
#         else:
#             return False

if __name__ == '__main__':
    print('This will get executed only if')
    print('the module is invoked directly.')
    print('It will not run when this module is imported')
    #my_function()
