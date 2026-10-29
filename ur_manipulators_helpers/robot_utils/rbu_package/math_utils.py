import numpy as np 
import scipy as spy
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.signal import argrelextrema
import scipy.sparse as sparse
from rdp import rdp
import pandas as pd
import cv2
import random
from scipy.spatial import distance as dist
from dtw import dtw
from scipy.integrate import simpson as simps
import os
from typing import Tuple, Dict, Any
from collections import deque
from sklearn.metrics import root_mean_squared_error
from scipy.spatial.distance import cdist # For efficient distance calculations

class TrajectoryComparator:
    """
    Analyzes, compares, and logs 3D trajectory data for robotics experiments.
    """
    def __init__(self, results_filepath='trajectory_comparison_results.csv'):
        """
        Initializes the comparator with the path to the results file.

        Args:
            results_filepath (str): The path to the CSV file where results will be stored.
        """
        self.filepath = results_filepath
        self.df = self._load_or_create_df()

    def _calculate_metrics(self, base_pos, comp_pos, base_vel, comp_vel, base_acc, comp_acc, dt):
        """Calculates a dictionary of metrics between two trajectories."""
        # Ensure trajectories are the same length for direct comparison
        min_len = min(len(base_pos), len(comp_pos))
        base_pos, comp_pos = base_pos[:min_len], comp_pos[:min_len]
        base_vel, comp_vel = base_vel[:min_len], comp_vel[:min_len]
        base_acc, comp_acc = base_acc[:min_len], comp_acc[:min_len]

        # Position Metrics
        pos_error = np.linalg.norm(base_pos - comp_pos, axis=1)
        # Velocity Metrics
        vel_error = np.linalg.norm(base_vel - comp_vel, axis=1)
        # Acceleration Metrics
        acc_error = np.linalg.norm(base_acc - comp_acc, axis=1)
        
        # Path Length
        path_length = np.sum(np.linalg.norm(np.diff(comp_pos, axis=0), axis=1))
        
        # Jerk (Smoothness)
        jerk = np.linalg.norm(np.diff(comp_acc, n=1, axis=0) / dt, axis=1)
        total_jerk = simps(jerk**2, dx=dt)
        
        # Dynamic Time Warping (for position similarity, ignoring time alignment)
        # The 'distance' is the key output from DTW.
        dtw_distance, _, _, _ = dtw(base_pos, comp_pos, dist=lambda x, y: np.linalg.norm(x - y))
        
        metrics = {
            'pos_rmse': np.sqrt(np.mean(pos_error**2)),
            'pos_mae': np.mean(pos_error),
            'pos_peak_error': np.max(pos_error),
            'vel_rmse': np.sqrt(np.mean(vel_error**2)),
            'acc_rmse': np.sqrt(np.mean(acc_error**2)),
            'path_length': path_length,
            'smoothness_jerk': total_jerk,
            'dtw_distance': dtw_distance,
        }
        return metrics

    def run_comparison(self, baseline, computed, executed, dt, exp_params=None):
        """
        Performs a full comparison and logs it.

        Args:
            baseline (tuple): (pos, vel, acc) arrays for the original trajectory.
            computed (tuple): (pos, vel, acc) arrays for the avoidance trajectory.
            executed (tuple): (pos, vel, acc) arrays for the robot's actual trajectory.
            dt (float): The time step between trajectory points.
            exp_params (dict, optional): A dictionary of parameters for this specific experiment.
        """
        # Comparison 1: Computed vs. Baseline (How much did the planner deviate?)
        comp_vs_base = self._calculate_metrics(baseline[0], computed[0], baseline[1], computed[1], baseline[2], computed[2], dt)
        
        # Comparison 2: Executed vs. Computed (How well did the robot follow the plan?)
        exec_vs_comp = self._calculate_metrics(computed[0], executed[0], computed[1], executed[1], computed[2], executed[2], dt)

        # Flatten results into a single dictionary for logging
        log_data = {}
        if exp_params:
            log_data.update(exp_params)
            
        for key, val in comp_vs_base.items():
            log_data[f'comp_vs_base_{key}'] = val
        for key, val in exec_vs_comp.items():
            log_data[f'exec_vs_comp_{key}'] = val
            
        self._log_to_file(log_data)
        print("Successfully calculated and logged metrics for the latest experiment.")

    def _load_or_create_df(self):
        """Loads the results dataframe from file or creates a new one."""
        if os.path.exists(self.filepath):
            return pd.read_csv(self.filepath)
        return pd.DataFrame()

    def _log_to_file(self, log_data):
        """Appends a new row of data to the dataframe and saves to CSV."""
        new_row = pd.DataFrame([log_data])
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        self.df.to_csv(self.filepath, index=False)

class MovingAverageFilter3D:
    """
    A Moving Average Filter for 3D data points.
    Each data point is expected to be a NumPy array of shape (3, 1).
    """
    def __init__(self, window_size: int):
        """
        Initializes the Moving Average Filter.

        Args:
            window_size (int): The number of past data points to include in the average.
                               Must be a positive integer.
        """
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("Window size must be a positive integer.")
        self.window_size = window_size
        # The deque will store the last 'window_size' data points (each a (3,1) np.array)
        self.data_window = deque(maxlen=window_size)

    def update(self, new_value: np.ndarray) -> np.ndarray:
        """
        Updates the filter with a new 3D data point and returns the smoothed value.

        Args:
            new_value (np.ndarray): The new data point, a NumPy array of shape (3, 1).

        Returns:
            np.ndarray: The smoothed data point, a NumPy array of shape (3, 1).
        """
        if not isinstance(new_value, np.ndarray) or new_value.shape != (3,):
            raise ValueError("Input new_value must be a NumPy array of shape (3, 1).")
        
        self.data_window.append(new_value)
        
        # Calculate the mean of the arrays currently in the window.
        # np.array(list(self.data_window)) creates a NumPy array of shape 
        # (current_window_length, 3, 1).
        # np.mean along axis=0 averages over the samples in the window.
        if not self.data_window:
             # This case should ideally not be reached if window_size > 0 and update is called.
             # However, as a fallback, return NaNs or the new_value itself if window is empty.
            return np.full((3,), np.nan) 
        
        # Convert deque of (3,1) arrays to a single NumPy array of shape (N, 3, 1)
        # where N is the current number of items in the deque.
        window_as_array = np.array(list(self.data_window))
        
        # Calculate mean along the first axis (the "samples" axis)
        smoothed_value = np.mean(window_as_array, axis=0)
        
        return smoothed_value # Shape will be (3,1)

class ExponentialMovingAverage3D:
    """
    An Exponential Moving Average Filter for 3D data points.
    Each data point is expected to be a NumPy array of shape (3, 1).
    """
    def __init__(self, alpha: float = 0.3):
        """
        Initializes the Exponential Moving Average Filter.

        Args:
            alpha (float, optional): The smoothing factor, between 0 (exclusive) 
                                     and 1 (inclusive). A smaller alpha results in
                                     more smoothing but more lag. Defaults to 0.3.
        """
        if not (isinstance(alpha, float) and 0.0 < alpha <= 1.0):
            raise ValueError("Alpha must be a float between 0 (exclusive) and 1 (inclusive).")
        self.alpha = alpha
        self.ema_previous = None # Stores the previous EMA value, a (3,1) np.array

    def update(self, new_value: np.ndarray) -> np.ndarray:
        """
        Updates the filter with a new 3D data point and returns the smoothed value.

        Args:
            new_value (np.ndarray): The new data point, a NumPy array of shape (3, 1).

        Returns:
            np.ndarray: The smoothed data point, a NumPy array of shape (3, 1).
        """
        if not isinstance(new_value, np.ndarray) or new_value.shape != (3,):
            raise ValueError("Input new_value must be a NumPy array of shape (3, 1).")

        if self.ema_previous is None:
            # Initialize with the first value
            self.ema_previous = new_value.copy() 
            return self.ema_previous
        
        # Apply EMA formula: EMA_current = alpha * new_value + (1 - alpha) * EMA_previous
        ema_current = (new_value * self.alpha) + (self.ema_previous * (1 - self.alpha))
        self.ema_previous = ema_current # Update for the next iteration
        
        return ema_current

def generate_interception_trajectory(original_trajectory, obstacle_start_pos, object_speed=1.0):
    """
    Calculates the core interception data: point, time, and path.
    This is a helper function for the main timestep generator.
    """
    # Using numpy arrays makes the calculations dimension-agnostic
    original_trajectory_np = [np.array(p) for p in original_trajectory]
    obstacle_start_pos_np = np.array(obstacle_start_pos)

    # 1. Calculate the total length of the original trajectory
    total_length = 0
    segment_lengths = []
    for i in range(len(original_trajectory_np) - 1):
        segment_length = np.linalg.norm(original_trajectory_np[i+1] - original_trajectory_np[i])
        segment_lengths.append(segment_length)
        total_length += segment_length

    if total_length == 0:
        raise ValueError("Original trajectory has zero length.")

    # 2. Determine the interception distance
    lower_bound = total_length / 3.0
    upper_bound = 2 * total_length / 3.0
    interception_distance = random.uniform(lower_bound, upper_bound)

    # 3. Find the interception point on the original trajectory
    current_length = 0
    interception_point_np = None
    for i in range(len(segment_lengths)):
        if current_length + segment_lengths[i] >= interception_distance:
            remaining_dist = interception_distance - current_length
            direction_vector = (original_trajectory_np[i+1] - original_trajectory_np[i]) / segment_lengths[i]
            interception_point_np = original_trajectory_np[i] + remaining_dist * direction_vector
            break
        current_length += segment_lengths[i]

    # 4. Calculate the time to interception
    if object_speed <= 0:
        object_speed = 1.0  # Ensure speed is positive to avoid division by zero
    time_to_intercept = interception_distance / object_speed
    
    return interception_point_np, time_to_intercept, obstacle_start_pos_np

def generate_interception_with_timesteps(original_trajectory, obstacle_start_pos, object_speed=1.0, time_step=1.0):
    """
    Generates the obstacle's trajectory with positions at specified time steps.

    Args:
        original_trajectory (list of tuples): Path of the original object.
        obstacle_start_pos (tuple): Starting coordinates of the obstacle.
        object_speed (float, optional): Constant speed of the object. Defaults to 1.0.
        time_step (float, optional): The time interval for reporting obstacle position. Defaults to 1.0.

    Returns:
        tuple: A tuple containing:
            - timed_trajectory (list of tuples): A list of (time, position) for the obstacle.
            - interception_point (tuple): The point of interception.
            - time_to_intercept (float): The total time until interception.
    """
    # Get the core interception data
    interception_point_np, time_to_intercept, obstacle_start_pos_np = generate_interception_trajectory(
        original_trajectory, obstacle_start_pos, object_speed
    )

    timed_trajectory = []
    
    if time_to_intercept == 0:
        # Obstacle starts at the interception point, so it doesn't move.
        timed_trajectory.append((0.0, np.array(obstacle_start_pos_np)))
        return timed_trajectory, np.array(interception_point_np), time_to_intercept

    # 5. Calculate the obstacle's constant velocity vector
    obstacle_velocity_vector = (interception_point_np - obstacle_start_pos_np) / time_to_intercept

    # 6. Generate positions at each time step
    current_time = 0.0
    while current_time < time_to_intercept:
        position = obstacle_start_pos_np + obstacle_velocity_vector * current_time
        timed_trajectory.append((current_time, np.array(position)))
        current_time += time_step

    # 7. Ensure the final interception point is the last entry
    timed_trajectory.append((time_to_intercept, np.array(interception_point_np)))

    return timed_trajectory, np.array(interception_point_np), time_to_intercept

def custom_trajectory_generator(path, polyfit= False, p_degree=5, num_output_points = 100, offset=0.15):
    """
    Generates a trajectory from a given path using polynomial fitting.
    
    Parameters:
        path (np.ndarray): The input path as an array of shape (n_points, n_dim).
        polyfit (bool): Whether to use polynomial fitting. Default is False.
        degree (int): Degree of the polynomial if polyfit is True. Default is 5.
    
    Returns:
        np.ndarray: The generated trajectory.
    """
    # You might want to choose a start_point_index strategically,
    # e.g., the point with the smallest x-value, or just default to 0.
    # For this example, let's find the point closest to the origin to start, or just pick one.
    # start_idx = np.argmin(np.linalg.norm(unordered_points, axis=1)) # Example: point closest to origin
    start_idx = 0 # Or simply pick index 1: [0.0, 0.0, 1.0]

    # Read the CSV file into a Pandas DataFrame
    df = pd.read_csv(path)
     # Convert the relevant columns to a NumPy array
    # If your DataFrame has columns named 'x', 'y', 'z':
    if all(col in df.columns for col in ['x', 'y', 'z']):
        points_collection = df[['x', 'y', 'z']].to_numpy()
    else:
        # If no specific column names or just want all columns as points
        points_collection = df.to_numpy()
        print(f"Warning: Assuming all columns in the CSV form the point coordinates. Shape: {points_collection.shape}")

    if offset > 0.:
        points_collection[:,2] = offset
    print("Points loaded using Pandas:")
    print(points_collection)
    print(f"Shape: {points_collection.shape}")

    # trajectory_greedy_nn = order_points_greedy_nn(points_collection)

    # print("\nGenerated trajectory from unordered points (greedy nearest-neighbor):")
    # print(f"Starting from point index {start_idx} (original coordinates: {points_collection[start_idx]})")
    # print(trajectory_greedy_nn)
    # print(f"Shape: {trajectory_greedy_nn.shape}")

    if polyfit:
        # print(f"Original trajectory shape: {trajectory_greedy_nn.shape}")
        print(f"Fitting with polynomial degree: {p_degree}")

        # Perform smoothing
        trajectory = smooth_trajectory_polynomial_fit(
            points_collection,
            degree=p_degree,
            num_smooth_points=num_output_points
        )
        print(f"Smoothed trajectory shape: {trajectory.shape}")
    else:
        trajectory = points_collection

    velocity_grad, acceleration_grad = get_derivatives_gradient(trajectory)

    return trajectory, velocity_grad, acceleration_grad

def get_derivatives_gradient(data):
    """
    Calculates first and second derivatives using numpy.gradient.
    This method is very fast and suitable for clean, well-sampled data.
    """
    # axis=0 ensures the gradient is computed for each column (x, y, z) independently.
    # The third argument, `edge_order=2`, can provide more accurate boundary derivatives.
    velocity = np.gradient(data, axis=0)
    acceleration = np.gradient(velocity, axis=0)
    return velocity, acceleration

def order_points_greedy_nn(points_array: np.ndarray, start_point_index: int = 0) -> np.ndarray:
    """
    Orders a set of points using a greedy nearest-neighbor approach.

    Args:
        points_array (np.ndarray): An array of shape (N, D) where N is the number
                                   of points and D is their dimension.
        start_point_index (int): The index of the point to start the trajectory from.

    Returns:
        np.ndarray: An array of shape (N, D) with points ordered to form a trajectory.
    """
    num_points = points_array.shape[0]
    if num_points < 2:
        return points_array.copy()

    # Ensure start_point_index is valid
    if not (0 <= start_point_index < num_points):
        raise ValueError(f"start_point_index {start_point_index} is out of bounds for {num_points} points.")

    ordered_indices = [start_point_index]
    remaining_indices = list(range(num_points))
    remaining_indices.pop(start_point_index) # More robust way to remove

    current_original_index = start_point_index

    while remaining_indices:
        current_point_coords = points_array[current_original_index:current_original_index+1] # Shape (1, D)
        remaining_points_coords = points_array[remaining_indices] # Shape (num_remaining, D)

        # Calculate distances from the current point to all remaining points
        distance_matrix_slice = cdist(current_point_coords, remaining_points_coords)
        distances_to_remaining = distance_matrix_slice[0] # Shape (num_remaining,)

        # Find the index of the nearest neighbor *within the remaining_points_coords array*
        idx_in_remaining_coords = np.argmin(distances_to_remaining)
        
        # Get the original index of this nearest neighbor
        next_original_index = remaining_indices.pop(idx_in_remaining_coords)
        
        ordered_indices.append(next_original_index)
        current_original_index = next_original_index
        
    return points_array[ordered_indices]

def smooth_trajectory_polynomial_fit(
    trajectory: np.ndarray,
    degree: int,
    num_smooth_points: int = None
) -> np.ndarray:
    """
    Smooths a 3D trajectory by fitting a polynomial to each coordinate.

    Args:
        trajectory (np.ndarray): The input trajectory, an array of shape (N, 3)
                                 where N is the number of points.
        degree (int): The degree of the polynomial to fit. Must be less than N.
        num_smooth_points (int, optional): The number of points to generate for
                                           the smoothed trajectory. If None, uses
                                           the same number of points as the input
                                           trajectory, evaluating at original
                                           parameterization points. Defaults to None.

    Returns:
        np.ndarray: The smoothed trajectory, an array of shape (M, 3) where M is
                    num_smooth_points or N if num_smooth_points is None.
                    Returns the original trajectory or an empty array if fitting
                    is not possible (e.g., too few points).
    """
    N = trajectory.shape[0]
    if trajectory.ndim != 2 or trajectory.shape[1] != 3:
        raise ValueError("Trajectory must be a NumPy array of shape (N, 3).")

    if N == 0:
        return np.empty((0, 3), dtype=trajectory.dtype)
    
    if N == 1: # Single point trajectory
        output_N = num_smooth_points if num_smooth_points is not None else 1
        return np.tile(trajectory[0], (output_N, 1))

    # Ensure degree is valid
    if degree >= N:
        print(f"Warning: Polynomial degree ({degree}) is too high for {N} points. "
              f"Adjusting degree to {N - 1}.")
        degree = N - 1
    if degree < 0: # Should not happen if N >= 1
        print("Error: Polynomial degree cannot be negative. Returning original trajectory.")
        return trajectory.copy()


    # 1. Parameterize the Trajectory using normalized cumulative arc length
    # Calculate distances between consecutive points
    diffs = np.diff(trajectory, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)

    # Calculate cumulative arc length
    # parameter_t[0] = 0
    # parameter_t[i] = sum of lengths of segments 0 to i-1
    parameter_t = np.zeros(N)
    parameter_t[1:] = np.cumsum(segment_lengths)

    # Handle cases where all points are identical (total length is zero)
    if parameter_t[-1] < 1e-9: # Effectively zero total length
        print("Warning: All points in the trajectory are (nearly) identical. "
              "The 'smoothed' trajectory will consist of this repeated point.")
        output_N = num_smooth_points if num_smooth_points is not None else N
        return np.tile(trajectory[0], (output_N, 1))
    
    # Normalize the parameter to be in the range [0, 1]
    parameter_t_normalized = parameter_t / parameter_t[-1]

    # 2. Fit Polynomials for each coordinate
    coeffs_x = np.polyfit(parameter_t_normalized, trajectory[:, 0], degree)
    coeffs_y = np.polyfit(parameter_t_normalized, trajectory[:, 1], degree)
    coeffs_z = np.polyfit(parameter_t_normalized, trajectory[:, 2], degree)

    # 3. Create polynomial functions
    poly_x = np.poly1d(coeffs_x)
    poly_y = np.poly1d(coeffs_y)
    poly_z = np.poly1d(coeffs_z)

    # 4. Evaluate Polynomials to get smoothed coordinates
    if num_smooth_points is None:
        # Evaluate at the original parameterization points for direct comparison
        t_eval = parameter_t_normalized
    else:
        # Evaluate over a new, evenly spaced set of parameter values
        t_eval = np.linspace(0, 1, num_smooth_points)

    x_smooth = poly_x(t_eval)
    y_smooth = poly_y(t_eval)
    z_smooth = poly_z(t_eval)

    # 5. Combine Coordinates
    smoothed_trajectory = np.vstack([x_smooth, y_smooth, z_smooth]).T
    
    return smoothed_trajectory

def compute_D1(n, dt):
    '''
    Compute the matrices used to estimate the first derivative.
      n float  : dimensionality
      dt float : timestep
    '''

    d1_p = np.ones([n - 1])
    d1_p[0] = 4.
    d1_m = - np.ones([n-1])
    d1_m[-1] = - 4.
    D1 = sparse.diags((d1_p, d1_m), [1, -1]).toarray()
    D1[0,0] = - 3.
    D1[0, 2] = -1.
    D1[-1, -3] = 1.
    D1[-1,-1] = 3.
    D1 /= 2 * dt
    return D1

def compute_D2(n, dt):
    '''
    Compute the matrices used to estimate the first derivative.
      n float  : dimensionality
      dt float : timestep
    '''

    d2_p = np.ones([n-1])
    d2_p[0] = -5.
    d2_m = np.ones([n-1])
    d2_m[-1] = -5.
    d2_c = - 2. * np.ones([n])
    d2_c[0] = 2.
    d2_c[-1] = 2.
    D2 = sparse.diags((d2_p, d2_c, d2_m), [1, 0, -1]).toarray()
    D2[0, 2] = 4.
    D2[0, 3] = -1.
    D2[-1, -3] = 4.
    D2[-1, -4] = -1.
    D2 /= dt ** 2
    return D2

# def curve_max_min(curve):
#     c = np.copy(curve)
#     max_idx_vn = argrelextrema(c, np.greater, order=10)
#     max_values_vn = c[max_idx_vn]
#     max_ind_top_vn = np.argsort(-max_values_vn)
#     max_top_vn = np.sort(max_values_vn)[::-1]
#     # print(max_idx_n[0][max_ind_top_n], max_top_n, max_ind_top_n)
#     min_idx_vn = argrelextrema(c, np.less, order=10)
#     min_values_vn = c[min_idx_vn]
#     min_ind_top_vn = np.argsort(min_values_vn)
#     min_top_vn = np.sort(min_values_vn)
#     return [max_idx_vn, max_ind_top_vn, max_top_vn], [min_idx_vn, min_ind_top_vn, min_top_vn]

def maxmin_func_norm(f, norm=True):
    if norm:
        a_norm = np.linalg.norm(f, axis=1)
    else:
        a_norm = f
    max_idx_n = argrelextrema(a_norm, np.greater, order=10)
    max_values_n = a_norm[max_idx_n]
    max_ind_top_n = np.argsort(-max_values_n)
    max_top_n = np.sort(max_values_n)[::-1]
    # print(max_idx_n[0][max_ind_top_n], max_top_n, max_ind_top_n)
    min_idx_n = argrelextrema(a_norm, np.less, order=10)
    min_values_n = a_norm[min_idx_n]
    min_ind_top_n = np.argsort(min_values_n)
    min_top_n = np.sort(min_values_n)
    max_values = zip(max_idx_n[0][max_ind_top_n[:1]], max_top_n[:1])
    min_values = zip(min_idx_n[0][min_ind_top_n[:1]], min_top_n[:1])

    return max_values, min_values

def mse_similarity(y1, y2):
    if np.isnan(np.mean(y1)) or np.isnan(np.mean(y2)):
        print("NaN values detected in input arrays.")
        mse = np.nanmean((y1 - y2) ** 2)

        mean_y1 = np.nanmean(y1)
        mean_y2 = np.nanmean(y2)

        y_1 = np.nan_to_num(y1, nan=mean_y1)
        y_2 = np.nan_to_num(y2, nan=mean_y2)
        rmse = root_mean_squared_error(y_1, y_2)
        return np.array([100 * (1 - mse / np.max((y1 - y1.mean()) ** 2)), 100 * (1 - rmse)])  # Convert to a percentage

    mse = np.nanmean((y1 - y2) ** 2)
    rmse = root_mean_squared_error(y1, y2)
    return np.array([100 * (1 - mse / np.max((y1 - y1.mean()) ** 2)), 100 * (1 - rmse)])  # Convert to a percentage


def rmse(actual, pred, ax=None): 
    actual, pred = np.array(actual), np.array(pred)
    return np.sqrt(np.square(np.subtract(actual,pred)).mean(axis=ax))

def derivative(f, x, epsilon=1e-10):
    """
    Calculate derivative of f at x
    """

    x_ = x + epsilon
    value = (f(x_) - f(x)) / epsilon

    return value

def partial_derivative(f, x, i, epsilon = 1e-10):
    """
    Calculate partial derivative of f at xi
    """
    x_ = np.copy(x).astype(np.float64)
    x_[i] = x_[i] + epsilon
    value = (f(x_) - f(x)) / epsilon

    return value

def jacobian(f, x, epsilon = 1e-10):
    """
    Calculate Jacobian of f wrt x
    """
    f_ = f(x)
    value = np.zeros((len(f_), len(x)))

    for i in range(len(x)):
        f_ = partial_derivative(f, x, i, epsilon)
        value[:,i] = f_

    return value

def newton_method(f, x_init = 0, epsilon = 1e-10):
    """
    Newton Raphson Optimizer
    """
    prev_value = x_init + 2 * epsilon
    value = x_init

    iterations = 0
    while np.abs(prev_value -value) > epsilon:
        prev_value = value

        f_dash = derivative(f, value)
        value = value - f(value) / f_dash

        iterations += 1
    
    print(f"Newton method converged in {iterations} iterations")

    return value

def newton_method_vector(f, n, x_init = np.array([]), epsilon = 1e-10, max_iterations = 1e06):
    """
    Newton Raphson Optimizer for Vector Functions

    ...

    Parameters
    ---
    f: Function to calculate root for
    n: Input vector size of f
    x_init(optional): Initial value of x
    epsilon(optional): Adjustable precision
    max_iterations(optinonal): Max number of iterations to run
    Returns
    ---
    x: Value of root
    """
    if x_init.size == 0:
        x_init = np.zeros(n)

    prev_value = x_init + 2 * epsilon
    value = x_init

    iterations = 0
    while np.all(np.abs(prev_value -value) > epsilon) and iterations < max_iterations:
        prev_value = value

        J = jacobian(f, value)
        value = value - np.dot(np.linalg.pinv(J), f(value))

        iterations += 1
    
    print(f"Newton method converged in {iterations} iterations")

    return value

def add_gaussian_noise(array, mean=0.0, std_dev=0.01):
  """
  Adds Gaussian noise to a NumPy array.

  Args:
    array (np.ndarray): The input array.
    mean (float): The mean of the normal distribution. Defaults to 0.0.
    std_dev (float): The standard deviation of the normal distribution.
                     This controls the magnitude of the noise. Defaults to 0.01.

  Returns:
    np.ndarray: The array with added noise.
  """
  noise = np.random.normal(mean, std_dev, size=array.shape)
  return array + noise