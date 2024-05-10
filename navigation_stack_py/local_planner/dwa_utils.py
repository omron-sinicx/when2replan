from __future__ import annotations

from typing import Tuple
import numpy as np
import numba


@numba.jit(nopython=True, cache=True, fastmath=True)
def compute_new_vel(
    current_vel: np.ndarray, target_vel: np.ndarray, acc_limits: np.ndarray, dt: float
) -> np.ndarray:
    """
    Compute new robot velocity based on current velocity, target velocity and acceleration limits.
    Args:
        current_vel: current robot velocity [vx, vy, w]
        target_vel: target robot velocity [vx, vy, w]
        acc_limits: acceleration limits [x, y, theta], (3,)
        dt: time interval
    Returns:
        new_vel: new robot velocity [vx, vy, w]
    """

    new_vel: np.ndarray = np.zeros(3)

    for i in range(3):
        if current_vel[i] < target_vel[i]:
            new_vel[i] = min(target_vel[i], current_vel[i] + acc_limits[i] * dt)
        elif current_vel[i] > target_vel[i]:
            new_vel[i] = max(target_vel[i], current_vel[i] - acc_limits[i] * dt)
        else:
            new_vel[i] = current_vel[i]

    return new_vel


@numba.jit(nopython=True, cache=True, fastmath=True)
def motion_model(current_pose: np.ndarray, vel: np.ndarray, dt: float) -> np.ndarray:
    """
    Compute new robot pose based on current pose and velocity.
    Args:
        current_pose: current robot pose [x, y, theta] in global coordinate
        vel: current robot velocity [vx, vy, w] in robot coordinate
        dt: time interval
    Returns:
        new_pose: new robot pose [x, y, theta]
    """
    xy = current_pose[0:2]
    rot_mat = np.array(
        [
            [np.cos(current_pose[2]), -np.sin(current_pose[2])],
            [np.sin(current_pose[2]), np.cos(current_pose[2])],
        ]
    )
    new_xy = xy + rot_mat.dot(vel[0:2]) * dt

    _new_yaw = current_pose[2] + vel[2] * dt
    # -pi ~ pi
    new_yaw = np.arctan2(np.sin(_new_yaw), np.cos(_new_yaw))

    new_pose = np.array([new_xy[0], new_xy[1], new_yaw])

    return new_pose


@numba.jit(nopython=True, cache=True, fastmath=True)
def generate_trajectory(
    pose: np.ndarray,
    vel: np.ndarray,
    target_vel: np.ndarray,
    accel_limits: np.ndarray,
    prediction_step: int,
    prediction_interval: float,
) -> np.ndarray:
    """
    Forward simulation of the robot's trajectory with dynamics limitations.
    Args:
        pose: current robot pose [x, y, theta]
        vel: current robot velocity [vx, vy, w]
        target_vel: target robot velocity [vx, vy, w]
        accel_limits: acceleration limitation [ax, ay, aw]
        prediction_step: prediction step size

    Returns:
        prediction trajectory prediction_step x [x, y, theta, vx, vy, w] (prediction_step, 6)
    """

    traj: np.ndarray = np.zeros((prediction_step, 6))
    predict_vel: np.ndarray = np.array([vel[0], vel[1], vel[2]])
    predict_pose: np.ndarray = np.array([pose[0], pose[1], pose[2]])
    for i in range(prediction_step):
        predict_vel = compute_new_vel(
            predict_vel, target_vel, accel_limits, prediction_interval
        )
        predict_pose = motion_model(predict_pose, predict_vel, prediction_interval)
        traj[i, :] = np.array(
            [
                predict_pose[0],
                predict_pose[1],
                predict_pose[2],
                predict_vel[0],
                predict_vel[1],
                predict_vel[2],
            ]
        )

    return traj


@numba.jit(nopython=True, cache=True, fastmath=True)
def find_nearest_point(pos: np.ndarray, path: np.ndarray) -> int:
    """
    Find the nearest index on the path.
    Args:
        pos: current robot pose [x, y]
        path: reference global path in global coordinate
    Returns:
        nearest_index: nearest index on the path
    """
    dists = np.sqrt((path[:, 0] - pos[0]) ** 2 + (path[:, 1] - pos[1]) ** 2)
    nearest_index = np.argmin(dists)

    return nearest_index
