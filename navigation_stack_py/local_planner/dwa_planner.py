from __future__ import annotations

from typing import Tuple
import numpy as np
import math
import numba

from .dwa_utils import generate_trajectory, find_nearest_point
from navigation_stack_py.common.robot_observation import RobotObservation
from navigation_stack_py.local_planner import LocalPlannerBase, LocalPlannerOutput
from navigation_stack_py.utils import MapHandler, ParameterHandler
from navigation_stack_py.common import RobotCommand, AgentState


class DWAPlanner(LocalPlannerBase):
    def __init__(self, params: ParameterHandler) -> None:
        super().__init__()
        self._local_cost_map: MapHandler = None

        # common parameters
        self.CONTROL_INTERVAL_TIME = params.control_interval_time
        self.ROBOT_RADIUS = params.robot_radius
        self.INFLATION_RADIUS = params.robot_radius + params.inflation_radius  # [m]

        # Sampling parameters
        self.PREDICTION_INTERVAL: float = 0.1  # [s]
        self.PREDICTION_STEP: int = 10  # number of steps for the prediction; prediction horizon is PREDICTION_STEP * PREDICTION_INTERVAL
        self.LINEAR_VEL_X_RESOLUTION: int = 5
        self.LINEAR_VEL_Y_RESOLUTION: int = 3
        self.ANGULAR_VEL_RESOLUTION: int = 8
        # Dynamics limitations
        self.LINEAR_VEL_X_LIMIT: dict = {"MAX": 2.0, "MIN": 0.0}  # [m/s]
        self.LINEAR_VEL_Y_LIMIT: dict = {"MAX": 0.1, "MIN": -0.1}
        self.ANGULAR_VEL_LIMIT: dict = {"MAX": 2.0, "MIN": -2.0}  # [rad/s]
        self.ACCEL_LIMIT: list = [2.5, 2.5, 4.0]  # ax[m/s^2], ay[m/s^2], dw[rad/s^2]
        # Weights for the cost function
        self.PATH_DISTANCE_BIAS: float = 1.0  # [1/m] : 軌道終端からグローバルパスまでへの距離に対する重み
        self.PATH_ANGLE_BIAS: float = 0.0  # [1/rad] : 軌道終端からグローバルパスへの方向に対する重み
        self.GOAL_DISTANCE_BIAS: float = 0.0  # [1/m] : 軌道終端からゴールまでへの距離に対する重み
        self.GOAL_ANGLE_BIAS: float = 0.0  # [1/rad] : 軌道終端からゴールへの方向に対する重み
        self.SUB_GOAL_DISTANCE_BIAS: float = 1.0  # [1/m] : 軌道終端からサブゴールまでへの距離に対する重み
        self.SUB_GOAL_ANGLE_BIAS: float = (
            0.0001  # [1/rad] : 軌道終端からサブゴールへの方向に対する重み = 停止時の不要な回転を抑制
        )
        # Goal reach params
        self.POSITION_TOLERANCE: float = 1.0  # [m]
        self.ANGLE_TOLERANCE: float = np.pi / 180.0 * 360.0  # [rad]
        # Sub-goal search
        self.SUB_GOAL_AHEAD: float = 3.0  # [s]
        self.SUB_GOAL_AHEAD_OFFSET: float = 2.0  # [m]

        if params.dwa_config is not None:
            self.LINEAR_VEL_X_LIMIT["MAX"] = params.dwa_config["vel_x_limit"]["max"]
            self.LINEAR_VEL_X_LIMIT["MIN"] = params.dwa_config["vel_x_limit"]["min"]
            self.LINEAR_VEL_Y_LIMIT["MAX"] = params.dwa_config["vel_y_limit"]["max"]
            self.LINEAR_VEL_Y_LIMIT["MIN"] = params.dwa_config["vel_y_limit"]["min"]
            self.ANGULAR_VEL_LIMIT["MAX"] = params.dwa_config["vel_theta_limit"]["max"]
            self.ANGULAR_VEL_LIMIT["MIN"] = params.dwa_config["vel_theta_limit"]["min"]
            self.LINEAR_VEL_X_RESOLUTION = params.dwa_config["vel_x_samples"]
            self.LINEAR_VEL_Y_RESOLUTION = params.dwa_config["vel_y_samples"]
            self.ANGULAR_VEL_RESOLUTION = params.dwa_config["vel_theta_samples"]
            self.PATH_DISTANCE_BIAS = params.dwa_config["path_distance_bias"]
            self.PATH_ANGLE_BIAS = params.dwa_config["path_angle_bias"]
            self.SUB_GOAL_DISTANCE_BIAS = params.dwa_config["sub_goal_distance_bias"]
            self.SUB_GOAL_ANGLE_BIAS = params.dwa_config["sub_goal_angle_bias"]
        else:
            print("DWAPlanner: No dwa_config is given. Use default parameters.")

    def get_max_speed(self) -> float:
        vx = self.LINEAR_VEL_X_LIMIT["MAX"]
        vy = self.LINEAR_VEL_Y_LIMIT["MAX"]
        return np.hypot(vx, vy)

    def set_costmap(self, costmap: MapHandler) -> None:
        self._local_cost_map = costmap.get_inflated_map(
            inflation_radius=self.INFLATION_RADIUS
        )

    def get_costmap(self) -> MapHandler:
        return self._local_cost_map

    def is_goal_reached(
        self, robot_obs: RobotObservation, goal_state: AgentState
    ) -> bool:
        is_goal = False
        dist_to_goal = np.hypot(
            robot_obs.state.pos[0] - goal_state.pos[0],
            robot_obs.state.pos[1] - goal_state.pos[1],
        )
        _angle = robot_obs.state.yaw - goal_state.yaw
        angle_to_goal = np.abs(np.arctan2(np.sin(_angle), np.cos(_angle)))

        if (
            dist_to_goal < self.POSITION_TOLERANCE
            and angle_to_goal < self.ANGLE_TOLERANCE
        ):
            is_goal = True

        return is_goal

    def is_goal_reached_np(self, robot_pos: np.ndarray, goal_pos: np.ndarray) -> bool:
        """
        robot_pos: [x, y, yaw]
        goal_pos: [x, y, yaw]
        """
        is_goal = False
        dist_to_goal = np.hypot(robot_pos[0] - goal_pos[0], robot_pos[1] - goal_pos[1])
        _angle = robot_pos[2] - goal_pos[2]
        angle_to_goal = np.abs(np.arctan2(np.sin(_angle), np.cos(_angle)))

        if (
            dist_to_goal < self.POSITION_TOLERANCE
            and angle_to_goal < self.ANGLE_TOLERANCE
        ):
            is_goal = True

        return is_goal

    def _search_sub_goal(
        self, current_pos: np.ndarray, current_vx: float, reference_path: np.ndarray
    ) -> int:
        """
        Set the sub-goal from global reference path
        Args:
            current_pos: (x,y)
            reference_path: reference path (N, 2) [x,y]
        Returns:
            sub_goal_index
        """
        if reference_path is None:
            return None

        if reference_path.shape[0] == 0:
            return None

        #  calc nearest point on reference path
        nearest_point_index = find_nearest_point(current_pos, reference_path)

        # Search sub-goal
        sub_goal_index: int = nearest_point_index
        accumulate_dist: float = 0.0

        for i in range(nearest_point_index, reference_path.shape[0]):
            if i == reference_path.shape[0] - 1:
                sub_goal_index = reference_path.shape[0] - 1
                break
            else:
                next_index = min(i + 1, reference_path.shape[0] - 1)
                accumulate_dist += np.linalg.norm(
                    reference_path[next_index, :] - reference_path[i, :]
                )
                sub_goal_ahead = (
                    current_vx * self.SUB_GOAL_AHEAD + self.SUB_GOAL_AHEAD_OFFSET
                )
                if accumulate_dist > sub_goal_ahead:
                    sub_goal_index = next_index
                    break
                else:
                    pass

        return sub_goal_index

    def compute_velocity_command(
        self,
        robot_obs: RobotObservation,
        goal_state: AgentState,
        reference_path: np.ndarray,
    ) -> LocalPlannerOutput:
        """
        Compute the velocity command for the robot by using the Dynamic Window Approach.
        Args:
            robot_obs: robot observation
            goal_state: goal state
            reference_path: reference path (N, 2) [x, y] in continuous coordinates
        Returns:
            output: local planner output
        """

        # GOAL REACHED
        is_goal_reached = self.is_goal_reached(robot_obs, goal_state)

        if is_goal_reached or reference_path is None:
            zero_command = RobotCommand(
                linear_vel=np.array([0.0, 0.0]), angular_vel=0.0
            )
            return LocalPlannerOutput(
                control_command=zero_command,
                is_goal_reached=True,
                predict_path_list=None,
                best_index=None,
                path_obs_costs=None,
                sub_goal_index=None,
            )

        # Search sub-goal
        sub_goal_index = self._search_sub_goal(
            robot_obs.state.pos, robot_obs.state.linear_vel[0], reference_path[0]
        )

        # compute dynamic window
        linear_x_dw, linear_y_dw, angular_dw = self._calc_dynamic_window(
            robot_obs.state
        )

        # calculate candidates
        vx_candidates = np.linspace(
            linear_x_dw[0], linear_x_dw[1], self.LINEAR_VEL_X_RESOLUTION
        )
        vy_candidates = np.linspace(
            linear_y_dw[0], linear_y_dw[1], self.LINEAR_VEL_Y_RESOLUTION
        )
        w_candidates = np.linspace(
            angular_dw[0], angular_dw[1], self.ANGULAR_VEL_RESOLUTION
        )

        # forward prediction
        current_pose = np.array(
            [robot_obs.state.pos[0], robot_obs.state.pos[1], robot_obs.state.yaw]
        )
        current_vel = np.array(
            [
                robot_obs.state.linear_vel[0],
                robot_obs.state.linear_vel[1],
                robot_obs.state.angular_vel,
            ]
        )
        accel_limits = np.array(
            [self.ACCEL_LIMIT[0], self.ACCEL_LIMIT[1], self.ACCEL_LIMIT[2]]
        )
        prediction_step = int(self.PREDICTION_STEP)
        prediction_interval = float(self.PREDICTION_INTERVAL)

        target_vel_array = np.array(
            np.meshgrid(vx_candidates, vy_candidates, w_candidates)
        ).T.reshape(-1, 3)
        predict_traj_array = self._generate_trajectory_array(
            current_pose,
            current_vel,
            target_vel_array,
            accel_limits,
            prediction_step,
            prediction_interval,
        )

        # evaluate trajectory and find best trajectory
        costs, obs_cost = self._eval_all_trajectories(
            predict_traj_array, goal_state, reference_path, sub_goal_index
        )

        best_index = self._find_best_trajectory(predict_traj_array, costs)

        # control command
        if best_index == -1:
            # Not found best trajectory
            linear_cmd = np.array([0.0, 0.0])
            angular_cmd = 0.0
        else:
            linear_cmd = predict_traj_array[best_index][0, 3:5]
            angular_cmd = predict_traj_array[best_index][0, 5]
        robot_command = RobotCommand(linear_vel=linear_cmd, angular_vel=angular_cmd)

        # predict path
        predict_path_list: list = []
        for predict_traj in predict_traj_array:
            predict_path: np.ndarray = np.zeros((self.PREDICTION_STEP, 2))
            predict_path[:, 0] = predict_traj[:, 0]
            predict_path[:, 1] = predict_traj[:, 1]
            predict_path_list.append(predict_path)

        # prediction path obstacle cost
        # robot_pixel = self._local_cost_map.meter2pixel(self.ROBOT_RADIUS)
        # _, obs_cost_arr = self._calc_obs_cost(predict_path_list[best_index], robot_pixel)

        return LocalPlannerOutput(
            control_command=robot_command,
            predict_path_list=predict_path_list,
            best_index=best_index,
            is_goal_reached=is_goal_reached,
            path_obs_costs=obs_cost,
            sub_goal_index=sub_goal_index,
        )

    def _calc_dynamic_window(self, robot_state: AgentState) -> Tuple[list, list, list]:
        """
        Calculate dynamic window based on the current robot state
        Args:
            robot_state:
        Returns:
            dynamic_window: dynamic window [v_min, v_max, w_min, w_max]
        """

        # Dynamic window from robot specification
        linear_x_window_s = [
            self.LINEAR_VEL_X_LIMIT["MIN"],
            self.LINEAR_VEL_X_LIMIT["MAX"],
        ]
        linear_y_window_s = [
            self.LINEAR_VEL_Y_LIMIT["MIN"],
            self.LINEAR_VEL_Y_LIMIT["MAX"],
        ]
        angular_window_s = [
            self.ANGULAR_VEL_LIMIT["MIN"],
            self.ANGULAR_VEL_LIMIT["MAX"],
        ]

        # Dynamic window from robot motion
        linear_x_window_d = [
            robot_state.linear_vel[0]
            - self.ACCEL_LIMIT[0] * self.CONTROL_INTERVAL_TIME,
            robot_state.linear_vel[0]
            + self.ACCEL_LIMIT[0] * self.CONTROL_INTERVAL_TIME,
        ]
        linear_y_window_d = [
            robot_state.linear_vel[1]
            - self.ACCEL_LIMIT[1] * self.CONTROL_INTERVAL_TIME,
            robot_state.linear_vel[1]
            + self.ACCEL_LIMIT[1] * self.CONTROL_INTERVAL_TIME,
        ]
        angular_window_d = [
            robot_state.angular_vel - self.ACCEL_LIMIT[2] * self.CONTROL_INTERVAL_TIME,
            robot_state.angular_vel + self.ACCEL_LIMIT[2] * self.CONTROL_INTERVAL_TIME,
        ]

        # calculate dynamic window
        linear_x_dynamic_window = [
            max(linear_x_window_s[0], linear_x_window_d[0]),
            min(linear_x_window_s[1], linear_x_window_d[1]),
        ]
        linear_y_dynamic_window = [
            max(linear_y_window_s[0], linear_y_window_d[0]),
            min(linear_y_window_s[1], linear_y_window_d[1]),
        ]
        angular_dynamic_window = [
            max(angular_window_s[0], angular_window_d[0]),
            min(angular_window_s[1], angular_window_d[1]),
        ]

        return linear_x_dynamic_window, linear_y_dynamic_window, angular_dynamic_window

    @staticmethod
    @numba.jit(nopython=True, cache=True, fastmath=True)
    def _generate_trajectory_array(
        pose: np.ndarray,
        vel: np.ndarray,
        target_vel_array: np.ndarray,
        accel_limits: np.ndarray,
        prediction_step: int,
        prediction_interval: float,
    ) -> np.ndarray:
        """
        Forward simulation of the robot's trajectory with dynamics limitations.
        Args:
            pose: current robot pose [x, y, theta]
            vel: current robot velocity [vx, vy, w]
            target_vel_array: target robot velocity array N x [vx, vy, w], (N, 3)
            accel_limits: acceleration limitation [ax, ay, aw]
            prediction_step: prediction step size

        Returns:
            prediction trajectory array:   N, prediction_step x [x, y, theta, vx, vy, w] (N, prediction_step, 6)
        """

        predict_traj_array: np.ndarray = np.zeros(
            (target_vel_array.shape[0], prediction_step, 6)
        )

        for i in range(target_vel_array.shape[0]):
            predict_traj_array[i, :] = generate_trajectory(
                pose,
                vel,
                target_vel_array[i],
                accel_limits,
                prediction_step,
                prediction_interval,
            )

        return predict_traj_array

    def _eval_trajectory(
        self,
        traj: np.ndarray,
        goal_state: AgentState,
        reference_path: np.ndarray,
        sub_goal_index: int,
    ) -> float:
        """
        Evaluate the trajectory by computing the cost function.
        Args:
            traj: trajectory PREDICTION_STEP x [x, y, theta, vx, vy, w]
            goal_state: goal state in global coordinate
            reference path: reference global path in global coordinate [0, N, 2] (x,y)
        Returns:
            cost: cost function value
        """
        cost = 0.0

        # eval distance to goal
        dx_to_goal = goal_state.pos[0] - traj[-1, 0]
        dy_to_goal = goal_state.pos[1] - traj[-1, 1]

        cost = self.GOAL_DISTANCE_BIAS * math.sqrt(dx_to_goal**2 + dy_to_goal**2)

        # Angle to goal
        # angle_to_goal = np.arctan2(dy_to_goal, dx_to_goal)
        traj_angle = np.arctan2(np.sin(traj[-1, 2]), np.cos(traj[-1, 2]))
        _angle_to_goal = goal_state.yaw - traj_angle
        angle_to_goal = np.arctan2(np.sin(_angle_to_goal), np.cos(_angle_to_goal))
        cost += self.GOAL_ANGLE_BIAS * abs(angle_to_goal)

        # eval distance to sub goal
        dx_to_sub_goal = reference_path[0, sub_goal_index, 0] - traj[-1, 0]
        dy_to_sub_goal = reference_path[0, sub_goal_index, 1] - traj[-1, 1]
        cost += self.SUB_GOAL_DISTANCE_BIAS * math.sqrt(
            dx_to_sub_goal**2 + dy_to_sub_goal**2
        )

        # eval angle to sub goal
        _angle_to_sub_goal = np.arctan2(dy_to_sub_goal, dx_to_sub_goal) - traj_angle
        angle_to_sub_goal = np.arctan2(
            np.sin(_angle_to_sub_goal), np.cos(_angle_to_sub_goal)
        )
        cost += self.SUB_GOAL_ANGLE_BIAS * abs(angle_to_sub_goal)

        #  eval distance to reference path
        nearest_idx_of_terminal = find_nearest_point(traj[-1, 0:2], reference_path[0])
        dx_to_path = reference_path[0, nearest_idx_of_terminal, 0] - traj[-1, 0]
        dy_to_path = reference_path[0, nearest_idx_of_terminal, 1] - traj[-1, 1]

        cost += self.PATH_DISTANCE_BIAS * math.sqrt(dx_to_path**2 + dy_to_path**2)

        # eval angle to reference path
        # angle_to_path = np.arctan2(dy_to_path, dx_to_path)
        # _angle_to_path = reference_path[0, nearest_idx_of_terminal, 2] - traj[-1, 2]
        # angle_to_path = np.arctan2(np.sin(_angle_to_path), np.cos(_angle_to_path))
        next_index = min(nearest_idx_of_terminal + 1, reference_path[0].shape[0] - 1)
        dx_reference_path = (
            reference_path[0, next_index, 0]
            - reference_path[0, nearest_idx_of_terminal, 0]
        )
        dy_reference_path = (
            reference_path[0, next_index, 1]
            - reference_path[0, nearest_idx_of_terminal, 1]
        )
        reference_path_angle = np.arctan2(dy_reference_path, dx_reference_path)

        cost += self.PATH_ANGLE_BIAS * abs(reference_path_angle - traj_angle)

        # collision check
        for pose in traj:
            is_collision = self._local_cost_map.check_collision(
                pose[0:2], self.ROBOT_RADIUS
            )
            if is_collision:
                cost += np.inf

        return cost

    def _eval_all_trajectories(
        self,
        trajs: np.ndarray,
        goal_state: AgentState,
        reference_path: np.ndarray,
        sub_goal_index: int,
    ) -> np.ndarray:
        """
        Evaluate all trajectories by computing the cost function.
        Args:
            trajs: NUM_SAMPLES x PREDICTION_STEP x [x, y, theta, vx, vy, w]
            goal_state: goal state in global coordinate
            reference path: reference global path in global coordinate
            sub_goal_index: index of sub goal
        Returns:
            costs: cost function values
            obs_costs: obstacle cost function values
        """
        costs = np.zeros(trajs.shape[0])
        obs_costs = np.zeros(trajs.shape[0])

        for i in range(trajs.shape[0]):
            costs[i] = self._eval_trajectory(
                trajs[i], goal_state, reference_path, sub_goal_index
            )
            obs_costs[i], _ = self._calc_obs_cost(trajs[i, :, 0:2])

        return costs, obs_costs

    def _find_best_trajectory(self, trajs: np.ndarray, costs: np.ndarray) -> int:
        """
        Find the best trajectory by minimizing the cost function.
        Args:
            trajs: trajectories NUM_SAMPLE x PREDICTION_STEP x [x, y, theta, vx, vy, w]
            costs: cost function values
        Returns:
            best_traj: best trajectory index if not found, -1 otherwise
        """
        if trajs.shape[0] == 0:
            print("No trajectory found")
            best_index = -1
        else:
            best_index = np.argmin(costs)

        return best_index

    def _calc_obs_cost(self, path: np.ndarray) -> list[float, np.ndarray]:
        """
        Calculate the cost function of obstacle.
        Args:
            path: trajectory PREDICTION_STEP x [x, y, theta, vx, vy, w]
            radius_index: pixel of robot radius
        Returns:
            cost: cost function value
            cost_array: cost function array
        """
        cost_arr: np.ndarray = np.zeros(len(path))

        path_ij = self._local_cost_map.pose_array2index_array(path)

        occ = self._local_cost_map.get_map_as_np("occupancy")

        for i in range(len(path_ij)):
            cost_arr[i] = occ[path_ij[i, 0], path_ij[i, 1]]

        cost = np.sum(cost_arr)

        return cost, cost_arr
