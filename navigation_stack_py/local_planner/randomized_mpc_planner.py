"""
    Randomized Model Predictive Control (R-MPC) local planner
    Kohei Honda, 2022
"""

from __future__ import annotations

from typing import Callable, Tuple
import numpy as np

from .dwa_utils import find_nearest_point
from navigation_stack_py.common.robot_observation import RobotObservation
from navigation_stack_py.local_planner import LocalPlannerBase, LocalPlannerOutput
from navigation_stack_py.utils import MapHandler, ParameterHandler
from navigation_stack_py.common import RobotCommand, AgentState


def state_function(state: np.ndarray, input: np.ndarray, dt: float) -> np.ndarray:
    """
    State function

    Args:
        state (np.ndarray): state vector (x, y, theta)
        input (np.ndarray): input vector (vx, vy, w)
        dt (float): time interval

    Returns:
        np.ndarray: next state vector (x, y, theta)
    """

    xy = state[0:2]
    rot_mat = np.array(
        [[np.cos(state[2]), -np.sin(state[2])], [np.sin(state[2]), np.cos(state[2])]]
    )
    new_xy = xy + rot_mat.dot(input[0:2]) * dt

    _new_yaw = state[2] + input[2] * dt
    # -pi ~ pi
    new_yaw = np.arctan2(np.sin(_new_yaw), np.cos(_new_yaw))

    new_state = np.array([new_xy[0], new_xy[1], new_yaw])

    return new_state


def cost_function(
    predicted_state: np.ndarray,
    input_series: np.ndarray,
    goal_state: np.ndarray,
    reference_path: np.ndarray,
    dt: float,
    state_weights: dict,
    input_weights: dict,
) -> float:
    """
    Cost function

    Args:
        state (np.ndarray): state vector (dim_horizon) x (x, y, theta)
        input (np.ndarray): input vector (dim_horizon) x (vx, vy, w)
        goal_state (np.ndarray): goal state vector (x, y, theta)
        reference_path (np.ndarray): reference path (dim_reference_path) x (x, y)
        cost_map (MapHandler): cost map
        dt (float): time interval

    Returns:
        float: cost
    """
    cost = 0.0

    # stage cost
    for i in range(1, predicted_state.shape[0] - 1):
        # distance to goal
        dist_to_goal = np.linalg.norm(predicted_state[i, 0:2] - goal_state[0:2])
        cost += state_weights["dist_to_goal"] * dist_to_goal

        # dist to reference path
        index = find_nearest_point(predicted_state[i, 0:2], reference_path)
        if index is not None and index > 0:
            cost += state_weights["dist_to_path"] * np.linalg.norm(
                predicted_state[i, 0:2] - reference_path[index]
            )

        # speed: better to be fast
        cost -= input_weights["vx"] * (input_series[i, 0]) ** 2

    # terminal cost
    dist_to_goal = predicted_state[-1, 0:2] - goal_state[0:2]
    cost += 10 * state_weights["dist_to_goal"] * np.linalg.norm(dist_to_goal)

    return cost


def constraint(
    predicted_state: np.ndarray,
    input_series: np.ndarray,
    cost_map: MapHandler,
    dt: float,
) -> int:
    """
    Constraint function

    Args:
        state (np.ndarray): state vector (num_horizon) x (x, y, theta)
        input (np.ndarray): input vector (num_horizon) x (vx, vy, w)
        cost_map (MapHandler): cost map
        dt (float): time interval

    Returns:
        bool: True if constraint is satisfied
    """
    is_satisfied = True

    # safety constraint
    robot_radius = 1.0

    for i in range(predicted_state.shape[0]):
        # Safety constraint
        is_collision = cost_map.check_collision(predicted_state[i, 0:2], robot_radius)
        if is_collision:
            is_satisfied = False
            break

    return int(is_satisfied)


class RandomizedMPCPlanner(LocalPlannerBase):
    """
    Randomized MPC for local planning

    params: parameters for local planner
    params: state function
    params: cost function
    params: constraint function
    """

    def __init__(
        self,
        params: ParameterHandler,
        state_function: Callable = state_function,
        cost_function: Callable = cost_function,
        constraint: Callable = constraint,
    ) -> None:
        super().__init__()

        ######### Parameters #########
        self._inflation_radius = params.robot_radius + params.inflation_radius

        # Goal reach params
        self._control_interval_time: float = params.control_interval_time
        self._position_tolerance: float = 1.0  # [m]
        self._angle_tolerance: float = np.pi / 180.0 * 360.0  # [rad]

        # sub-goal params
        self._sub_goal_ahead: float = 3.0  # [s]
        self._sub_goal_ahead_offset: float = 1.0  # [m]

        # Dynamics limitations
        self._vx_limit: dict = {"max": 4.0, "min": 0.0}  # [m/s]
        self._vy_limit: dict = {"max": 0.1, "min": -0.1}
        self._vw_limit: dict = {"max": 2.0, "min": -2.0}  # [rad/s]

        # MPC params
        self._num_samples: int = 300  # number of samples of random input
        self._num_horizon: int = 50  # number of prediction horizon steps
        self._num_freq: int = 3  # maximum frequency of I-FFT
        self._num_synthesis: int = 5  # number of synthetic solution

        # weight params
        self._state_weights: dict = {"dist_to_goal": 1.0, "dist_to_path": 1.0}
        self._input_weights: dict = {"vx": 0.1, "vy": 0.0, "w": 0.0}

        # MPC function
        self._state_function = state_function
        self._cost_function = cost_function
        self._constraint = constraint

        # random generator seed
        seed = 0
        self._rng = np.random.default_rng(seed)

        ######### Variables #########

        # for random input
        self._waves = None

        self._local_cost_map: MapHandler = None

    def set_costmap(self, cost_map: MapHandler) -> None:
        self._local_cost_map = cost_map.get_inflated_map(
            inflation_radius=self._inflation_radius
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
            dist_to_goal < self._position_tolerance
            and angle_to_goal < self._angle_tolerance
        ):
            is_goal = True

        return is_goal

    def get_max_speed(self) -> float:
        vx = self._vx_limit["max"]
        vy = self._vy_limit["max"]
        return np.hypot(vx, vy)

    def compute_velocity_command(
        self,
        robot_obs: RobotObservation,
        goal_state: AgentState,
        reference_path: np.ndarray,
    ) -> LocalPlannerOutput:
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

        # Generate random solution candidates
        min_value = np.array(
            [self._vx_limit["min"], self._vy_limit["min"], self._vw_limit["min"]]
        )
        max_value = np.array(
            [self._vx_limit["max"], self._vy_limit["max"], self._vw_limit["max"]]
        )
        random_inputs = self._generate_random_solutions(
            min_value=min_value,
            max_value=max_value,
            num_samples=self._num_samples,
            num_freq=self._num_freq,
            num_horizon=self._num_horizon,
        )

        # Forward simulation for all solution candidates
        def state_func_batch(
            state_batch: np.ndarray, input_batch: np.ndarray, dt: float
        ) -> np.ndarray:
            return np.array(
                [
                    self._state_function(state=state, input=input, dt=dt)
                    for state, input in zip(state_batch, input_batch)
                ]
            )

        initial_state = np.array(
            [robot_obs.state.pos[0], robot_obs.state.pos[1], robot_obs.state.yaw]
        )
        predicted_states = self._predict(
            state_func_batch=state_func_batch,
            initial_state=initial_state,
            input_vectors=random_inputs,
            dt=self._control_interval_time,
        )

        # Check constraints for all solution candidates
        def constraint_func_batch(
            state_batch: np.ndarray, input_batch: np.ndarray
        ) -> np.ndarray:
            return np.array(
                [
                    self._constraint(
                        predicted_state=state,
                        input_series=input,
                        cost_map=self._local_cost_map,
                        dt=self._control_interval_time,
                    )
                    for state, input in zip(state_batch, input_batch)
                ]
            )

        valid_predicted_states, valid_inputs = self._check_constraints(
            constraint_func=constraint_func_batch,
            predicted_states=predicted_states,
            predicted_inputs=random_inputs,
            cost_map=self._local_cost_map,
            dt=self._control_interval_time,
        )

        # Search sub-goal
        sub_goal_index = self._search_sub_goal(
            robot_obs.state.pos, robot_obs.state.linear_vel[0], reference_path[0]
        )
        if valid_predicted_states.shape[0] == 0:
            # pure rotation
            rotate_command = RobotCommand(
                linear_vel=np.array([0.0, 0.0]), angular_vel=max_value[2]
            )
            return LocalPlannerOutput(
                control_command=rotate_command,
                is_goal_reached=False,
                predict_path_list=None,
                best_index=None,
                path_obs_costs=None,
                sub_goal_index=sub_goal_index,
            )

        # Find the best solution candidate

        sub_goal_pos = reference_path[0][sub_goal_index]

        def cost_func_batch(
            state_batch: np.ndarray, input_batch: np.ndarray
        ) -> np.ndarray:
            return np.array(
                [
                    self._cost_function(
                        predicted_state=state,
                        input_series=input,
                        goal_state=sub_goal_pos,
                        reference_path=reference_path[0],
                        dt=self._control_interval_time,
                        state_weights=self._state_weights,
                        input_weights=self._input_weights,
                    )
                    for state, input in zip(state_batch, input_batch)
                ]
            )

        costs = cost_func_batch(valid_predicted_states, valid_inputs)

        # find best solution by mean candidates
        best_indices = np.argsort(costs)[: self._num_synthesis]
        solution = np.mean(
            valid_inputs[best_indices, 0, :], axis=0
        )  # best solution is the first step input of the best candidate

        # to display synthesized path
        best_predicted_state = np.mean(valid_predicted_states[best_indices], axis=0)

        linear_cmd = solution[0:2]
        angular_cmd = solution[2]
        robot_cmd = RobotCommand(linear_vel=linear_cmd, angular_vel=angular_cmd)

        predict_path_list: list = []
        for i, valid_state in enumerate(valid_predicted_states):
            if i == best_indices[0]:
                predict_path_list.append(best_predicted_state[:, 0:2])
            else:
                predict_path_list.append(valid_state[:, 0:2])

        return LocalPlannerOutput(
            control_command=robot_cmd,
            is_goal_reached=False,
            predict_path_list=predict_path_list,
            best_index=best_indices[0],
            path_obs_costs=None,
            sub_goal_index=sub_goal_index,
        )

    def _setup_waves(self, num_freq: int, num_horizon: int) -> np.ndarray:
        if self._waves is not None:
            return self._waves

        shape = (2 * num_freq - 2, num_horizon)

        self._waves = np.zeros(shape)
        thetas = np.linspace(0, 2 * np.pi, num_horizon)
        for i in range(num_freq - 1):
            self._waves[i, :] = np.cos(thetas * (i + 1))
            self._waves[i + num_freq - 1, :] = np.sin(thetas * (i + 1))

        return self._waves

    def _generate_random_solutions(
        self,
        min_value: np.ndarray,
        max_value: np.ndarray,
        num_samples: int,
        num_horizon: int,
        num_freq: int,
    ) -> np.ndarray:
        """
        Generate random input by inverse fast Fourier transform

        Args:
            min_value (float): minimum value (N, )
            max_value (float): maximum value (N, )
            num_samples (int): number of generated input samples
            num_horizon (int): number of prediction horizon; prediction time = num_horizon * control_interval_time
            num_freq (int): number of frequencies

        Returns:
            np.ndarray: input vector (num_samples, num_horizon, N)
        """
        if min_value.shape != max_value.shape:
            raise ValueError("min_value and max_value must have the same shape")
        else:
            solution_dim = min_value.shape[0]

        # prepare waves
        waves = self._setup_waves(num_freq, num_horizon)

        # generate random input
        generated_solutions = np.zeros((num_samples, num_horizon, solution_dim))

        for i in range(solution_dim):
            # random weights in frequency domain
            # uniform distribution
            # weights =  0.5 * (max_value[i] - min_value[i]) * (2 * self._rng.random((num_samples, 2 * num_freq - 2)) - 1)

            # normal distribution
            mean = (max_value[i] + min_value[i]) / 2
            # variance = (max_value[i] - min_value[i]) / 2
            variance = (max_value[i] - min_value[i]) / 4
            weights = self._rng.normal(
                loc=mean, scale=variance, size=(num_samples, 2 * num_freq - 2)
            )

            # 1d solution
            solution = np.dot(weights, waves)
            solution = np.clip(solution, min_value[i], max_value[i])

            # inverse fast Fourier transform
            generated_solutions[:, :, i] = solution

        return generated_solutions

    def _predict(
        self,
        state_func_batch: Callable,
        initial_state: np.ndarray,
        input_vectors: np.ndarray,
        dt,
    ) -> np.ndarray:
        """
        Predict state with input vectors by Euler method
        param: initial_state (dim_state, ): initial state
        param: input_vectors (num_samples, num_horizon, dim_input): input vectors: solution candidates
        param: state_func (Callable): state function
        param: dt (float): time step

        return: predicted_states (num_samples, num_horizon, dim_state): predicted states
        """

        num_horizon = input_vectors.shape[1]
        num_samples = input_vectors.shape[0]

        predicted_states = np.zeros((num_samples, num_horizon, initial_state.shape[0]))
        predicted_states[:, 0, :] = initial_state
        for i in range(num_horizon - 1):
            predicted_states[:, i + 1, :] = state_func_batch(
                predicted_states[:, i, :], input_vectors[:, i, :], dt
            )

        return predicted_states

    def _check_constraints(
        self,
        constraint_func: Callable,
        predicted_states: np.ndarray,
        predicted_inputs: np.ndarray,
        cost_map: MapHandler,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Check constraints and return valid states and inputs
        param: constraint_func (Callable): constraint function
        param: predicted_states (num_samples, num_horizon, dim_state): predicted states
        param: predicted_inputs (num_samples, num_horizon, dim_input): predicted inputs
        param: cost_map (MapHandler): cost map
        param: dt (float): time step

        return: valid_states (num_valid_samples, num_horizon, dim_state): valid states
                valid_inputs (num_valid_samples, num_horizon, dim_input): valid inputs
        """
        if predicted_states.shape[0] != predicted_inputs.shape[0]:
            raise ValueError(
                "predicted_states and predicted_inputs must have the same number of samples"
            )

        sample_num = predicted_states.shape[0]

        mask = np.ones(sample_num)

        mask[:] = constraint_func(predicted_states[:], predicted_inputs[:])

        valid_states = predicted_states[mask == 1]
        valid_inputs = predicted_inputs[mask == 1]

        return valid_states, valid_inputs

    def _eval_predicted_states(
        self,
        cost_func_batch: Callable,
        predicted_states: np.ndarray,
        predicted_inputs: np.ndarray,
        target_state: np.ndarray,
        cost_map: MapHandler,
        dt: float,
    ) -> np.ndarray:
        """
        Evaluate predicted states
        param: cost_func_batch (Callable): cost function for batch data
        param: predicted_states (num_samples, num_horizon, dim_state): predicted states
        param: predicted_inputs (num_samples, num_horizon, dim_input): predicted inputs
        param: target_state (dim_state, ): target state
        param: cost_map (MapHandler): cost map
        dt (float): time step interval

        return: costs (num_samples, ): costs
        """
        num_samples = predicted_states.shape[0]

        # evaluate predicted states
        costs = np.zeros((num_samples,))

        costs[:] = cost_func_batch(predicted_states[:], predicted_inputs[:])

        return costs

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
                    current_vx * self._sub_goal_ahead + self._sub_goal_ahead_offset
                )
                if accumulate_dist > sub_goal_ahead:
                    sub_goal_index = i
                    break
                else:
                    pass

        return sub_goal_index
