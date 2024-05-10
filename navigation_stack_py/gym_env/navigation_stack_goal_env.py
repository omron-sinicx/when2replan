from __future__ import annotations
from cmath import inf
from enum import IntEnum, Enum
from typing import Any, Dict, Tuple, Union, Optional
from collections import OrderedDict
from scipy import interpolate

import io
import yaml
import numpy as np
import gym
from gym import spaces, GoalEnv
from matplotlib import pyplot as plt
from collections import defaultdict, deque
from gym.envs.registration import register

from navigation_stack_py.global_planner import DijkstraPlanner
from navigation_stack_py.local_planner import DWAPlanner, LocalPlannerOutput
from navigation_stack_py.simulator import Simulator
from navigation_stack_py.utils import visualizer, ParameterHandler, MapHandler
from navigation_stack_py.common import RobotObservation, AgentState


class Action(IntEnum):
    not_update_reference_path: int = 0
    update_reference_path: int = 1

    # For baseline
    force_update_reference_path: int = -1


class RLParameters:
    def __init__(self, rl_config: str = None):
        if rl_config is None:
            print("Default reward and observation parameters are used.")

            ## Reward parameters ##
            # Positive reward
            self.sgt_reward = 1.0
            self.spl_reward = 0.0
            self.speed_reward = 0.0
            # Negative penalty
            self.collision_penalty = 0.0
            self.replan_penalty = 0.0
            self.oscillation_penalty = 0.0
            self.stuck_penalty = 0.0

            ## Observation parameters ##
            self.dim_scan = 80  # Number of scan points used for observation
            self.num_scans = 1  # Number of accumulated scans
            self.scan_interval = 1.5  # Interval between two scans to be accumulated
            self.accumulated_scan_horizon = 3.0  # Horizon of accumulated scans
            self.dim_reference_path = (
                25  # Number of reference path points used for observation
            )
            self.range_reference_path = 50  # Horizon of reference path points, i.e. use the number of points from current position
            self.dim_goal = 2  # dimension of relative goal pose used for observation
            self.dim_global_planner_status = (
                1  # dimension of global planner status used for observation
            )
            self.dim_current_pose = 3  # dimension of current pose for HER
            self.dim_time_step = 1  # dimension of time step for HER

            ## Action parameters ##
            self.num_actions = 2  # Number of actions [0: not update reference path, 1: update reference path]

            ## counterfactual parameters ##
            self.is_counterfactual = False
            self.cf_prediction_horizon = 10  # Prediction horizon for counterfactual
            self.cf_prediction_interval = 0.2  # Prediction interval for counterfactual

        else:
            config = yaml.safe_load(open(rl_config, "r"))

            ## Reward parameters ##
            reward = config["reward"]
            self.sgt_reward = reward["sgt_reward"]
            self.spl_reward = reward["spl_reward"]
            self.speed_reward = reward["speed_reward"]
            self.collision_penalty = reward["collision_penalty"]
            self.replan_penalty = reward["replan_penalty"]
            self.oscillation_penalty = reward["oscillation_penalty"]
            self.stuck_penalty = reward["stuck_penalty"]

            ## Observation parameters ##
            observation = config["observation"]
            self.dim_scan = observation["dim_scan"]
            self.num_scans = observation["num_scans"]
            self.scan_interval = observation["scan_interval"]
            self.accumulated_scan_horizon = observation["accumulated_scan_horizon"]
            self.dim_reference_path = observation["dim_reference_path"]
            self.range_reference_path = observation["range_reference_path"]
            self.dim_goal = observation["dim_goal"]
            self.dim_global_planner_status = observation["dim_global_planner_status"]
            self.dim_current_pose = observation["dim_current_pose"]
            self.dim_time_step = 1

            ## Action parameters ##
            action = config["action"]
            self.num_actions = action["num_actions"]

            ## counterfactual parameters ##
            counterfactual = config["counterfactual"]
            self.is_counterfactual = counterfactual["is_counterfactual"]
            self.cf_prediction_horizon = counterfactual["prediction_horizon"]
            self.cf_prediction_interval = counterfactual["prediction_interval"]

        # Dimension of observation, +1 for available global planner or not
        self.dim_observation = (
            2 * (self.dim_scan * self.num_scans + self.dim_reference_path)
            + self.dim_goal
            + self.dim_global_planner_status
            + self.dim_current_pose
            + self.dim_time_step
        )


class NavigationStackGoalEnv(GoalEnv):
    """Gym wrapper environment of Navigation stack (2D-Simulator, global planner, and local planner)"""

    # common setting
    metadata = {"render.modes": ["rgb_array"]}
    D_TYPE = np.float32

    class VisualizeMode(Enum):
        none = "none"  # No visualization
        observation = "observation"  # Only observation
        birdeye = "birdseye"  # birdseye view

    def __init__(
        self,
        navigation_config: str,
        rl_config: str = None,
        seed: int = 0,
        visualize_mode: str = "none",
        is_training: bool = True,
        continuous_action: bool = False,
    ):
        super(NavigationStackGoalEnv, self).__init__()

        self._is_training = is_training

        # Parameter handler
        if navigation_config is None:
            raise ValueError("navigation_config_path is None")

        elif visualize_mode == "observation":
            self._visualize_mode = self.VisualizeMode.observation
        elif visualize_mode == "birdeye":
            self._visualize_mode = self.VisualizeMode.birdeye
        elif visualize_mode == "none":
            self._visualize_mode = self.VisualizeMode.none
        else:
            raise ValueError("Invalid visualize_mode: {}".format(visualize_mode))

        self._navigation_config_path = navigation_config
        self._seed = seed
        self._params = ParameterHandler()
        self._params.init(self._navigation_config_path, seed=self._seed)

        # Simulator
        self._simulator = Simulator()
        self._counterfactual_simulator = Simulator()

        # Global Planner
        self._global_planner = DijkstraPlanner(self._params)

        # Local Planner
        self._local_planner = DWAPlanner(self._params)

        # params for RL
        self._rl_params = RLParameters(rl_config)
        self.continuous_action = continuous_action
        if continuous_action:
            self.action_space = spaces.Box(
                -1, 1, shape=(self._rl_params.num_actions,), dtype=np.float32
            )
        else:
            self.action_space = spaces.Discrete(self._rl_params.num_actions)

        # for random baseline
        self._rng = np.random.default_rng(seed)

        # Resolution for optimal goal time hash table [m]
        self._optimal_goal_time_table_resolution: float = 1.0

        # observation space
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    low=-inf,
                    high=inf,
                    shape=(self._rl_params.dim_observation,),
                    dtype=self.D_TYPE,
                ),
                "achieved_goal": spaces.Box(
                    low=-inf,
                    high=inf,
                    shape=(self._rl_params.dim_observation,),
                    dtype=self.D_TYPE,
                ),
                "desired_goal": spaces.Box(
                    low=-inf,
                    high=inf,
                    shape=(self._rl_params.dim_observation,),
                    dtype=self.D_TYPE,
                ),
            }
        )

    def reset(self, reset_scenario: bool = True):
        """
        Reset environment with new variables
        """
        # reset new params with random seed
        if reset_scenario:
            # set feasible and valid static map, goal state, and initial state
            (
                self._optimal_goal_time,
                self._optimal_path_length,
                self._optimal_goal_time_table,
            ) = self._reset_feasible_scenario()
            self._goal_state = self._params.goal_state
        else:
            self._static_map = self._simulator.reset(self._params)
            self._goal_state = self._params.goal_state

        # reset variables
        self._birdeye_image_arr = None
        self._step_count = 0
        self._data_dict = defaultdict(lambda: None)
        self._stucking_time = 0.0
        self._average_stuck_time = 0.0
        self._max_stuck_time = 0.0
        self._stuck_previous_checked_pos = None
        self._timer_for_stuck = 0.0
        self._is_stuck = False
        self._num_update_global_planner = 0
        self._accumulated_scans: deque = deque(
            maxlen=int(
                self._rl_params.accumulated_scan_horizon
                / self._params.control_interval_time
            )
        )
        self._end = False
        self._is_oscillation = False
        self._num_oscillation = 0
        self._oscillation_time = 0.0
        self._timer_for_oscillation = 0.0
        self._oscillation_previous_checked_pos = None
        self._flipping_num = 0
        self._flipping_counter_deque: deque = deque(
            maxlen=self._params.flipping_count_num
        )
        self._prev_diff_angle_to_goal = None
        self._traveled_path_length = 0.0
        self._observation_image_arr: np.ndarray = None
        self._counter_for_global_planner_lock = 0  # Not lock global planner
        self._counter_for_hybrid_baseline = 0  # Not lock global planner
        self._previous_global_path = None
        self._previous_global_path_indices = None

        self._is_collision = False
        self._is_timeout = False
        self._is_goal = False

        _ = self._counterfactual_simulator.reset(self._params)

        observation, _, _, _ = self.step(raw_action=1)

        return observation

    def step(self, raw_action: Union[int, float]):
        if self.continuous_action:
            if raw_action > 0:
                action = Action.update_reference_path
            else:
                action = Action.not_update_reference_path
        else:
            action = Action(raw_action)

        # observation of robot
        robot_obs = self._simulator.get_observation()

        # update global planner
        global_reference_indices = None
        global_reference_path = None
        is_replan = False
        if action == Action.not_update_reference_path:
            global_reference_path = self._previous_global_path
            global_reference_indices = self._previous_global_path_indices
            is_replan = False
        if action == Action.update_reference_path:
            if self._is_locked_global_planner(self._counter_for_global_planner_lock):
                # Cannot update global planner because of calculation time
                global_reference_path = self._previous_global_path
                global_reference_indices = self._previous_global_path_indices
            else:
                # Update global planner
                is_replan = True
                self._global_planner.set_costmap(robot_obs.static_map_with_scan)
                (
                    global_reference_indices,
                    global_reference_path,
                ) = self._global_planner.make_plan(
                    robot_obs.state.pos, self._goal_state.pos
                )

                # lock
                self._counter_for_global_planner_lock = self._reset_counter()

                # If not found path, do not update global planner
                if len(global_reference_indices) == 1:
                    global_reference_path = self._previous_global_path
                    global_reference_indices = self._previous_global_path_indices
                else:
                    self._previous_global_path = global_reference_path
                    self._previous_global_path_indices = global_reference_indices

                # log
                self._num_update_global_planner += 1

        # For hybrid baseline force update global planner
        if action == Action.force_update_reference_path:
            if self._is_locked_global_planner(self._counter_for_hybrid_baseline):
                # Cannot update global planner because of calculation time
                global_reference_path = self._previous_global_path
                global_reference_indices = self._previous_global_path_indices
                pass
            else:
                # Update global planner
                self._global_planner.set_costmap(robot_obs.static_map_with_scan)
                (
                    global_reference_indices,
                    global_reference_path,
                ) = self._global_planner.make_plan(
                    robot_obs.state.pos, self._goal_state.pos
                )

                # reset counter
                self._counter_for_hybrid_baseline = self._reset_counter()

                # If not found path, do not update global planner
                if len(global_reference_indices) == 1:
                    global_reference_path = self._previous_global_path
                    global_reference_indices = self._previous_global_path_indices
                else:
                    self._previous_global_path = global_reference_path
                    self._previous_global_path_indices = global_reference_indices

                is_replan = True
                self._num_update_global_planner += 1

        # update local planner
        self._local_planner.set_costmap(robot_obs.static_map_with_scan)
        local_planner_output = self._local_planner.compute_velocity_command(
            robot_obs, self._goal_state, global_reference_path
        )

        # update simulator
        obstacle_map, robot_traj = self._simulator.step(
            local_planner_output.control_command
        )

        # Prepare for rendering
        # inflated_map = self._global_planner.get_costmap()
        inflated_map = self._local_planner.get_costmap()

        if self._visualize_mode == self.VisualizeMode.birdeye:
            self._birdeye_image_arr = visualizer.render(
                static_map=self._static_map,
                obstacle_map=obstacle_map,
                inflation_layer=inflated_map.get_inflation_layer(),
                robot_observation=robot_obs,
                robot_traj=robot_traj,
                goal_state=self._goal_state,
                robot_radius=self._params.robot_radius,
                global_path=global_reference_indices,
                local_path_list=local_planner_output.predict_path_list,
                local_path_best_index=local_planner_output.best_index,
                sub_goal_index=local_planner_output.sub_goal_index,
            )

        # Observation
        goal_robot_obs = self._simulator.get_goal_observation(self._params.goal_state)
        observation = self._get_obs_dict(
            current_robot_obs=robot_obs,
            current_reference_path=global_reference_path,
            goal_robot_obs=goal_robot_obs,
        )

        ## Done
        done: bool = False
        # If reach goal
        if local_planner_output.is_goal_reached:
            done = True
            self._is_goal = True

        # If exceed time limit
        if self._step_count >= self._params.max_episode_steps:
            done = True
            self._is_timeout = True

        # If collision
        if robot_obs.is_collision:
            done = True
            self._is_collision = True

        # update info
        # TimeLimit.truncated is used in stable-baselines3 to handle timeout
        info = {
            "is_goal": self._is_goal,
            "is_collision": self._is_collision,
            "is_timeout": self._is_timeout,
            "TimeLimit.truncated": self._is_timeout,
            "is_success": self._is_goal,
            "optimal_goal_time_table": self._optimal_goal_time_table,
        }

        # Calculate reward
        reward = float(
            self.compute_reward(
                observation["achieved_goal"], observation["desired_goal"], info
            )
        )

        # Stuck detection
        self._stucking_time = self._detect_stuck(
            current_pos=robot_obs.state.pos, current_stucking_time=self._stucking_time
        )
        self._average_stuck_time = self._stucking_time / (
            self._step_count + 1
        ) + self._average_stuck_time * (self._step_count) / (self._step_count + 1)
        self._max_stuck_time = max(self._stucking_time, self._max_stuck_time)

        # Oscillation detection
        self._oscillation_time = self._detect_oscillation(
            current_pos=robot_obs.state.pos,
            current_oscillating_time=self._oscillation_time,
        )
        self._is_oscillation = (
            self._oscillation_time >= self._params.oscillation_detect_time
        )
        self._num_oscillation += int(self._is_oscillation)

        # for event-drive baseline
        deviation_from_reference_path, _ = self._calc_deviation_from_reference_path(
            reference_path=global_reference_path, robot_pos=robot_obs.state.pos
        )
        self._is_event_triggered = False

        if self._stucking_time >= self._params.stuck_time_threshold:
            self._is_event_triggered = True
            self._is_stuck = True
        else:
            self._is_event_triggered = False
            self._is_stuck = False

        if deviation_from_reference_path >= self._params.deviation_threshold:
            self._is_event_triggered = True

        if global_reference_path.shape[0] == 0 or global_reference_path is None:
            self._is_event_triggered = True

        # log variables
        self._data_dict = defaultdict(lambda: None)
        self._data_dict["action"] = int(action)
        self._data_dict["seed"] = self._seed
        self._data_dict["time_step"] = self._step_count
        self._data_dict["time"] = self._step_count * self._params.control_interval_time
        self._data_dict["optimal_goal_time"] = self._optimal_goal_time
        self._data_dict["collision"] = int(robot_obs.is_collision)
        self._data_dict["collision_time"] = (
            float(robot_obs.is_collision) * self._params.control_interval_time
        )
        self._data_dict["stuck_time"] = self._stucking_time
        self._data_dict["average_stuck_time"] = self._average_stuck_time
        self._data_dict["max_stuck_time"] = self._max_stuck_time
        self._data_dict["is_oscillation"] = self._is_oscillation
        self._data_dict["oscillation_time"] = self._oscillation_time
        self._data_dict["num_oscillation"] = self._num_oscillation
        self._data_dict["linear_speed"] = np.linalg.norm(robot_obs.state.linear_vel)
        self._traveled_path_length += (
            np.linalg.norm(robot_obs.state.linear_vel)
            * self._params.control_interval_time
        )
        self._data_dict["travel_distance"] = self._traveled_path_length
        self._data_dict["update_global_planner"] = int(is_replan)
        self._data_dict["num_update_global_planner"] = self._num_update_global_planner
        self._data_dict["reach_goal"] = int(local_planner_output.is_goal_reached)
        if local_planner_output.is_goal_reached:
            self._data_dict["goal_time"] = (
                self._step_count * self._params.control_interval_time
            )
        else:
            self._data_dict["goal_time"] = -1

        if self._step_count >= self._params.max_episode_steps:
            self._data_dict["reach_max_time"] = 1
        else:
            self._data_dict["reach_max_time"] = 0

        # metric for navigation performance
        if done:
            # Success score weighted goal time
            self._data_dict["SGT"] = self._SGT(
                optimal_goal_time=self._optimal_goal_time,
                actual_goal_time=self._step_count * self._params.control_interval_time,
                is_collision=robot_obs.is_collision,
            )
            # Success score weighted path length
            self._data_dict["SPL"] = self._SPL(
                optimal_path_length=self._optimal_path_length,
                actual_path_length=self._traveled_path_length,
                is_goal=local_planner_output.is_goal_reached,
            )
        else:
            # Not goal yet
            self._data_dict["SGT"] = 0.0
            self._data_dict["SPL"] = 0.0

        self._data_dict["done"] = done

        # update time counter
        self._step_count += 1
        # count down for global planner lock
        self._counter_for_global_planner_lock = self._count_down(
            self._counter_for_global_planner_lock
        )
        self._counter_for_hybrid_baseline = self._count_down(
            self._counter_for_hybrid_baseline
        )

        # The env is end when done==True at previous step
        # This is because getting values at end time in the callback function of train
        # This means that call step() on an extra time
        if self._end:
            # done
            zero_reward = 0.0
            return observation, zero_reward, True, info
        else:
            # Not done
            self._end = done
            return observation, reward, False, info

    def render(self, mode="rgb_array") -> np.ndarray:
        """
        return: (x, y, 3)
        """
        if self._visualize_mode == self.VisualizeMode.observation:
            return self._observation_image_arr
        elif self._visualize_mode == self.VisualizeMode.birdeye:
            # birds eye
            return self._birdeye_image_arr
        elif self._visualize_mode == self.VisualizeMode.none:
            pass

    def close(self):
        plt.close()

    def get_max_steps(self) -> int:
        return self._params.max_episode_steps

    def get_max_global_hz(self) -> int:
        return self._params.max_global_planner_hz

    def get_data(self) -> defaultdict:
        """
        Get data for saving logs
            Returns: data dictionary at current time step
        """
        return self._data_dict

    def is_end(self) -> bool:
        return self._end

    def copy_data(self) -> dict:
        # default dict to dict
        data = dict(self._data_dict)

        return data

    def timer_baseline(self) -> int:
        """
        Timer baseline call to update global planner in every control interval time
        This means that replanning at regular intervals
        """
        return int(Action.update_reference_path)

    def random_baseline(self) -> int:
        """
        Random baseline call to update global planner in every control interval time
        This means that replanning at regular intervals
        """
        return int(
            self._rng.choice(
                [Action.update_reference_path, Action.not_update_reference_path]
            )
        )

    def event_triggered_baseline(self) -> int:
        """
        Event-driven baseline call to update global planner when event is triggered
        """
        if self._is_event_triggered:
            return int(Action.update_reference_path)
        else:
            return int(Action.not_update_reference_path)

    def hybrid_baseline(self) -> int:
        """
        Force updates by event-driven replanning with timer replanning
        """
        if self._is_oscillation:
            # oscillation -> NOT replan
            return int(Action.not_update_reference_path)
        elif self._is_event_triggered:
            # stuck or deviate from reference path -> force replan
            return int(Action.force_update_reference_path)
        else:
            # Normal -> timer replan
            return int(Action.update_reference_path)

    def manual_baseline(self) -> int:
        """
        Manual operation
        """
        print("Please enter action:")
        print(
            "Global Planner locked or not: ",
            self._is_locked_global_planner(self._counter_for_global_planner_lock),
        )
        print("Enter: Not replan, Space key: Replan")
        a = input()

        if a == "":
            return int(Action.not_update_reference_path)
        elif a == " ":
            return int(Action.update_reference_path)
        else:
            print("Invalid action")
            return int(Action.not_update_reference_path)

    def _is_locked_global_planner(self, count: int) -> bool:
        return count != 0

    def _count_down(self, counter) -> int:
        if counter > 0:
            return counter - 1
        else:
            return 0

    def _reset_counter(self) -> int:
        return self._params.max_global_planner_hz

    def _detect_stuck(
        self, current_pos: np.ndarray, current_stucking_time: float
    ) -> float:
        """
        Detect stuck
        If Robot has not moved during STUCK_CHECK_INTERVAL within STUCK_RADIUS, the stack is considered to be continuous
            Returns: stucking time
            Args: current_pos: current position of the robot
        """
        if self._stuck_previous_checked_pos is None:
            self._stuck_previous_checked_pos = current_pos

        self._timer_for_stuck += self._params.control_interval_time

        stucking_time = current_stucking_time
        if self._timer_for_stuck >= self._params.stuck_check_interval:
            # Check stuck
            if (
                np.linalg.norm(current_pos - self._stuck_previous_checked_pos)
                < self._params.stuck_radius
            ):
                # Not moving -> accumulate stuck count
                stucking_time += self._params.stuck_check_interval
            else:
                stucking_time = 0.0

            self._stuck_previous_checked_pos = current_pos
            self._timer_for_stuck = 0.0

        return stucking_time

    # Detect oscillation, but not good accuracy
    def _detect_oscillation(
        self, current_pos: np.ndarray, current_oscillating_time
    ) -> float:
        """
        Detect oscillation of robot in the same place
        If Robot has not moved during OSCILLATION_CHECK_INTERVAL within OSCILLATION_RADIUS, the oscillation is considered to be continuous
            Returns: oscillating time
            Args: current_pos: current position of the robot
        """
        if self._oscillation_previous_checked_pos is None:
            self._oscillation_previous_checked_pos = current_pos

        self._timer_for_oscillation += self._params.control_interval_time

        oscillating_time = current_oscillating_time

        if self._timer_for_oscillation >= self._params.oscillation_check_interval:
            # Relative pose to goal
            curr_relative_pos = current_pos - self._params.goal_state.pos
            prev_relative_pos = (
                self._oscillation_previous_checked_pos - self._params.goal_state.pos
            )

            diff_dist_to_goal = np.linalg.norm(curr_relative_pos) - np.linalg.norm(
                prev_relative_pos
            )
            diff_angle_to_goal = np.arctan2(
                curr_relative_pos[1], curr_relative_pos[0]
            ) - np.arctan2(prev_relative_pos[1], prev_relative_pos[0])

            if self._prev_diff_angle_to_goal is None:
                self._prev_diff_angle_to_goal = diff_angle_to_goal

            # initialize deque
            while len(self._flipping_counter_deque) < self._params.flipping_count_num:
                self._flipping_counter_deque.append(np.sign(diff_angle_to_goal))

            # add new value
            if np.abs(diff_angle_to_goal) > self._params.flipping_angle_threshold:
                self._flipping_counter_deque.append(np.sign(diff_angle_to_goal))

            # If more than half of the signs are different, flip judgment
            num_plus = np.sum(np.array(self._flipping_counter_deque) > 0)
            num_minus = self._params.flipping_count_num - num_plus
            diff_sign = np.abs(num_plus - num_minus)
            if diff_sign < self._params.flipping_count_num / 2:
                # Flip
                self._flipping_num += 1
            else:
                # not flip
                self._flipping_num = 0

            # Judge oscillation
            if (
                diff_dist_to_goal <= self._params.oscillation_goal_dist_threshold
                and self._flipping_num >= self._params.flipping_num_threshold
            ):
                # Not moving to goal and flipping -> accumulate oscillation count
                oscillating_time += self._params.oscillation_check_interval
            else:
                oscillating_time = 0.0

            self._oscillation_previous_checked_pos = current_pos
            self._prev_diff_angle_to_goal = diff_angle_to_goal
            self._timer_for_oscillation = 0.0

        return oscillating_time

    def _get_goal_time_from_table(
        self,
        pos: np.ndarray,
        goal_time_map: np.ndarray,
        origin: np.ndarray,
        resolution: float,
    ) -> float:
        """
        Get goal time from hash table
        Returns: goal time
        Args: pos: current position of the robot
              map: goal time map
              origin: origin of the costmap
              resolution: resolution of the goal time map
        return: optimal goal time
        """
        x = int((pos[0] - origin[0]) / resolution)
        y = int((pos[1] - origin[1]) / resolution)
        return goal_time_map[y, x]

    def _calc_optimal_goal_map(
        self, start_pos: np.ndarray, start_obs: RobotObservation, resolution: float
    ) -> np.ndarray:
        """
        Create hash table for optimal goal time to goal.
        param: resolution: resolution of the map [m]
        param: start_pos: start position of the robot [m] (x,y)
        param: start_obs: start observation of the robot
        return: optimal goal time map
        -----> y
        |
        |
        x
        """
        # prepare map
        map_size = self._static_map.get_map_size()
        x_num = int(map_size[0] / resolution)
        y_num = int(map_size[1] / resolution)
        map = np.zeros((x_num, y_num))

        # calculate optimal goal time for each grid
        self._global_planner.set_costmap(start_obs.static_map_with_scan)
        for x in range(map.shape[0]):
            for y in range(map.shape[1]):
                origin = self._static_map.get_origin()
                pos_x = origin[0] + x * resolution
                pos_y = origin[1] + y * resolution
                reference_indices, reference_path = self._global_planner.make_plan(
                    start_pos=start_pos, goal_pos=np.array([pos_x, pos_y])
                )
                if reference_path is None or len(reference_indices) == 1:
                    # Not reachable
                    map[y, x] = None
                else:
                    optimal_goal_time, _ = self._estimate_optimal_scores(
                        reference_path[0]
                    )
                    map[y, x] = max(
                        optimal_goal_time, 0.1
                    )  # minimum goal time is 0.1 [s] to avoid division by zero

        # interpolate None values in the map
        x = np.arange(0, map.shape[0])
        y = np.arange(0, map.shape[1])
        array = np.ma.masked_invalid(map)
        xx, yy = np.meshgrid(y, x)
        x1 = xx[~array.mask]
        y1 = yy[~array.mask]
        newarr = array[~array.mask]
        map_interpolated = interpolate.griddata(
            (y1, x1), newarr.ravel(), (yy, xx), method="nearest", fill_value=None
        )

        return map_interpolated

    def _reset_feasible_scenario(self) -> float:
        """
        Reset feasible scenario
        returns: estimated goal reaching time and path length by initial planned reference path
        """

        # reset simulator with static map and obstacles
        self._params.reset_static_map()
        self._params.reset_robot_initial_state()
        self._params.reset_moving_obstacles()
        self._params.reset_goal_state()
        self._static_map = self._simulator.reset(self._params)

        # reset goal state and initial state
        is_valid = False
        while not is_valid:
            # set params
            self._static_map = self._simulator.reset(self._params)

            # check initial state collision
            margin = 0.3
            is_collision = self._simulator.get_merged_map().check_collision(
                self._params.robot_initial_state.pos, self._params.robot_radius + margin
            )
            if is_collision:
                is_valid = False
                self._params.reset_robot_initial_state()
                continue

            # check goal state collision with static map and moving obstacles
            margin = 0.3
            is_collision = self._static_map.check_collision(
                self._params.goal_state.pos, self._params.robot_radius + margin
            )
            is_collision = self._simulator.get_merged_map().check_collision(
                self._params.goal_state.pos, self._params.robot_radius + margin
            )
            if is_collision:
                is_valid = False
                self._params.reset_goal_state()
                continue

            # Run global planner
            robot_obs = self._simulator.get_observation()
            self._global_planner.set_costmap(robot_obs.static_map_with_scan)
            reference_indices, reference_path = self._global_planner.make_plan(
                start_pos=robot_obs.state.pos, goal_pos=self._params.goal_state.pos
            )

            # check feasibility
            if reference_path is None or len(reference_indices) == 1:
                is_valid = False
                self._params.reset_robot_initial_state()
                self._params.reset_goal_state()
                # Retry
                continue

            # Estimate goal score
            optimal_goal_time, optimal_path_length = self._estimate_optimal_scores(
                reference_path[0]
            )

            # check enough length of start to goal
            map_size = self._static_map.get_map_size()
            if optimal_path_length < map_size[0] * self._params.min_path_length_ratio:
                is_valid = False
                self._params.reset_robot_initial_state()
                self._params.reset_goal_state()
                # Retry
                continue
            is_valid = True

        # calculate optimal goal time table
        optimal_goal_time_table = self._calc_optimal_goal_map(
            start_pos=self._params.robot_initial_state.pos,
            start_obs=robot_obs,
            resolution=self._optimal_goal_time_table_resolution,
        )
        # print("optimal_goal time: ", optimal_goal_time)
        # print("optimal_goal time (map)", self._get_goal_time_from_table(self._params.goal_state.pos, optimal_goal_time_table, self._static_map.get_origin(), self._optimal_goal_time_table_resolution))
        # plt.imshow(optimal_goal_time_table, cmap='gray')
        # plt.show()
        # exit()

        return optimal_goal_time, optimal_path_length, optimal_goal_time_table

    def _estimate_optimal_scores(
        self, reference_path: np.ndarray
    ) -> Tuple[float, float]:
        """
        Estimate goal scores: optimal goal time and path length
        Args:
            reference_path: reference path (N, 2)
        Returns:
            optimal_goal_time: estimated goal time [s]
            optimal_path_length: estimated path length [m]
        """
        # path smoothing by dynamic window average
        _, optimal_path_length = self._path_smoothing(reference_path)

        # estimate goal time
        max_speed = self._local_planner.get_max_speed()
        if max_speed <= 0.0:
            raise ValueError("max_speed is zero or negative")

        optimal_goal_time = optimal_path_length / max_speed

        return optimal_goal_time, optimal_path_length

    def _path_smoothing(self, path: np.ndarray) -> Tuple(np.ndarray, float):
        """
        Path smoothing by dynamic window average
        Args: path: path (N, 2)
        Returns:
            smoothed path (N, 2)
            path length
        """
        # smoothing initial reference path by dynamic window
        window_size = 5
        smoothed_path = []
        for i in range(len(path)):
            lower_bound = max(0, i - window_size)
            upper_bound = min(len(path), i + window_size)
            mean_pos = np.mean(path[lower_bound:upper_bound], axis=0)
            smoothed_path.append(mean_pos)
        smoothed_path = np.array(smoothed_path)

        # calculate path length
        length = 0
        for i in range(len(smoothed_path) - 1):
            length += np.linalg.norm(smoothed_path[i] - smoothed_path[i + 1])

        return smoothed_path, length

    def _goal_time_score(
        self, optimal_goal_time: float, actual_goal_time: float
    ) -> float:
        lower = 1.0 * optimal_goal_time
        upper = 4.0 * optimal_goal_time

        # This score: 0.25 ~ 1.0
        score = optimal_goal_time / np.clip(actual_goal_time, lower, upper)

        return score

    def _SGT(
        self, optimal_goal_time: float, actual_goal_time: float, is_collision: bool
    ) -> float:
        """
        Success weighted (normalized) Goal Time
        score: 0.0 ~ 1.0, 0.0
                 0.0: collision
                 2.5: not reach goal and not collision
                 1.0: optimal goal time
        """
        if is_collision:
            return 0.0
        else:
            return self._goal_time_score(optimal_goal_time, actual_goal_time)

    def _SPL(
        self, optimal_path_length: float, actual_path_length: bool, is_goal: bool
    ) -> float:
        """
        Success weighted (normalized) Path Length
        score: 0.0 ~ 1.0
        """
        if is_goal:
            return optimal_path_length / max(optimal_path_length, actual_path_length)
        else:
            return 0.0

    def _rotate_pos(self, pos: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate position
        Args:
            pos: position (x, y)
            angle: angle (rad)
        Returns:
            rotated position (x, y)
        """
        rot_mat = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        return np.matmul(rot_mat, pos)

    def _rotate_points(self, points: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate path
        Args:
            path: path (N, 2)
            angle: angle (rad)
        Returns:
            rotated path (N, 2)
        """
        rotated_path = []
        for pos in points:
            rotated_path.append(self._rotate_pos(pos, angle))
        return np.array(rotated_path)

    def _get_obs_dict(
        self,
        current_robot_obs: RobotObservation,
        current_reference_path: np.ndarray,
        goal_robot_obs: RobotObservation,
    ) -> OrderedDict:
        """
        Helper to create the observation.
        :return: (OrderedDict<int or ndarray>)
        """
        current_obs = self._observe(
            current_robot_obs, current_reference_path, is_set_observation_arr=True
        )

        # reference path at goal
        goal_reference_path = np.ndarray((1, 1, 2))
        goal_reference_path[0, 0, :] = goal_robot_obs.state.pos
        goal_obs = self._observe(
            goal_robot_obs, goal_reference_path, is_set_observation_arr=False
        )

        return OrderedDict(
            [
                ("observation", current_obs.copy()),
                ("achieved_goal", current_obs.copy()),
                ("desired_goal", goal_obs.copy()),
            ]
        )

    def _observe(
        self,
        robot_obs: RobotObservation,
        global_reference_path: np.ndarray,
        is_set_observation_arr=True,
    ) -> np.ndarray:
        """
        Observe the environment for RL
            Returns: observation vector
            Args:
                robot_obs: robot observation
                global_path: reference path calculated by global planner (1, N, 2)
                robot_traj: robot trajectory (N, 2)
        """

        observed_vec_list = []

        ## Current pos and angle ##
        current_pos = robot_obs.state.pos
        current_yaw = robot_obs.state.yaw
        current_pose = np.array([current_pos[0], current_pos[1], current_yaw])
        observed_vec_list += list(current_pose)

        ## current time step ##
        observed_vec_list.append(float(self._step_count))

        ## vector to Goal ##
        vec_to_goal = self._params.goal_state.pos - robot_obs.state.pos
        rotated_vec_to_goal = self._rotate_pos(vec_to_goal, -robot_obs.state.yaw)
        observed_vec_list += list(rotated_vec_to_goal)

        ## Global planner is locked or not ##
        # Dim: 1 (0 ~ self.MAX_HZ_FOR_GLOBAL_PLANNER) if 0, global planner is not locked
        observed_vec_list.append(float(self._counter_for_global_planner_lock))

        ## scans ##
        if (
            robot_obs.relative_poses_scan_points is not None
            and len(robot_obs.relative_poses_scan_points) > 0
        ):
            scan_poses = robot_obs.relative_poses_scan_points[:, :2]
        else:
            scan_poses = np.array([[0, 0]])
        compressed_relative_scans = self._reshape_arr(
            scan_poses, (self._rl_params.dim_scan, 2)
        )
        # (N, 2) -> (2N, )
        compressed_relative_scans = self._rotate_points(
            compressed_relative_scans, -robot_obs.state.yaw
        )
        # add current scan to accumulated scans
        if self._accumulated_scans is None or len(self._accumulated_scans) == 0:
            # fill accumulated scans with current scan
            while len(self._accumulated_scans) < int(
                self._rl_params.accumulated_scan_horizon
                / self._params.control_interval_time
            ):
                self._accumulated_scans.append(compressed_relative_scans)
        else:
            # append current scan to accumulated scans
            self._accumulated_scans.append(compressed_relative_scans)

        # Picked up scans from accumulated scans
        picked_scans = []
        for i in range(self._rl_params.num_scans):
            index_interval = int(
                self._rl_params.scan_interval / self._params.control_interval_time
            )
            index = min(1 + i * index_interval, len(self._accumulated_scans))
            picked_scans.append(self._accumulated_scans[-index])

        # (N, 2) -> (2N, )
        for picked_scan in picked_scans:
            observed_vec_list += list(picked_scan.reshape(-1))

        # reference paths
        nearest_index = self._find_nearest_index(
            global_reference_path[0], robot_obs.state.pos
        )
        clipped_path = self._slice(
            global_reference_path[0][nearest_index:],
            self._rl_params.range_reference_path,
            from_back=False,
        )
        relative_clipped_path = clipped_path - robot_obs.state.pos
        compressed_clipped_path = self._reshape_arr(
            relative_clipped_path, (self._rl_params.dim_reference_path, 2)
        )
        compressed_clipped_path = self._rotate_points(
            compressed_clipped_path, -robot_obs.state.yaw
        )
        observed_vec_list += list(compressed_clipped_path.reshape(-1))

        # Visualize observation
        if (
            self._visualize_mode == self.VisualizeMode.observation
            and is_set_observation_arr
        ):
            fig, axes = plt.subplots(1, 1, figsize=(5, 5))
            s = 10.0

            axes.scatter(
                compressed_clipped_path[:, 0], compressed_clipped_path[:, 1], s=s, c="g"
            )

            for i, scan in enumerate(picked_scans):
                axes.scatter(scan[:, 0], scan[:, 1], s=s, c="r")
            # vec to goal
            axes.scatter(rotated_vec_to_goal[0], rotated_vec_to_goal[1], s=s, c="k")

            axes.set_xlim(-10, 10)
            axes.set_ylim(-10, 10)
            fig.tight_layout()
            axes, plt.axis("off")

            # convert to numpy array
            io_buf = io.BytesIO()
            fig.savefig(io_buf, format="raw")
            plt.close(fig)
            io_buf.seek(0)
            self._observation_image_arr = np.reshape(
                np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1),
            )
            io_buf.close()

        # list to array
        observed_vec = np.array(observed_vec_list, dtype=self.D_TYPE)

        # check data type
        if observed_vec.dtype != self.D_TYPE:
            print("observed_vec.dtype: ", observed_vec.dtype)
            raise ValueError("observed_vec.dtype != self.D_TYPE")

        return observed_vec

    def _calc_obstacle_dists(self, scan_relative_poses: np.ndarray) -> np.ndarray:
        """
        Calculate relative distances of obstacles from robot with given FOV and region
        Args:
            scan_relative_poses: (N, 3) [x,y, yaw]
        Return:
            obstacle_dists: (FOV_DIVIDE_NUM, 1)
        """
        obstacle_dists = np.zeros(shape=(self.FOV_DIVIDE_NUM,), dtype=self.D_TYPE)
        angle = (self.FOV[1] - self.FOV[0]) / self.FOV_DIVIDE_NUM

        if scan_relative_poses.shape[0] == 0:
            return obstacle_dists

        for i in range(self.FOV_DIVIDE_NUM):
            min_angle = self.FOV[0] + i * angle
            max_angle = self.FOV[0] + (i + 1) * angle
            # scan relative posesの中で, min_angleとmax_angleを含むものを抽出
            candidates = scan_relative_poses[
                (scan_relative_poses[:, 2] >= min_angle)
                & (scan_relative_poses[:, 2] <= max_angle)
            ]
            # candidatesの中で最小の距離を計算
            if candidates.shape[0] > 0:
                min_dist = np.min(np.linalg.norm(candidates[:, :2], axis=1))
                obstacle_dists[i] = min_dist
            else:
                # No scan point in this region
                obstacle_dists[i] = -1  # free space
        return obstacle_dists

    def _calc_relative_traj(
        self,
        robot_traj: np.ndarray,
        source_pos: np.ndarray,
        sampled_num: int,
        interval: float,
    ) -> np.ndarray:
        """
        Calculate relative robot trajectory from goal position from latest pos to the past of interval time
        Args:
            robot_traj: (N, 2)
            source_pos: (2,) source position from which the relative trajectory is calculated
            sampled_num: int : number of points to be used for relative trajectory
            interval : float : interval time of robot trajectory from latest point
        Return:
            relative_traj: (N, 2)
        """

        if robot_traj.shape[0] == 0:
            raise ValueError("robot_traj.shape[0] == 0")

        if sampled_num == 0:
            return None

        candidate_num = int(interval / self._params.control_interval_time)

        # get candidate points from latest point
        candidate_traj = robot_traj[-candidate_num:]

        # sampling by moving average
        sampled_traj = self._reshape_arr(candidate_traj, (sampled_num, 2))

        # transform to relative trajectory
        relative_traj = sampled_traj - source_pos

        return relative_traj

    def _find_nearest_index(self, points: np.ndarray, point: np.ndarray) -> int:
        """
        Find nearest point index from given points
        Args:
            points: (N, 2)
            point: (2,)
        Return:
            nearest_point_index: int
        """
        dists = np.linalg.norm(points - point, axis=1)
        nearest_point_index = np.argmin(dists)
        return nearest_point_index

    def _calc_deviation_from_reference_path(
        self, reference_path: np.ndarray, robot_pos: np.ndarray
    ) -> float:
        """
        Calculate deviation from reference path
        Args:
            reference_path: (1, N, 2) [x,y]
            robot_pos: (2, ) [x,y]
        Return:
            deviation: float
        """
        relative_pos = reference_path[0] - robot_pos
        dists = np.linalg.norm(relative_pos, axis=1)
        index = np.argmin(dists)
        min_dist = dists[index]
        return min_dist, index

    def _slice(self, arr, size, from_back=False):
        if from_back:
            if arr.shape[0] < size:
                return arr
            else:
                return arr[-size:]
        else:
            if arr.shape[0] < size:
                return arr
            else:
                return arr[:size]

    def _reshape_arr(
        self, arr: np.ndarray, shape: tuple, from_back=False
    ) -> np.ndarray:
        """
        Reshape array to given shape with moving average
        Args:
            arr: (N, ....)
            shape: (M, ....)
            from_back: bool : if True, the last element of arr is the latest element
        Return:
            reshaped_arr: (M, ....)
        """
        new_arr = np.zeros(shape=shape, dtype=self.D_TYPE)

        if arr is None:
            raise ValueError("arr is None")

        if arr.shape[0] == 0:
            return new_arr

        if shape[0] == 0:
            return new_arr

        if arr.shape[0] == shape[0]:
            return arr
        elif arr.shape[0] < shape[0]:
            # fill blank with a value
            if from_back:
                # fill with the last value
                # [x,y] -> [x,y,y, ...]
                new_arr[: arr.shape[0]] = arr
                new_arr[arr.shape[0] :] = arr[-1]
            else:
                # fill with the first value
                # [x, y] -> [x, x, y, ...]
                new_arr[: shape[0] - arr.shape[0]] = arr[0]
                new_arr[-arr.shape[0] :] = arr
        else:
            # Convolve (N, ...) -> (M, ...), N > M
            new_arr = self._convolve(arr, shape[0], from_back)

        if new_arr.shape != shape:
            print("new_arr.shape", new_arr.shape)
            print("shape", shape)
            # raise ValueError('new_arr.shape != shape')

        return new_arr

    def _convolve(self, arr: np.ndarray, size: int, from_last: bool) -> np.ndarray:
        """
        Convolve array by moving average, N -> M (N > M)
        args:
            arr: (N, L)
            size: int
        return:
            convolved_arr: (M, L)
        """
        if arr.shape[0] < size:
            raise ValueError("arr.shape[0] < size")

        convolved_arr = np.zeros(shape=(size, arr.shape[1]), dtype=self.D_TYPE)
        window_size = int(arr.shape[0] / size)

        if not from_last:
            # from front
            for i in range(size):
                # moving average
                # convolved_arr[i] = np.mean(arr[i * window_size:(i + 1) * window_size], axis=0)
                # sampling
                convolved_arr[i] = arr[i * window_size]
        else:
            # from back
            for i in range(size):
                lower_bound = max(0, arr.shape[0] - (i + 1) * window_size)
                upper_bound = min(arr.shape[0], arr.shape[0] - i * window_size)
                # moving average
                # convolved_arr[i] = np.mean(arr[lower_bound:upper_bound], axis=0)
                # sampling
                convolved_arr[i] = arr[upper_bound - 1]
        return convolved_arr

    def _convert_to_obs_vector(self, arr: np.ndarray, batch_size: int) -> np.ndarray:
        """
        Convert array to observation vector separated by batch size
        arr: (batch_size, m) or (N,)
        return: (batch_size, N/batch_size or m)
        """

        state = np.array(arr).reshape(batch_size, -1)

        return state

    # compute reward from vectorized and not vectorized observations and infos
    # The desired goal can be moved by HER computation, so reward shold be only dependent on the observation
    # Do not use goal or not information in the infos because the infos are not changed by HER computation
    def compute_reward(
        self,
        achieved_goals: np.ndarray,
        desired_goals: np.ndarray,
        infos: Union[dict, np.ndarray[dict]],
    ) -> np.float32:
        """
        Compute reward for HER
        Args:
            achieved_goals: current observation (batch_size, dim_observation) or (dim_observation, )
            desired_goals: Changeable observation of goal (batch_size, dim_observation) or (dim_observation, )
            infos: infos (batch_size, ) NOTE: infos are not changed by HER computation
        Return:
            reward: np.float32 (batch_size, )
        """
        # As using vectorized env, achieved_goal and desired_goal are batched
        batch_size = achieved_goals.shape[0] if len(achieved_goals.shape) > 1 else 1

        # (batch_size, dim_observation)
        desired_goals = self._convert_to_obs_vector(desired_goals, batch_size)
        achieved_goals = self._convert_to_obs_vector(achieved_goals, batch_size)

        if batch_size == 1:
            infos = np.array([infos])

        # We return the reward as a float32 array to use vectorized envs
        rewards = np.zeros(shape=(batch_size,), dtype=np.float32)

        for i in range(batch_size):
            current_pos = achieved_goals[i][0:3]
            goal_pos = desired_goals[i][0:3]

            is_goal_reached = self._local_planner.is_goal_reached_np(
                current_pos, goal_pos
            )

            if infos[i]["is_collision"]:
                rewards[i] = self._rl_params.collision_penalty
            elif infos[i]["is_timeout"]:
                rewards[i] = 0.0  # TODO: optional reward
            elif is_goal_reached:
                # goal score
                time_step = achieved_goals[i][3]
                optimal_goal_time_table = infos[i]["optimal_goal_time_table"]
                # optimal goal time is calculated here from the desired goals
                # because the desired goals can be moved by HER computation
                optimal_goal_time = self._get_goal_time_from_table(
                    goal_pos,
                    optimal_goal_time_table,
                    self._static_map.get_origin(),
                    self._optimal_goal_time_table_resolution,
                )
                rewards[i] = self._rl_params.sgt_reward * self._goal_time_score(
                    optimal_goal_time=optimal_goal_time,
                    actual_goal_time=time_step * self._params.control_interval_time,
                )
            else:
                rewards[i] = 0.0

        return rewards

    def _forward_simulation(
        self,
        current_robot_obs: RobotObservation,
        current_obstacles: list,
        current_robot_traj: np.ndarray,
        initial_replan: bool,
        prediction_length: int,
    ) -> list[np.ndarray]:
        """
        Forward simulate robot state with action for counterfactual evaluation
        return:
            future_robot_obs_list
        """
        observation_list = []

        self._counterfactual_simulator.set_robot_state_and_obstacles_and_traj(
            current_robot_obs.state, current_obstacles, robot_traj=current_robot_traj
        )

        reference_path = self._previous_global_path
        reference_path_indices = self._previous_global_path_indices

        for i in range(prediction_length):
            # action = Action.not_update_reference_path
            # Not update global planner
            robot_obs = self._counterfactual_simulator.get_observation()

            if initial_replan and i == 0:
                # replan reference path once
                self._global_planner.set_costmap(robot_obs.static_map_with_scan)
                reference_path_indices, reference_path = self._global_planner.make_plan(
                    robot_obs.state.pos, self._goal_state.pos
                )

            if len(reference_path_indices) == 1:
                # Not found path
                reference_path = self._previous_global_path
                reference_path_indices = self._previous_global_path_indices

            self._local_planner.set_costmap(robot_obs.static_map_with_scan)
            local_planner_output = self._local_planner.compute_velocity_command(
                robot_obs, self._goal_state, reference_path
            )

            # update simulator
            obstacle_map, robot_traj = self._counterfactual_simulator.step(
                local_planner_output.control_command
            )

            # Prepare for rendering
            # inflated_map = self._global_planner.get_costmap()
            inflated_map = self._local_planner.get_costmap()

            # observe
            observation = self._observe(
                robot_obs=robot_obs,
                global_path=reference_path,
                robot_traj=robot_traj,
                is_set_observation_arr=False,
            )

            observation_list += [observation]

        return observation_list


register(
    id="NavigationStackGoalEnv-v0",
    entry_point="navigation_stack_py.gym_env.navigation_stack_goal_env:NavigationStackGoalEnv",
)
