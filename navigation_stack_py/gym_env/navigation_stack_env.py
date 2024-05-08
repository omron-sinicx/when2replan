from __future__ import annotations
from cmath import inf
from enum import IntEnum, Enum
from typing import Tuple, List

import io
import numpy as np
import gym
from gym import spaces
from matplotlib import pyplot as plt
from collections import defaultdict, deque
from gym.envs.registration import register

from navigation_stack_py.global_planner import (
    DijkstraPlanner,
    RRTStarPlanner,
    RRTPlanner,
    PRMPlanner,
)
from navigation_stack_py.local_planner import (
    DWAPlanner,
    LocalPlannerOutput,
    RandomizedMPCPlanner,
)
from navigation_stack_py.simulator import Simulator
from navigation_stack_py.utils import visualizer, ParameterHandler, MapHandler
from navigation_stack_py.common import RobotObservation, AgentState, MovingObstacle


class Action(IntEnum):
    not_update_reference_path: int = 0
    update_reference_path: int = 1

    # For baseline
    force_update_reference_path: int = -1


class RLParameters:
    def __init__(self, config: dict = None):
        if config is None:
            print("Default reward and observation parameters are used.")

            ## Reward parameters ##
            # Positive reward
            self.sgt_reward = 1.0
            self.spl_reward = 0.0
            self.speed_reward = 0.0
            # Negative penalty
            self.collision_penalty = 0.0
            self.replan_penalty = 0.0

            ## Observation parameters ##
            self.dim_scan = 80  # Number of scan points used for observation
            self.num_scans = 1  # Number of accumulated scans
            self.scan_interval = 1.5  # Interval between two scans to be accumulated
            self.accumulated_scan_horizon = 3.0  # Horizon of accumulated scans
            self.dim_previous_path = (
                25  # Number of previous path points used for observation
            )
            self.range_previous_path = 50  # Horizon of previous path points, i.e. use the number of points from current position
            self.dim_reference_path = (
                25  # Number of reference path points used for observation
            )
            self.num_reference_paths = (
                1  # Number of reference paths used for observation
            )
            self.range_reference_path = 50  # Horizon of reference path points, i.e. use the number of points from current position
            self.dim_goal = 2  # dimension of relative goal pose used for observation
            self.dim_global_planner_status = (
                1  # dimension of global planner status used for observation
            )

            ## Action parameters ##
            self.num_actions = 2  # Number of actions [0: not update reference path, 1: update reference path]

        else:
            ## Reward parameters ##
            reward = config["reward"]
            self.sgt_reward = reward["sgt_reward"]
            self.spl_reward = reward["spl_reward"]
            self.speed_reward = reward["speed_reward"]
            self.collision_penalty = reward["collision_penalty"]
            self.replan_penalty = reward["replan_penalty"]

            ## Observation parameters ##
            observation = config["observation"]
            self.dim_scan = observation["dim_scan"]
            self.num_scans = observation["num_scans"]
            self.scan_interval = observation["scan_interval"]
            self.accumulated_scan_horizon = observation["accumulated_scan_horizon"]
            self.dim_previous_path = observation["dim_previous_path"]
            self.range_previous_path = observation["range_previous_path"]
            self.dim_reference_path = observation["dim_reference_path"]
            self.num_reference_paths = observation["num_reference_path"]
            self.range_reference_path = observation["range_reference_path"]
            self.dim_goal = observation["dim_goal"]
            self.dim_global_planner_status = observation["dim_global_planner_status"]

            ## Action parameters ##
            action = config["action"]
            self.num_actions = action["num_actions"]

        # Dimension of observation, +1 for available global planner or not
        self.dim_observation = (
            2
            * (
                self.dim_scan * self.num_scans
                + self.dim_previous_path * self.num_reference_paths
                + self.dim_reference_path
            )
            + self.dim_goal
            + self.dim_global_planner_status
        )


class NavigationStackEnv(gym.Env):
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
        navigation_config: dict,
        scenario_list: list[dict],
        rl_config: dict = None,
        seed: int = 0,
        visualize_mode: str = "none",
        save_fig: bool = False,
        save_visualized_data: bool = False,
    ):
        super(NavigationStackEnv, self).__init__()

        # Parameter handler
        if navigation_config is None:
            raise ValueError("navigation_config_path is None")
        else:
            self._navigation_config = navigation_config

        if scenario_list is None or len(scenario_list) == 0:
            raise ValueError("scenario_list is None or empty")
        else:
            self._scenario_list = scenario_list

        if visualize_mode == "observation":
            self._visualize_mode = self.VisualizeMode.observation
        elif visualize_mode == "birdeye":
            self._visualize_mode = self.VisualizeMode.birdeye
        elif visualize_mode == "none":
            self._visualize_mode = self.VisualizeMode.none
        else:
            raise ValueError("Invalid visualize_mode: {}".format(visualize_mode))

        self._save_fig = save_fig
        
        self._save_visualized_data = save_visualized_data

        # for random baseline
        self._rng = np.random.default_rng(seed)

        self._seed = seed
        self._params = ParameterHandler(seed=seed)
        scenario = self._rng.choice(self._scenario_list)
        self._params.init(navigation_config=self._navigation_config, scenario=scenario)

        # Simulator
        self._simulator = Simulator()

        # Global Planner
        if self._params.global_planner_type == "RRT":
            self._global_planner = RRTPlanner(self._params)
        elif self._params.global_planner_type == "RRT-star":
            self._global_planner = RRTStarPlanner(self._params)
        elif self._params.global_planner_type == "Dijkstra":
            self._global_planner = DijkstraPlanner(self._params)
        elif self._params.global_planner_type == "PRM":
            self._global_planner = PRMPlanner(self._params)
        else:
            raise ValueError(
                "Invalid global_planner_type: {}".format(
                    self._params.global_planner_type
                )
            )

        # Local Planner
        if self._params.local_planner_type == "DWA":
            self._local_planner = DWAPlanner(self._params)
        elif self._params.local_planner_type == "MPC":
            self._local_planner = RandomizedMPCPlanner(self._params)
        else:
            raise ValueError(
                "Invalid local_planner_type: {}".format(self._params.local_planner_type)
            )

        # params for RL
        self._rl_params = RLParameters(rl_config)
        self.action_space = spaces.Discrete(self._rl_params.num_actions)

        # observation space
        self.observation_space = spaces.Box(
            low=-inf,
            high=inf,
            shape=(self._rl_params.dim_observation,),
            dtype=self.D_TYPE,
        )

    def reset(self, reset_scenario: bool = True):
        """
        Reset environment with new variables
        """
        # reset new params with random seed
        if reset_scenario:
            # reset params
            scenario = self._rng.choice(self._scenario_list)
            self._params.init(
                navigation_config=self._navigation_config, scenario=scenario
            )
            # set feasible and valid static map, goal state, and initial state
            (
                self._optimal_goal_time,
                self._optimal_path_length,
                self._previous_global_path_indices,
                self._previous_global_path,
            ) = self._reset_feasible_scenario()
            self._goal_state = self._params.goal_state
        else:
            raise NotImplementedError("Not implemented yet")
            # self._static_map = self._simulator.reset(self._params)
            # self._goal_state = self._params.goal_state

        # reset variables

        ## image buffers ##
        self._birdeye_image_arr_buffer: list[np.ndarray] = []
        self._observation_image_arr_buffer: list[np.ndarray] = []
        self._static_map_image_arr_buffer: list[np.ndarray] = []
        self._obstacle_image_arr_buffer: list[np.ndarray] = []
        self._robot_image_arr_buffer: list[np.ndarray] = []
        self._traj_image_arr_buffer: list[np.ndarray] = []
        self._obstacle_traj_image_arr_buffer: list[np.ndarray] = []
        self._obstacles_history = deque(maxlen=30)

        self._step_count = 0
        self._stucking_time = 0.0
        self._average_stuck_time = 0.0
        self._max_stuck_time = 0.0
        self._stuck_previous_checked_pos = None
        self._timer_for_stuck = 0.0
        self._num_update_global_planner = 0
        self._accumulated_scans: deque = deque(
            maxlen=int(
                self._rl_params.accumulated_scan_horizon
                / self._params.control_interval_time
            )
        )
        self._reference_path_queue: deque = deque(
            maxlen=self._rl_params.num_reference_paths
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
        self._dist_to_goal = 0.0
        self._prev_dist_for_distance_based_baseline: float = 0.0
        self._counter_for_timer_triggered_baseline: int = (
            0  # for timer triggered baseline
        )

        observation, _, _, _ = self.step(action=Action.update_reference_path)

        return observation

    def _rollout_local_planner_and_simulator(
        self,
        global_reference_path: np.ndarray,
        global_reference_indices: np.ndarray,
        rollout_num: int,
    ) -> Tuple[np.ndarray,]:
        """
        Rollout local planner and simulator with given global path
        params:global_reference_path
        prams:global_reference_indices
        params:rollout_num: iteration number for rollout
        return:
            robot_observation, local_planner_output, robot_trajectory, new_step_count
        """
        if rollout_num < 1:
            raise ValueError("rollout_num must be greater than 0")
        
        visualized_data_list = []

        step_count = self._step_count
        for i in range(rollout_num):
            # step count
            step_count += 1

            # for timer triggered baseline
            self._counter_for_timer_triggered_baseline = self._count_down(
                self._counter_for_timer_triggered_baseline
            )

            # get robot observation from simulator
            robot_obs = self._simulator.get_observation()

            # update local planner
            self._local_planner.set_costmap(robot_obs.static_map_with_scan)
            local_planner_output = self._local_planner.compute_velocity_command(
                robot_obs, self._goal_state, global_reference_path
            )

            # update simulator
            obstacle_map, robot_traj = self._simulator.step(
                local_planner_output.control_command
            )

            # calculate traveled path length and distance to goal
            self._traveled_path_length += np.linalg.norm(
                robot_traj[-1] - robot_traj[-2]
            )
            self._dist_to_goal = np.linalg.norm(robot_traj[-1] - self._goal_state.pos)

            # Stuck detection
            self._stucking_time = self._detect_stuck(
                current_pos=robot_obs.state.pos,
                current_stucking_time=self._stucking_time,
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

            # Prepare for rendering
            if self._visualize_mode == self.VisualizeMode.birdeye:
                inflated_map = self._local_planner.get_costmap()
                obstacle_list = self._simulator.get_obstacles_status()
                self._obstacles_history.appendleft(obstacle_list)

                # all layer
                self._birdeye_image_arr_buffer.append(
                    visualizer.render(
                        static_map=self._static_map,
                        obstacle_map=obstacle_map,
                        obstacles_history=self._obstacles_history,
                        inflation_layer=inflated_map.get_inflation_layer(),
                        robot_observation=robot_obs,
                        robot_traj=robot_traj,
                        goal_state=self._goal_state,
                        robot_radius=self._params.robot_radius,
                        global_path=global_reference_indices,
                        local_path_list=local_planner_output.predict_path_list,
                        local_path_best_index=local_planner_output.best_index,
                        sub_goal_index=local_planner_output.sub_goal_index,
                        visualize_local_path=False,
                        is_replan=rollout_num != 1 and i == 0,
                    )
                )

                if self._save_fig:
                    inflated_map = self._local_planner.get_costmap()
                    obstacle_list = self._simulator.get_obstacles_status()
                    # separate layer to draw figure for paper
                    (
                        static_image_array,
                        obstacle_image_array,
                        robot_image_array,
                        traj_image_array,
                        obstacle_traj_array,
                    ) = visualizer.render_separate(
                        static_map=self._static_map,
                        obstacles=obstacle_list,
                        robot_observation=robot_obs,
                        robot_traj=robot_traj,
                        goal_state=self._goal_state,
                        robot_radius=self._params.robot_radius,
                        is_replan=rollout_num != 1 and i == 0,
                        time_step=step_count,
                        max_time_step=self._params.max_episode_steps,
                        is_end=self._end and i == rollout_num - 1,
                    )
                    self._static_map_image_arr_buffer.append(static_image_array)
                    self._obstacle_image_arr_buffer.append(obstacle_image_array)
                    self._robot_image_arr_buffer.append(robot_image_array)
                    self._traj_image_arr_buffer.append(traj_image_array)
                    self._obstacle_traj_image_arr_buffer.append(obstacle_traj_array)
                    
                # log visualized data
                if self._save_visualized_data:
                    visualized_data = {}
                    visualized_data["robot_state"] = robot_obs.state
                    
                    visualized_data["scan"] = robot_obs.scan_points.get_points()
                    
                    visualized_data["is_replan"] = rollout_num != 1 and i == 0
                    
                    visualized_data["local_path_list"] = local_planner_output.predict_path_list
                    
                    visualized_data["local_path_best_index"] = local_planner_output.best_index
                    
                    visualized_data["goal"] = self._goal_state.pos
                    
                    obstacle_list = self._simulator.get_obstacles_status()
                    
                    # Clip obstacle position to map boundary to visualize (This is due to a bug of pymap2d)
                    viz_obstacle_list = []
                    for obstacle in obstacle_list:
                        pos = self._static_map.clip(obstacle.pos)
                        viz_obstacle = MovingObstacle(
                            pos=pos,
                            yaw=obstacle.yaw,
                            linear_vel=obstacle.linear_vel,
                            target_vel=obstacle.target_vel,
                            angular_vel=obstacle.angular_vel,
                            size=obstacle.size,
                            shape=obstacle.shape,
                            motion_model=obstacle.motion_model,
                        )
                        viz_obstacle_list.append(viz_obstacle)
                              
                    visualized_data["obstacles"] = viz_obstacle_list
                    
                    visualized_data["global_path"] = global_reference_path[0]
                    
                    visualized_data_list.append(visualized_data)
            
            # If collision or goal reached, end rollout
            if robot_obs.is_collision or local_planner_output.is_goal_reached:
                break

        return robot_obs, local_planner_output, robot_traj, step_count, visualized_data_list

    def step(self, action: int):
        # check action in Action class
        if action not in [e.value for e in Action]:
            raise ValueError(f"action is not in Action class: {action}")

        # observation of robot
        robot_obs = self._simulator.get_observation()
        
        visualized_data_list = []
        is_replan = False
        global_reference_path = None
        global_reference_indices = None
        if action == Action.not_update_reference_path:
            is_replan = False
            # Rollout local planner and simulator with previous global path by one step
            (
                rollout_robot_obs,
                local_planner_output,
                robot_traj,
                self._step_count,
                visualized_data
            ) = self._rollout_local_planner_and_simulator(
                global_reference_path=self._previous_global_path,
                global_reference_indices=self._previous_global_path_indices,
                rollout_num=1,
            )
            global_reference_path = self._previous_global_path
            global_reference_indices = self._previous_global_path_indices
            visualized_data_list.extend(visualized_data)
        elif action == Action.update_reference_path:
            is_replan = True
            # Rollout local planner and simulator with previous global path by global_planner_calculation_interval - 1
            # This rollout is to emulate of updating global path in actual non-blocking system
            # global_planner_calculation_interval - 1 is because the last step is rollout with updated global path
            # if global_planner_calculation_interval==1 means that the global planner can plan within local planner control interval
            if (
                self._params.global_planner_calculation_interval > 1
                and self._step_count > 0
            ):
                (
                    rollout_robot_obs,
                    local_planner_output,
                    robot_traj,
                    self._step_count,
                    visualized_data
                ) = self._rollout_local_planner_and_simulator(
                    global_reference_path=self._previous_global_path,
                    global_reference_indices=self._previous_global_path_indices,
                    rollout_num=self._params._global_planner_calculation_interval - 1,
                )
                visualized_data_list.extend(visualized_data)
                
            # calculate global path with robot observation before rollout
            self._global_planner.set_costmap(robot_obs.static_map_with_scan)
            (
                global_reference_indices,
                global_reference_path,
            ) = self._global_planner.make_plan(
                robot_obs.state.pos, self._goal_state.pos
            )
            # If not found path, do not update global planner
            if global_reference_path is None or len(global_reference_indices) == 1 or len(global_reference_indices) == 0:
                global_reference_path = self._previous_global_path
                global_reference_indices = self._previous_global_path_indices
            else:
                # update previous global path
                self._previous_global_path = global_reference_path
                self._previous_global_path_indices = global_reference_indices
            
            # reference path queue
            self._reference_path_queue.append(global_reference_path)
            
            # log
            self._num_update_global_planner += 1

           # Rollout local planner and simulator with new global path by one step to get observation of RL
            (
                rollout_robot_obs,
                local_planner_output,
                robot_traj,
                self._step_count,
                visualized_data
            ) = self._rollout_local_planner_and_simulator(
                global_reference_path=global_reference_path,
                global_reference_indices=global_reference_indices,
                rollout_num=1,
            )
            visualized_data_list.extend(visualized_data)

        # stack reference path queue
        if (
            self._reference_path_queue is None
            or len(self._reference_path_queue) < self._rl_params.num_reference_paths
        ):
            while len(self._reference_path_queue) < self._rl_params.num_reference_paths:
                self._reference_path_queue.append(global_reference_path)

        # Observation of RL
        observation = self._observe(
            robot_obs=rollout_robot_obs,
            global_path=global_reference_path,
            robot_traj=robot_traj,
        )

        ## Done
        done: bool = False

        # If reach goal
        if local_planner_output.is_goal_reached:
            done = True

        # If exceed time limit
        if self._step_count >= self._params.max_episode_steps:
            done = True

        # If collision
        if rollout_robot_obs.is_collision:
            done = True

        # Calculate reward
        reward = self._reward(
            action=action,
            robot_obs=rollout_robot_obs,
            local_planner_output=local_planner_output,
            is_done=done,
        )

        info = {}

        # timeout suggestion
        if self._step_count >= self._params.max_episode_steps:
            info["TimeLimit.truncated"] = True
        else:
            info["TimeLimit.truncated"] = False

        # log variables
        data = {}
        data["action"] = int(action)
        data["seed"] = self._seed
        data["time_step"] = self._step_count
        data["time"] = self._step_count * self._params.control_interval_time
        data["optimal_goal_time"] = self._optimal_goal_time
        data["collision"] = int(rollout_robot_obs.is_collision)
        data["collision_time"] = (
            float(rollout_robot_obs.is_collision) * self._params.control_interval_time
        )
        data["stuck_time"] = self._stucking_time
        data["average_stuck_time"] = self._average_stuck_time
        data["max_stuck_time"] = self._max_stuck_time
        data["is_oscillation"] = self._is_oscillation
        data["oscillation_time"] = self._oscillation_time
        data["num_oscillation"] = self._num_oscillation
        data["linear_speed"] = np.linalg.norm(rollout_robot_obs.state.linear_vel)
        data["travel_distance"] = self._traveled_path_length
        data["update_global_planner"] = int(is_replan)
        data["num_update_global_planner"] = self._num_update_global_planner
        data["reach_goal"] = int(local_planner_output.is_goal_reached)
        if local_planner_output.is_goal_reached:
            data["goal_time"] = self._step_count * self._params.control_interval_time
        else:
            data["goal_time"] = -1

        if self._step_count >= self._params.max_episode_steps:
            data["reach_max_time"] = 1
        else:
            data["reach_max_time"] = 0

        # metric for navigation performance
        if done:
            # Success score weighted goal time
            data["SGT"] = self._SGT(
                optimal_goal_time=self._optimal_goal_time,
                actual_goal_time=self._step_count * self._params.control_interval_time,
                is_collision=rollout_robot_obs.is_collision,
            )
            # Success score weighted path length
            data["SPL"] = self._SPL(
                optimal_path_length=self._optimal_path_length,
                actual_path_length=self._traveled_path_length,
                is_goal=local_planner_output.is_goal_reached,
            )
        else:
            # Not goal yet
            data["SGT"] = 0.0
            data["SPL"] = 0.0

        data["done"] = done
        data["reward"] = reward

        info["data"] = data
        info["visualized_data_list"] = visualized_data_list

        return observation, reward, done, info

    def render(self, mode="rgb_array") -> list[np.ndarray]:
        """
        get the rendered image buffer and delete the buffer
        return: N x (x, y, 3)
        N: number of images
        """
        render_image_list = []

        if self._visualize_mode == self.VisualizeMode.observation:
            # Observation image
            if (
                self._observation_image_arr_buffer is None
                or len(self._observation_image_arr_buffer) == 0
            ):
                raise ValueError("No render images")
            render_image_list = self._observation_image_arr_buffer
            self._observation_image_arr_buffer = []
        elif self._visualize_mode == self.VisualizeMode.birdeye:
            # birds eye view image
            if (
                self._birdeye_image_arr_buffer is None
                or len(self._birdeye_image_arr_buffer) == 0
            ):
                raise ValueError("No render images")
            render_image_list = self._birdeye_image_arr_buffer
            self._birdeye_image_arr_buffer = []
        elif self._visualize_mode == self.VisualizeMode.none:
            pass

        return render_image_list

    def get_layer_buffers(
        self,
    ) -> Tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        return (
            self._static_map_image_arr_buffer,
            self._obstacle_image_arr_buffer,
            self._robot_image_arr_buffer,
            self._traj_image_arr_buffer,
            self._obstacle_traj_image_arr_buffer,
        )

    def close(self):
        plt.close()

    def get_max_steps(self) -> int:
        return self._params.max_episode_steps

    def get_max_global_hz(self) -> int:
        # raise NotImplementedError()
        return self._params.global_planner_calculation_interval
    
    def get_static_map_polygons(self) -> List[List]:
        return self._simulator._static_map.get_polygons()

    def is_end(self) -> bool:
        return self._end

    ###### Baseline methods ######

    def time_based_replan(self) -> int:
        """
        Timer baseline call to update global planner in every control interval time
        This means that replanning at regular intervals
        """
        if (
            self._previous_global_path.shape[0] == 0
            or self._previous_global_path is None
        ):
            return int(Action.update_reference_path)

        if self._is_in_count(self._counter_for_timer_triggered_baseline):
            # Not update
            return int(Action.not_update_reference_path)
        else:
            #  update reference path
            self._counter_for_timer_triggered_baseline = self._reset_counter(
                self._params.time_triggered_replan_threshold
            )
            return int(Action.update_reference_path)

    def time_based_replan_patience(
        self, th_to_goal_dist: float = 3.0, patience_time: int = 5
    ) -> int:
        """
        Timer baseline with cancel call to update global planner in every control interval time
        If robot is close to the goal, cancel the timer and stop updating global planner until stucking patience_time
        """
        # Check if robot is close to the goal
        if self._dist_to_goal < th_to_goal_dist:
            if self._stucking_time < patience_time:
                return int(Action.not_update_reference_path)
            else:
                return int(Action.update_reference_path)

        # Time-based
        if self._is_in_count(self._counter_for_timer_triggered_baseline):
            # Not update
            return int(Action.not_update_reference_path)
        else:
            #  update reference path
            self._counter_for_timer_triggered_baseline = self._reset_counter(
                self._params.time_triggered_replan_threshold
            )
            return int(Action.update_reference_path)

    def no_replan(self) -> int:
        """
        No replan baseline
        """
        if (
            self._previous_global_path.shape[0] == 0
            or self._previous_global_path is None
        ):
            return int(Action.update_reference_path)
        return int(Action.not_update_reference_path)

    def distance_based_replan(self) -> int:
        """
        Distance-based baseline call to update global planner when robot move a certain distance
        th_dist: threshold distance to update global planner [m]
        """
        if (
            self._previous_global_path.shape[0] == 0
            or self._previous_global_path is None
        ):
            return int(Action.update_reference_path)

        delta_dist = (
            self._traveled_path_length - self._prev_dist_for_distance_based_baseline
        )
        if delta_dist >= self._params.distance_replan_threshold:
            self._prev_dist_for_distance_based_baseline = self._traveled_path_length
            return int(Action.update_reference_path)
        elif delta_dist < 0:
            raise ValueError("Traveled path length is smaller than previous one")
        else:
            return int(Action.not_update_reference_path)

    def stuck_based_replan(self) -> int:
        """
        Stuck-based baseline call to update global planner when robot is stuck for a certain time
        th_time: threshold time to decide whether robot is stuck
        """
        if (
            self._previous_global_path.shape[0] == 0
            or self._previous_global_path is None
        ):
            return int(Action.update_reference_path)

        if self._stucking_time >= self._params.stuck_replan_time_threshold:
            return int(Action.update_reference_path)
        else:
            return int(Action.not_update_reference_path)

    def random_based_replan(self) -> int:
        """
        Random baseline call to update global planner in every control interval time
        This means that replanning at regular intervals
        """
        if (
            self._previous_global_path.shape[0] == 0
            or self._previous_global_path is None
        ):
            return int(Action.update_reference_path)

        return int(
            self._rng.choice(
                [Action.update_reference_path, Action.not_update_reference_path]
            )
        )

    def manual_based_replan(self) -> int:
        """
        Manual operation
        """
        print("Please enter action:")
        # print("Global Planner locked or not: ", self._is_locked_global_planner(self._counter_for_global_planner_lock))
        print("Enter: Not replan, Space key: Replan")
        a = input()

        if a == "":
            return int(Action.not_update_reference_path)
        elif a == " ":
            return int(Action.update_reference_path)
        else:
            print("Invalid action")
            return int(Action.not_update_reference_path)

    ##################################################

    def _is_in_count(self, count: int) -> bool:
        return count != 0

    def _count_down(self, counter) -> int:
        if counter > 0:
            return counter - 1
        else:
            return 0

    def _reset_counter(self, max_count: int) -> int:
        return max_count

    def _detect_stuck(
        self, current_pos: np.ndarray, current_stucking_time: float
    ) -> float:
        """
        Detect stuck
        If Robot has not moved during STUCK_CHECK_INTERVAL within STUCK_RADIUS, the stuck is considered to be continuous
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

        def reset_wo_static_map():
            self._params.reset_robot_initial_state()
            self._params.reset_moving_obstacles()
            self._params.reset_goal_state()

        is_valid = False
        while not is_valid:
            # set params
            self._static_map = self._simulator.reset(self._params)

            # check initial and goal pose collision
            margin = 0.3
            ### initial pose collision check with map and lidar scans
            is_collision_start_pose = self._simulator.get_merged_map().check_collision(
                self._params.robot_initial_state.pos, self._params.robot_radius + margin
            )
            if is_collision_start_pose:
                is_valid = False
                reset_wo_static_map()
                # Retry
                continue

            ### goal pose collision check with map and current lidar scans
            is_collision_goal = self._simulator.get_merged_map().check_collision(
                self._params.goal_state.pos, self._params.robot_radius + margin
            )
            if is_collision_goal:
                is_valid = False
                reset_wo_static_map()
                # Retry
                continue

            # Run global planner
            robot_obs = self._simulator.get_observation()
            self._global_planner.set_costmap(robot_obs.static_map_with_scan)
            reference_indices, reference_path = self._global_planner.make_plan(
                start_pos=robot_obs.state.pos, goal_pos=self._params.goal_state.pos
            )

            # check feasibility
            if (
                reference_path is None
                or len(reference_indices) == 1
                or len(reference_indices) == 0
            ):
                is_valid = False
                reset_wo_static_map()
                # Retry
                continue

            # Estimate goal score
            ## calculate optimal goal time and path length with optimal path by Dijkstra
            if self._params.global_planner_type != "Dijkstra":
                dijkstra_planner = DijkstraPlanner(self._params)
                dijkstra_planner.set_costmap(robot_obs.static_map_with_scan)
                optimal_path_indices, optimal_path = dijkstra_planner.make_plan(
                    start_pos=robot_obs.state.pos, goal_pos=self._params.goal_state.pos
                )
            else:
                optimal_path = reference_path
                optimal_path_indices = reference_indices

            optimal_goal_time, optimal_path_length = self._estimate_optimal_scores(
                optimal_path[0]
            )

            # check enough length of start to goal
            map_size = self._static_map.get_map_size()
            if optimal_path_length < map_size[0] * self._params.min_path_length_ratio:
                is_valid = False
                reset_wo_static_map()
                # Retry
                continue
            is_valid = True

        return optimal_goal_time, optimal_path_length, optimal_path_indices, optimal_path

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
                 0.25: not reach goal and not collision
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

    def _observe(
        self,
        robot_obs: RobotObservation,
        global_path: np.ndarray,
        robot_traj: np.ndarray,
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

        ## Global planner is locked or not ##
        # Dim: 1 (0 ~ self.MAX_HZ_FOR_GLOBAL_PLANNER) if 0, global planner is not locked
        if self._rl_params.dim_global_planner_status > 0:
            observed_vec_list.append(float(self._counter_for_global_planner_lock))

        ## vector to Goal ##
        vec_to_goal = self._params.goal_state.pos - robot_obs.state.pos
        rotated_vec_to_goal = self._rotate_pos(vec_to_goal, -robot_obs.state.yaw)
        observed_vec_list += list(rotated_vec_to_goal)

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

        # previous path
        previous_path = self._slice(
            robot_traj, self._rl_params.range_previous_path, from_back=True
        )
        relative_previous_path = previous_path - robot_obs.state.pos
        compressed_previous_path = self._reshape_arr(
            relative_previous_path,
            (self._rl_params.dim_previous_path, 2),
            from_back=True,
        )
        compressed_previous_path = self._rotate_points(
            compressed_previous_path, -robot_obs.state.yaw
        )
        observed_vec_list += list(compressed_previous_path.reshape(-1))

        # reference paths
        reference_paths = []
        for path in self._reference_path_queue:
            nearest_index = self._find_nearest_index(path[0], robot_obs.state.pos)
            clipped_path = self._slice(
                path[0][nearest_index:],
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
            reference_paths.append(compressed_clipped_path)
            observed_vec_list += list(compressed_clipped_path.reshape(-1))

        # Visualize observation
        if (
            self._visualize_mode == self.VisualizeMode.observation
            and is_set_observation_arr
        ):
            fig, axes = plt.subplots(1, 1, figsize=(5, 5))
            s = 30.0
            # compressed previous path
            axes.scatter(
                compressed_previous_path[:, 0],
                compressed_previous_path[:, 1],
                s=s,
                c="b",
                label="previous path",
            )
            axes.scatter(
                compressed_previous_path[:, 0],
                compressed_previous_path[:, 1],
                s=s,
                c="b",
            )

            for reference_path in reference_paths:
                axes.scatter(reference_path[:, 0], reference_path[:, 1], s=s, c="r")

            for i, scan in enumerate(picked_scans):
                axes.scatter(scan[:, 0], scan[:, 1], s=s, c="g")
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
            image_arr = np.reshape(
                np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1),
            )
            io_buf.close()
            self._observation_image_arr_buffer.append(image_arr)

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

    def _reward(
        self,
        action: Action,
        local_planner_output: LocalPlannerOutput,
        robot_obs: RobotObservation,
        is_done: bool,
    ) -> float:
        reward = 0.0

        # SGT: Success weighted by (normalized) Goal Time : 0.0 ~ 1.0
        if is_done:
            if local_planner_output.is_goal_reached:
                reward += self._rl_params.sgt_reward * self._SGT(
                    optimal_goal_time=self._optimal_goal_time,
                    actual_goal_time=self._step_count
                    * self._params.control_interval_time,
                    is_collision=robot_obs.is_collision,
                )

            # SPL: Success weighted by (normalized) Path Length : 0.0 ~ 1.0
            reward += self._rl_params.spl_reward * self._SPL(
                optimal_path_length=self._optimal_path_length,
                actual_path_length=self._traveled_path_length,
                is_goal=local_planner_output.is_goal_reached,
            )

        # GP update penalty
        if action == Action.update_reference_path:
            reward += self._rl_params.replan_penalty

        # collision
        if robot_obs.is_collision:
            # collision penalty
            reward += self._rl_params.collision_penalty

        # speed reward
        speed = np.linalg.norm(robot_obs.state.linear_vel)
        reward += self._rl_params.speed_reward * speed

        return reward


register(
    id="NavigationStackEnv-v0",
    entry_point="navigation_stack_py.gym_env.navigation_stack_env:NavigationStackEnv",
)
