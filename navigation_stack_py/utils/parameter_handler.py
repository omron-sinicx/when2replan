"""
Manage parameter and randomize them if needed
"""
from __future__ import annotations

from typing import Union
import yaml
import os
import numpy as np

from navigation_stack_py.common import AgentState, MovingObstacle, MotionModel, Shape


class ParameterHandler:
    def __init__(self, seed: int):
        current_path: str = os.path.dirname(os.path.abspath(__file__))
        self._project_path: str = os.path.dirname(os.path.dirname(current_path))
        # set random generator seed
        self._rng = np.random.default_rng(seed)

    def _load_navigation_config(self, navigation_config: dict):
        self._control_interval_time = navigation_config["common"][
            "control_interval_time"
        ]
        self._robot_radius = navigation_config["common"]["robot_radius"]
        self._inflation_radius = navigation_config["common"]["inflation_radius"]
        self._max_episode_steps = navigation_config["common"]["max_episode_steps"]
        self._global_planner_calculation_interval = navigation_config["common"][
            "global_planner_calculation_interval"
        ]
        self._local_planner_type = navigation_config["common"]["local_planner"]
        self._global_planner_type = navigation_config["common"]["global_planner"]

        # for baseline
        self._time_triggered_replan_threshold = navigation_config["baseline_params"][
            "time_triggered_replan_threshold"
        ]
        if (
            self.time_triggered_replan_threshold
            < self._global_planner_calculation_interval
        ):
            raise Exception(
                "Timer triggered replan interval should be larger than global planner calculation interval."
            )
        self._stuck_replan_time_threshold = navigation_config["baseline_params"][
            "stuck_replan_time_threshold"
        ]
        self._distance_replan_threshold = navigation_config["baseline_params"][
            "distance_replan_threshold"
        ]
        self._patience_dist_threshold = navigation_config["baseline_params"][
            "patience_dist_threshold"
        ]
        self._patience_time_threshold = navigation_config["baseline_params"][
            "patience_time_threshold"
        ]

        # handmade detection params
        ## stuck detection
        self.stuck_check_interval = navigation_config["handmade_detection"][
            "stuck_check_interval"
        ]
        self.stuck_radius = navigation_config["handmade_detection"]["stuck_radius"]

        ## oscillation
        self.oscillation_check_interval = navigation_config["handmade_detection"][
            "oscillation_check_interval"
        ]
        self.oscillation_detect_time = navigation_config["handmade_detection"][
            "oscillation_detect_time"
        ]
        self.oscillation_goal_dist_threshold = navigation_config["handmade_detection"][
            "oscillation_goal_dist_threshold"
        ]
        self.flipping_angle_threshold = navigation_config["handmade_detection"][
            "flipping_angle_threshold"
        ]
        self.flipping_count_num = navigation_config["handmade_detection"][
            "flipping_count_num"
        ]
        self.flipping_num_threshold = navigation_config["handmade_detection"][
            "flipping_num_threshold"
        ]

        # Planner params
        # DWA params (if there is)
        if "dwa_config" in navigation_config:
            self.dwa_config = navigation_config["dwa_config"]
        else:
            self.dwa_config = None

        # Dijkstra params (if there is)
        if "dijkstra_config" in navigation_config:
            self.dijkstra_config = navigation_config["dijkstra_config"]
        else:
            self.dijkstra_config = None

    def _load_scenario(self, scenario: dict):
        self._min_path_length_ratio = scenario["min_path_length_ratio"]
        self._known_static_map = self._load_known_static_map(scenario)
        self._unknown_static_obs = self._load_unknown_static_obs(scenario)
        self._robot_initial_state = self._load_robot_initial_state(scenario)
        self._goal_state = self._load_goal_state(scenario)
        self._moving_obstacles = self._load_moving_obstacles(scenario)

    def init(self, navigation_config: Union[str, dict], scenario: dict) -> None:
        if navigation_config is None:
            raise Exception("Navigation config not found.")
        if scenario is None:
            raise Exception("Scenario not found.")
        # set params
        self._load_navigation_config(navigation_config)
        self._load_scenario(scenario)
        # scenario is saved to reset
        self._scenario = scenario

    @property
    def global_planner_calculation_interval(self) -> float:
        return self._global_planner_calculation_interval

    @property
    def local_planner_type(self) -> str:
        return self._local_planner_type

    @property
    def global_planner_type(self) -> str:
        return self._global_planner_type

    @property
    def time_triggered_replan_threshold(self) -> float:
        return self._time_triggered_replan_threshold

    @property
    def stuck_replan_time_threshold(self) -> int:
        return self._stuck_replan_time_threshold

    @property
    def distance_replan_threshold(self) -> float:
        return self._distance_replan_threshold

    @property
    def patience_dist_threshold(self) -> float:
        return self._patience_dist_threshold

    @property
    def patience_time_threshold(self) -> int:
        return self._patience_time_threshold

    @property
    def max_episode_steps(self) -> int:
        return self._max_episode_steps

    @property
    def min_path_length_ratio(self) -> float:
        return self._min_path_length_ratio

    @property
    def control_interval_time(self) -> float:
        return self._control_interval_time

    @property
    def robot_radius(self) -> float:
        return self._robot_radius

    @property
    def inflation_radius(self) -> float:
        return self._inflation_radius

    @property
    def robot_initial_state(self) -> AgentState:
        return self._robot_initial_state

    @property
    def goal_state(self) -> AgentState:
        return self._goal_state

    @property
    def known_static_map(self) -> str:
        return self._known_static_map

    @property
    def unknown_static_obs(self) -> Union[None, str]:
        return self._unknown_static_obs

    @property
    def moving_obstacles(self) -> list:
        return self._moving_obstacles

    def _load_known_static_map(self, scenario: dict) -> str:
        if "random_map_mode" in scenario:
            # random load in known static map dir
            random_map_dir = self._get_path(scenario["known_static_map"])
            if not os.path.isdir(random_map_dir):
                raise Exception(
                    "Known static map should be a directory in random map mode."
                )
            dir_list = os.listdir(random_map_dir)
            if len(dir_list) == 0:
                raise Exception("No known static map found in random map mode.")
            # randomly select a map
            map_dir = self._rng.choice(dir_list)
            map_path = os.path.join(random_map_dir, map_dir, "map.yaml")
            return map_path
        else:
            # load given known static map
            static_map = self._get_path(scenario["known_static_map"])
            if static_map is None:
                raise Exception("Known static map not found.")
            return static_map

    def reset_static_map(self) -> None:
        self._known_static_map = self._load_known_static_map(self._scenario)

    def _load_unknown_static_obs(self, scenario: dict) -> Union[None, str]:
        if "unknown_static_obs" in scenario:
            return self._get_path(scenario["unknown_static_obs"])
        else:
            return None

    def reset_robot_initial_state(self) -> None:
        self._robot_initial_state = self._load_robot_initial_state(self._scenario)

    def _load_robot_initial_state(self, scenario: dict) -> AgentState:
        if scenario["start_pose_type"] == "specified":
            # determine start pose from specified position
            start_pose: dict = scenario["start_pose"]
        elif scenario["start_pose_type"] == "candidates":
            # randomly determine start pose from candidates
            start_pose: dict = self._get_random_from_candidates(
                scenario["start_pose_candidates"]
            )
        elif scenario["start_pose_type"] == "random":
            # randomly determine start pose from constraints
            start_pose: dict = self._get_random_initial_pose(
                scenario["start_pose_constraints"]
            )
        else:
            raise Exception("Unknown start pose type.")

        return AgentState(
            np.array([start_pose["x"], start_pose["y"]]), start_pose["yaw"]
        )

    def reset_goal_state(self) -> None:
        self._goal_state = self._load_goal_state(self._scenario)

    def _load_goal_state(self, scenario: dict) -> AgentState:
        if scenario["goal_pose_type"] == "specified":
            # determine goal pose from specified position
            goal_pose: dict = scenario["goal_pose"]
        elif scenario["goal_pose_type"] == "candidates":
            # randomly determine goal pose from candidates
            goal_pose: dict = self._get_random_from_candidates(
                scenario["goal_pose_candidates"]
            )
        elif scenario["goal_pose_type"] == "random":
            # randomly determine goal pose from constraints
            goal_pose: dict = self._get_random_initial_pose(
                scenario["goal_pose_constraints"]
            )
        else:
            raise Exception("Unknown goal pose type.")

        return AgentState(np.array([goal_pose["x"], goal_pose["y"]]), goal_pose["yaw"])

    def reset_moving_obstacles(self) -> None:
        self._moving_obstacles = self._load_moving_obstacles(self._scenario)

    def _load_moving_obstacles(self, scenario: dict) -> Union[list, None]:
        if "moving_obstacles" not in scenario:
            return None

        obs_info: dict = scenario["moving_obstacles"]
        obs_ids = []
        obs_list = []
        if obs_info["type"] == "all":
            # Spawn all obstacles.
            obs_candidates = obs_info["candidates"]
            for obs in obs_candidates:
                obs_list.append(self._obs_to_instance(obs))
        elif obs_info["type"] == "candidates":
            # Spawn obstacles randomly from candidates with overlap.
            obs_candidates = obs_info["candidates"]
            obs_num = obs_info["num"]
            for _ in range(obs_num):
                obs = self._get_random_from_candidates(obs_candidates)
                obs_list.append(self._obs_to_instance(obs))
        elif obs_info["type"] == "candidates_without_overlap":
            # Spawn obstacles randomly from candidates without overlap.
            obs_candidates = obs_info["candidates"]
            obs_num = obs_info["num"]
            while len(obs_list) < obs_num:
                obs = self._get_random_from_candidates(obs_candidates)
                # check overlap in the obs_list
                if obs["id"] in obs_ids:
                    continue
                else:
                    obs_ids.append(obs["id"])
                    obs_list.append(self._obs_to_instance(obs))

        elif obs_info["type"] == "candidates_pickup":
            # pick up 'must' obstacle and randomly spawn other obstacles from candidates.
            ## separate must and maybe candidates
            must_candidates = []
            maybe_candidates = []
            for obs in obs_info["candidates"]:
                if obs["type"] == "must":
                    must_candidates.append(obs)
                elif obs["type"] == "maybe":
                    maybe_candidates.append(obs)
                else:
                    raise Exception("Unknown obstacle type.")

            if len(must_candidates) > obs_info["num"]:
                raise Exception("Too many must obstacles.")

            ## add must obstacles
            obs_list = [self._obs_to_instance(obs) for obs in must_candidates]

            ## add maybe obstacles
            maybe_num = obs_info["num"] - len(must_candidates)
            for _ in range(maybe_num):
                obs = self._get_random_from_candidates(maybe_candidates)
                obs_list.append(self._obs_to_instance(obs))

        elif obs_info["type"] == "random":
            # Spawn obstacles randomly with some constraints.
            obs_constraints = obs_info["constraints"]
            obs_candidates = obs_info["candidates"]
            obs_num = obs_info["num"]
            default_obs = self._get_random_from_candidates(obs_candidates)
            for _ in range(obs_num):
                obs = self._get_random_obs_with_constraints(
                    obs_constraints, default_obs
                )
                obs_list.append(self._obs_to_instance(obs))
        else:
            raise Exception("Unknown obstacle selection type.")

        return obs_list

    def _get_path(self, path: str):
        return os.path.join(self._project_path, path)

    def _get_random_from_candidates(self, candidates: list) -> Union[dict, None]:
        if len(candidates) == 0:
            return None
        elif len(candidates) == 1:
            return candidates[0]
        else:
            return candidates[self._rng.integers(0, len(candidates))]

    def _obs_to_instance(self, obs: dict) -> MovingObstacle:
        x = obs["x"]
        y = obs["y"]
        yaw = obs["yaw"]
        vx = obs["vx"]
        vy = obs["vy"]
        vyaw = obs["vyaw"]
        shape = obs["shape"]
        size = obs["size"]
        motion_model_list = obs["motion_model"]
        motion_model = self._rng.choice(motion_model_list)

        if motion_model == "point_mass_model":
            # zero velocity to avoid collision
            vx = 0.0
            vy = 0.0
            vyaw = 0.0

        ins = MovingObstacle(
            pos=np.array([x, y]),
            yaw=yaw,
            linear_vel=np.array([vx, vy]),
            target_vel=np.array([vx, vy]),
            angular_vel=vyaw,
            size=size,
            shape=Shape.from_str(shape),
            motion_model=MotionModel.from_str(motion_model),
        )
        return ins

    def _get_random_obs_with_constraints(
        self, obs_constraints: dict, default_obs: dict
    ) -> dict:
        if default_obs is None:
            raise Exception("Default obstacle is not specified.")

        # Get constraints
        size_min = obs_constraints["size"]["min"]
        size_max = obs_constraints["size"]["max"]
        x_min = obs_constraints["position"]["x"]["min"]
        x_max = obs_constraints["position"]["x"]["max"]
        y_min = obs_constraints["position"]["y"]["min"]
        y_max = obs_constraints["position"]["y"]["max"]
        yaw_min = obs_constraints["position"]["yaw"]["min"]
        yaw_max = obs_constraints["position"]["yaw"]["max"]
        vx_min = obs_constraints["velocity"]["vx"]["min"]
        vx_max = obs_constraints["velocity"]["vx"]["max"]
        vy_min = obs_constraints["velocity"]["vy"]["min"]
        vy_max = obs_constraints["velocity"]["vy"]["max"]
        vyaw_min = obs_constraints["velocity"]["vyaw"]["min"]
        vyaw_max = obs_constraints["velocity"]["vyaw"]["max"]

        # Generate random values
        size = self._rng.uniform(size_min, size_max)
        x = self._rng.uniform(x_min, x_max)
        y = self._rng.uniform(y_min, y_max)
        yaw = self._rng.uniform(yaw_min, yaw_max)
        vx = self._rng.uniform(vx_min, vx_max)
        vy = self._rng.uniform(vy_min, vy_max)
        vyaw = self._rng.uniform(vyaw_min, vyaw_max)

        random_obs = default_obs
        random_obs["size"] = size
        random_obs["x"] = x
        random_obs["y"] = y
        random_obs["yaw"] = yaw
        random_obs["vx"] = vx
        random_obs["vy"] = vy
        random_obs["vyaw"] = vyaw

        return random_obs

    def _get_random_initial_pose(self, constraint: dict) -> dict:
        if constraint is None:
            raise Exception("Constraint is not specified.")

        x_min = constraint["x"]["min"]
        x_max = constraint["x"]["max"]
        y_min = constraint["y"]["min"]
        y_max = constraint["y"]["max"]
        yaw_min = constraint["yaw"]["min"]
        yaw_max = constraint["yaw"]["max"]

        x = self._rng.uniform(x_min, x_max)
        y = self._rng.uniform(y_min, y_max)
        yaw = self._rng.uniform(yaw_min, yaw_max)

        start_pose = {"x": x, "y": y, "yaw": yaw}

        return start_pose
