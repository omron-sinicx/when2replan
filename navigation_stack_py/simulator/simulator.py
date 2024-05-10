from __future__ import annotations

from typing import Tuple, List
import numpy as np
from navigation_stack_py.common import (
    AgentState,
    RobotCommand,
    RobotObservation,
    MovingObstacle,
    robot_observation,
)
from navigation_stack_py.utils import MapHandler, ParameterHandler


class Simulator:
    """Navigation Simulator"""

    # common settings
    CONTROL_INTERVAL: float  # [s]
    ROBOT_RADIUS: float  # [m]

    # simulator settings
    # Ray casting parameters
    NUM_SCAN_RAY: int = 200
    SCAN_RANGE: float = 10.0  # [m]
    SCAN_ANGLE_MIN: float = -np.pi
    SCAN_ANGLE_MAX: float = np.pi

    def __init__(self) -> None:
        self._robot_state: AgentState = None
        self._robot_traj: np.ndarray = np.array([[]])
        self._robot_observation: RobotObservation = None
        self._moving_obstacles: List = []

        self._static_map: MapHandler  # known static map updated in reset()
        self._static_obstacle_map: MapHandler  # unknown static obstacle map updated every step()
        self._merged_map: MapHandler  # merged map used for ray casting (= static_map + obstacle_map)

        # lidar settings
        self._scan_angles = np.linspace(
            self.SCAN_ANGLE_MIN, self.SCAN_ANGLE_MAX, self.NUM_SCAN_RAY
        )
        self._scan_ranges = (
            np.ones(self.NUM_SCAN_RAY) * self.SCAN_RANGE
        )  # scan range is constant, like lidar

    def reset(self, params: ParameterHandler) -> MapHandler:
        # common params
        self.CONTROL_INTERVAL = params.control_interval_time
        self.ROBOT_RADIUS = params.robot_radius

        # load 2d known static map
        static_map = params.known_static_map
        if static_map is None:
            raise ValueError("known static map is not found")
        else:
            self._static_map = MapHandler()
            self._static_map.load_map_from_file(static_map)

        # load unknown static obstacle map
        unknown_static_obs = params.unknown_static_obs
        if unknown_static_obs is None:
            # load white map with same size and resolution as known static map
            white_map = np.zeros(self._static_map.get_image_size())
            origin = self._static_map.get_origin()
            resolution = self._static_map.get_resolution()
            self._static_obstacle_map = MapHandler()
            self._static_obstacle_map.load_map_from_occupancy_map(
                white_map, origin, resolution
            )
        else:
            self._static_obstacle_map = MapHandler()
            self._static_obstacle_map.load_map_from_file(unknown_static_obs)

        # reset initial robot state
        start_state = params.robot_initial_state
        if start_state is None:
            self._robot_state = AgentState(
                pos=np.array([0.0, 0.0]),
                yaw=0.0,
                linear_vel=np.array([0.0, 0.0]),
                angular_vel=0.0,
            )
        else:
            self._robot_state = start_state

        # reset trajectory
        self._robot_traj = np.array([self._robot_state.pos])

        # reset moving obstacles states
        self._moving_obstacles = params.moving_obstacles

        if self._moving_obstacles is None:
            print("No moving obstacles are found")
            pass

        # update obstacle map
        obstacle_map = self._generate_obstacle_map(
            self._static_obstacle_map, self._moving_obstacles
        )
        min_dist_to_obstacle = obstacle_map.get_obs_dist(self._robot_state.pos)

        # update merged map
        self._merged_map = self._static_map.merge_map(obstacle_map)

        # update observation
        self._robot_observation = self._observe(self._robot_state, min_dist_to_obstacle)

        return self._static_map

    def set_robot_state_and_obstacles_and_traj(
        self,
        robot_state: AgentState,
        moving_obstacles: List[MovingObstacle],
        robot_traj: np.ndarray,
    ) -> None:
        self._robot_state = robot_state
        self._moving_obstacles = moving_obstacles
        self._robot_traj = robot_traj

    def get_obstacles_status(self) -> List:
        return self._moving_obstacles.copy()

    def step(self, control_cmd: RobotCommand) -> Tuple:
        # update state
        self._robot_state = self._update_robot_state(self._robot_state, control_cmd)

        # update trajectory
        self._robot_traj = np.append(self._robot_traj, [self._robot_state.pos], axis=0)

        # update moving obstacles
        self._moving_obstacles = self._update_moving_obstacles(self._moving_obstacles)

        # update obstacle map
        obstacle_map = self._generate_obstacle_map(
            self._static_obstacle_map, self._moving_obstacles
        )
        min_dist_to_obstacle = obstacle_map.get_obs_dist(self._robot_state.pos)

        # update merged map
        self._merged_map = self._static_map.merge_map(obstacle_map)

        # update observation
        self._robot_observation = self._observe(self._robot_state, min_dist_to_obstacle)

        return obstacle_map, self._robot_traj

    def get_observation(self) -> RobotObservation:
        return self._robot_observation

    # NOTE: current obstacle state is used, not future state
    def get_goal_observation(self, goal_state: AgentState) -> RobotObservation:
        robot_state = goal_state
        min_dist_to_obstacle = self._merged_map.get_obs_dist(robot_state.pos)
        robot_observation = self._observe(robot_state, min_dist_to_obstacle)
        return robot_observation

    def get_static_map(self) -> MapHandler:
        return self._static_map

    def get_merged_map(self) -> MapHandler:
        return self._merged_map

    def _observe(
        self, robot_state: AgentState, min_dist_to_moving_obs: float
    ) -> RobotObservation:
        scan_points_map = self._merged_map.get_scan_points_map(
            robot_state.pos, self._scan_angles, self._scan_ranges
        )
        scan_relative_poses = scan_points_map.calc_relative_pose_to_scan_points(
            robot_state.pos, robot_state.yaw
        )
        is_collision = self._merged_map.check_collision(
            robot_state.pos, self.ROBOT_RADIUS
        )
        static_map_with_scan = self._static_map.merge_map(scan_points_map)

        return RobotObservation(
            state=robot_state,
            scan_points=scan_points_map,
            is_collision=is_collision,
            static_map_with_scan=static_map_with_scan,
            relative_poses_scan_points=scan_relative_poses,
            min_dist_to_moving_obstacle=min_dist_to_moving_obs,
        )

    def _update_robot_state(
        self, robot_state: AgentState, control_cmd: RobotCommand
    ) -> AgentState:
        rot_mat = np.array(
            [
                [np.cos(robot_state.yaw), -np.sin(robot_state.yaw)],
                [np.sin(robot_state.yaw), np.cos(robot_state.yaw)],
            ]
        )
        new_pos = (
            robot_state.pos
            + rot_mat.dot(control_cmd.linear_vel) * self.CONTROL_INTERVAL
        )
        _new_yaw = robot_state.yaw + control_cmd.angular_vel * self.CONTROL_INTERVAL

        # -pi ~ pi
        new_yaw = np.arctan2(np.sin(_new_yaw), np.cos(_new_yaw))

        new_linear_vel = control_cmd.linear_vel
        new_angular_vel = control_cmd.angular_vel

        return AgentState(new_pos, new_yaw, new_linear_vel, new_angular_vel)

    def _update_obs_state(self, obs: MovingObstacle) -> MovingObstacle:
        return obs.step(self.CONTROL_INTERVAL, self._robot_state.pos, self.ROBOT_RADIUS)

    def _update_moving_obstacles(self, obs_list: List) -> List:
        new_obs_list = []
        for obs in obs_list:
            obs = self._update_obs_state(obs)
            new_obs_list.append(obs)
        return new_obs_list

    def _generate_obstacle_map(
        self, static_obstacle_map: MapHandler, moving_obstacles: List
    ) -> MapHandler:
        new_occ = static_obstacle_map.get_map_as_np("occupancy").copy()
        for obs in moving_obstacles:
            new_occ = obs.draw_2d(
                new_occ,
                static_obstacle_map.pose2index,
                static_obstacle_map.meter2pixel,
                static_obstacle_map.OCC_MAX_VAL,
            )

        obstacle_map = MapHandler()
        obstacle_map.load_map_from_occupancy_map(
            new_occ,
            static_obstacle_map.get_origin(),
            static_obstacle_map.get_resolution(),
        )

        return obstacle_map
