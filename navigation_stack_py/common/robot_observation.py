"""Data structure for the robot observation
"""
from __future__ import annotations

from typing import NamedTuple
import numpy as np
from navigation_stack_py.common import AgentState
from navigation_stack_py.utils import MapHandler


class RobotObservation(NamedTuple):
    state: AgentState
    scan_points: MapHandler = None  # scan points in the map
    relative_poses_scan_points: np.ndarray = (
        None  # relative poses N x (x, y, yaw) of the scan points from the robot pos
    )
    static_map_with_scan: MapHandler = None  # scan and static map
    is_collision: bool = False
    min_dist_to_moving_obstacle: float = None  # minimum distance to the moving obstacle

    def __str__(self) -> str:
        return "RobotObservation(state={})".format(self.state)

    def __repr__(self) -> str:
        return self.__str__()

    def __copy__(self):
        return RobotObservation(
            self.state,
            self.relative_poses_scan_points,
            self.scan_points,
            self.static_map_with_scan,
        )

    def __deepcopy__(self):
        return RobotObservation(
            self.state.copy(),
            self.relative_poses_scan_points.copy(),
            self.scan_points.copy(),
            self.static_map_with_scan.copy(),
        )
