"""Abstract class for local planners."""

from abc import ABC, abstractmethod
from typing import NamedTuple, Tuple
import numpy as np
from navigation_stack_py.utils import MapHandler
from navigation_stack_py.common import RobotCommand


class LocalPlannerOutput(NamedTuple):
    control_command: RobotCommand
    predict_path_list: list
    best_index: int
    is_goal_reached: bool
    path_obs_costs: np.ndarray
    sub_goal_index: int


class LocalPlannerBase(ABC):
    @abstractmethod
    def compute_velocity_command() -> LocalPlannerOutput:
        """Compute velocity command from local planner.
        Returns:
             RobotCommand: velocity command
        """
        pass

    @abstractmethod
    def is_goal_reached() -> bool:
        """Check if goal is reached.
        Returns:
             bool: True if goal is reached, False otherwise
        """
        pass

    @abstractmethod
    def set_costmap(self, costmap: MapHandler) -> None:
        """Set cost map for planning"""
        pass

    @abstractmethod
    def get_costmap(self) -> MapHandler:
        """Get cost map for planning"""
        pass

    @abstractmethod
    def get_max_speed(self) -> float:
        """Get maximum speed of robot"""
        pass
