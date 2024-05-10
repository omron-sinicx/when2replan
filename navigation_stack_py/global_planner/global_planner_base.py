"""Abstract class for global planners."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
from navigation_stack_py.utils import MapHandler


class GlobalPlannerBase(ABC):
    @abstractmethod
    def set_costmap(self, costmap: MapHandler) -> None:
        """Set cost map for planning"""
        pass

    @abstractmethod
    def make_plan(self, start_state: np.ndarray, goal_state: np.ndarray) -> Tuple:
        """Make plan from start state to goal state.
        Returns:
             planed_path_ij: planned_path in map index
             planed_path_xy: planed_path in global coordinate
        """
        pass
