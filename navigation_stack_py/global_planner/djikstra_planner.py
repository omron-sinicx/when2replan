from __future__ import annotations
from typing import Tuple
import numpy as np

from navigation_stack_py.global_planner.global_planner_base import GlobalPlannerBase
from navigation_stack_py.utils import MapHandler, ParameterHandler


class DijkstraPlanner(GlobalPlannerBase):
    def __init__(self, params: ParameterHandler) -> None:
        super().__init__()
        self._cost_map: MapHandler = None
        self.INFLATION_RADIUS = params.robot_radius + params.inflation_radius
        self.MASK: np.ndarray = None
        self.EXTRA_COSTS: np.ndarray = None
        self.INV_VALUE: float = None
        self.CONNECTED_NESS: int = 8

        if params.dijkstra_config is not None:
            self.CONNECTED_NESS = params.dijkstra_config["connectedness"]
        else:
            print("DijkstraPlanner: No dijkstra_config is given. Use default value.")

    def set_costmap(self, cost_map: MapHandler):
        self._cost_map = cost_map.get_inflated_map(
            inflation_radius=self.INFLATION_RADIUS
        )

    def make_plan(self, start_pos: np.ndarray, goal_pos: np.ndarray) -> Tuple:
        """
        Find global path by Dijkstra's algorithm
        Args:
            start_pos: start position of robot [x,y]
            goal_pos: goal position of robot [x,y]
        Returns:
            global_path_indices: indices of global path (N,2)
            global_path: global path (N,2) in continuous coordinate
        """

        if not isinstance(start_pos, np.ndarray):
            raise TypeError("start_state must be numpy.ndarray")

        if not isinstance(goal_pos, np.ndarray):
            raise TypeError("goal_state must be numpy.ndarray")

        if len(start_pos) != 2:
            raise ValueError("start_state must be 2-dimensional, (x,y)")

        if len(goal_pos) != 2:
            raise ValueError("goal_state must be 2-dimensional, (x,y)")

        dijk_map = self._cost_map.construct_dijkstra_map(
            goal_pose=goal_pos,
            mask=self.MASK,
            extra_costs=self.EXTRA_COSTS,
            inv_value=self.INV_VALUE,
            connectedness=self.CONNECTED_NESS,
        )

        shortest_path, indices, _ = self._cost_map.compute_shortest_path_dijkstra(
            start_pose=start_pos,
            dijkstra_map=dijk_map,
            connectedness=self.CONNECTED_NESS,
        )

        return indices, shortest_path

    def get_costmap(self) -> MapHandler:
        return self._cost_map
