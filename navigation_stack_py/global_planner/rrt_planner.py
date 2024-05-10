"""
    RRT Planner as a global planner
    Kohei Honda, 2022
"""

from __future__ import annotations

from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt

from navigation_stack_py.global_planner.global_planner_base import GlobalPlannerBase
from navigation_stack_py.utils import MapHandler, ParameterHandler
from navigation_stack_py.global_planner.rrt_base import Tree, RRTBase


class RRTPlanner(RRTBase):
    def __init__(self, params: ParameterHandler) -> None:
        super().__init__(params)

    def make_plan(self, start_pos: np.ndarray, goal_pos: np.ndarray) -> Tuple:
        # initialize tree
        self._reset()
        self._add_vertex(start_pos)

        is_goal = False
        for _ in range(self._max_iter):
            # random sample pos
            x = self._rng.uniform(self._area_limits[0], self._area_limits[1])
            y = self._rng.uniform(self._area_limits[2], self._area_limits[3])
            sample_pos = np.array([x, y])

            # get nearest vertex
            nearest_vertex, nearest_index = self._get_nearest_vertex(sample_pos)

            # get new vertex
            new_vertex = self._get_new_vertex(nearest_vertex, sample_pos)

            # check validity
            is_valid = self._check_validity(nearest_vertex, new_vertex)

            if is_valid:
                # add new vertex
                self._connect_tree(nearest_index, new_vertex)
            else:
                continue

            # check goal reachability
            if self._check_goal_reachable(goal_pos, new_vertex):
                is_goal = True
                break

        if is_goal:
            path = self._reconstruct_path(start_pos, goal_pos)
        else:
            # return best effort path
            nearest_vertex_to_goal, _ = self._get_nearest_vertex(goal_pos)
            path = self._reconstruct_path(start_pos, nearest_vertex_to_goal)

        interpolated_path = self._interpolate_path(path)

        path_in_costmap = self._cost_map.pose_array2index_array(interpolated_path)

        path_batch = np.expand_dims(interpolated_path, axis=0)

        return path_in_costmap, path_batch
