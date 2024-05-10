"""
    RRT-star Planner as a global planner
    Kohei Honda, 2022
"""

from __future__ import annotations

from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt

from navigation_stack_py.global_planner.global_planner_base import GlobalPlannerBase
from navigation_stack_py.utils import MapHandler, ParameterHandler
from navigation_stack_py.global_planner.rrt_base import Tree, RRTBase


class RRTStarPlanner(RRTBase):
    def __init__(self, params: ParameterHandler) -> None:
        super().__init__(params)

        self._radius_weight = 5.0
        self._path_smoothing_window_size = 5

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
                # find candidate vertices to be rewired
                candidate_indices = self._find_near_vertices(new_vertex)

                # choose parent vertex that minimizes cost from start vertex
                parent_index, cost = self._choose_parent_vertex(
                    new_vertex, candidate_indices
                )

                # connect to the tree
                self._connect_tree(parent_index, new_vertex)

                # add cost
                new_vertex_index = self._tree.vertices_count - 1
                self._tree.costs[new_vertex_index] = cost

                # rewrite edge and cost when cost get smaller via new vertex
                self._rewrite_edge(new_vertex_index, candidate_indices)

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

        # path smoothing
        smoothed_path = self._path_smoothing(
            interpolated_path, self._path_smoothing_window_size
        )

        path_in_costmap = self._cost_map.pose_array2index_array(smoothed_path)

        path_batch = np.expand_dims(smoothed_path, axis=0)

        return path_in_costmap, path_batch

    def _find_near_vertices(self, new_vertex: np.ndarray) -> List[int]:
        """
        Find near vertices
        params: new_vertex: np.ndarray
        return: List[int]: near indices of vertices, sorted by distance
        """

        num_vertices = self._tree.vertices.shape[0]

        if num_vertices == 0:
            raise ValueError("No vertices in the tree")

        if num_vertices == 1:
            return [0]

        # radius is adaptive to the number of vertices
        radius = self._radius_weight * np.sqrt(np.log(num_vertices) / num_vertices)

        # find vertices within radius from new vertex
        dists = np.linalg.norm(self._tree.vertices - new_vertex, axis=1)
        candidate_indices = np.where(dists < radius)[0]

        if len(candidate_indices) == 0:
            # add nearest vertex
            nearest_vertex, nearest_index = self._get_nearest_vertex(new_vertex)
            candidate_indices = np.array([nearest_index])

        return candidate_indices

    def _choose_parent_vertex(
        self, new_vertex: np.ndarray, candidate_indices: List[int]
    ) -> Tuple[int, float]:
        """
        choose parent vertex from candidate vertices by cost
        params: new_vertex: np.ndarray
        params: candidate_indices: List[int]
        return: int: index of parent vertex
        """

        if len(candidate_indices) == 0:
            return self._tree.vertices.shape[0] - 1

        # calculate costs(=dists) from start vertex to new vertex, via candidate vertices
        costs = np.zeros(len(candidate_indices))
        for i, candidate_index in enumerate(candidate_indices):
            if self._check_validity(self._tree.vertices[candidate_index], new_vertex):
                costs[i] = self._tree.costs[candidate_index] + np.linalg.norm(
                    self._tree.vertices[candidate_index] - new_vertex
                )
            else:
                costs[i] = float("inf")

        # minimum cost
        min_index = np.argmin(costs)
        new_vertex_index = candidate_indices[min_index]

        return new_vertex_index, costs[min_index]

    def _rewrite_edge(
        self, new_vertex_index: int, candidate_indices: List[int]
    ) -> None:
        """
        rewrite edge and cost when cost get smaller via new vertex
        params: new_vertex_index: int
        params: candidate_indices: List[int]
        """

        for candidate_index in candidate_indices:
            if self._check_validity(
                self._tree.vertices[candidate_index],
                self._tree.vertices[new_vertex_index],
            ):
                cost = self._tree.costs[new_vertex_index] + np.linalg.norm(
                    self._tree.vertices[candidate_index]
                    - self._tree.vertices[new_vertex_index]
                )
                if cost < self._tree.costs[candidate_index]:
                    self._tree.edges[candidate_index] = new_vertex_index
                    self._tree.costs[candidate_index] = cost

    def _path_smoothing(
        self, path: np.ndarray, window_size: int
    ) -> Tuple(np.ndarray, float):
        """
        Path smoothing by dynamic window average
        Args:
            path (N, 2)
            window_size (int)
        Returns:
            smoothed path (N, 2)
        """
        # smoothing initial reference path by dynamic window
        smoothed_path = []
        for i in range(len(path)):
            lower_bound = max(0, i - window_size)
            upper_bound = min(len(path), i + window_size)
            mean_pos = np.mean(path[lower_bound:upper_bound], axis=0)
            smoothed_path.append(mean_pos)
        smoothed_path = np.array(smoothed_path)

        return smoothed_path
