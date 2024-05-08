"""
    Probabilistic Roadmap (PRM) planner
    Kohei Honda, 2022
"""

from __future__ import annotations

from typing import Tuple, List
import numpy as np
from scipy.spatial import KDTree

from navigation_stack_py.global_planner.global_planner_base import GlobalPlannerBase
from navigation_stack_py.utils import MapHandler, ParameterHandler


class Node:
    """
    Node class for Dijkstra's algorithm.
    """

    def __init__(self, pos: np.ndarray, cost: float, parent_index: int) -> None:
        """
        pos: position of the node (x, y)
        cost: cost from start node
        parent_index: index of the parent node
        """
        self.pos = pos
        self.cost = cost
        self.parent_index = parent_index

    def __str__(self) -> str:
        return f"pos: {self.pos}, cost: {self.cost}, parent_index: {self.parent_index}"


class PRMPlanner(GlobalPlannerBase):
    def __init__(self, params: ParameterHandler) -> None:
        super().__init__()

        self._cost_map: MapHandler = None
        self._area_limits: np.ndarray = None
        self._inflation_radius = params.robot_radius + params.inflation_radius

        self._sample_num = 2000
        self._collision_check_sample_num = 10
        self._num_edges = 20  # number of edges of road map per node
        # self._path_smoothing_window_size = 5
        self._interpolate_interval = (
            0.1  # interval nodes to interpolate path when reconstruct [m]
        )

        self._rng = np.random.default_rng(seed=0)

    def reset(self) -> None:
        raise NotImplementedError

    def set_costmap(self, cost_map: MapHandler):
        self._cost_map = cost_map.get_inflated_map(
            inflation_radius=self._inflation_radius
        )
        self._area_limits = self._cost_map.get_area_limits()

    def get_costmap(self) -> MapHandler:
        return self._cost_map

    def make_plan(self, start_pos: np.ndarray, goal_pos: np.ndarray) -> Tuple:
        """
        Make a plan from start to goal by PRM.
        """

        # get samples in free space
        samples = self._sample_free_space(num_samples=self._sample_num)

        # add start and goal to samples
        samples.insert(0, start_pos)
        samples.append(goal_pos)

        # build a roadmap
        road_map = self._generate_road_map(samples)

        # dijkstra's algorithm
        path = self._dijkstra(
            road_map=road_map,
            samples=samples,
            start_index=0,
            goal_index=len(samples) - 1,
        )

        if path == [] or path is None:
            # return empty path len(path) == 1
            return [], []

        path = self._downsample_path(path, self._interpolate_interval)
        path = self._interpolate_path(path)
        # path smoothing
        # smoothed_path = self._path_smoothing(
        #     downsampled_path, self._path_smoothing_window_size
        # )

        path_in_costmap = self._cost_map.pose_array2index_array(path)
        path_batch = np.expand_dims(path, axis=0)

        return path_in_costmap, path_batch

    def _sample_free_space(self, num_samples: int) -> List[np.ndarray]:
        """
        Sample nodes on free space.
        """
        samples = []
        min_x = self._area_limits[0]
        max_x = self._area_limits[1]
        min_y = self._area_limits[2]
        max_y = self._area_limits[3]

        while len(samples) < num_samples:
            sample = self._rng.uniform([min_x, min_y], [max_x, max_y])
            is_collision = self._cost_map.check_collision(
                sample, self._inflation_radius
            )
            if not is_collision:
                samples.append(sample)

        return samples

    def _is_collision(self, poses: np.ndarray, radius) -> bool:
        obs_dists = self._cost_map.get_obs_dist_array(poses)
        return np.any(obs_dists < radius)

    def _check_validity(
        self, nearest_vertex: np.ndarray, new_vertex: np.ndarray
    ) -> bool:
        """
        Check if vertex is valid
        :param vertex: vertex to be checked
        :return: True if vertex is valid, False otherwise
        """
        # check collision between nearest_vertex and new_vertex
        poses = np.linspace(
            nearest_vertex, new_vertex, self._collision_check_sample_num
        )  # generate poses between nearest_vertex and new_vertex
        if self._is_collision(poses, radius=self._inflation_radius):
            return False

        return True

    def _generate_road_map(
        self, samples: List[np.ndarray]
    ) -> Tuple[List[List[int]], List[List[float]]]:
        """
        Generate a graph from samples.
        """
        # build kdtree
        kdtree = KDTree(samples)

        # get nearest neighbors
        road_map = []
        # distances = []
        for i, sample in enumerate(samples):
            edges = []
            # dists = []
            dists, neighbors = kdtree.query(sample, k=len(samples))
            for index, j in enumerate(neighbors):
                if i == j:
                    # self remove
                    continue

                # check collision between each nodes
                if self._check_validity(sample, samples[j]):
                    edges.append(j)

                if len(edges) >= self._num_edges:
                    break
            road_map.append(edges)

        return road_map

    def _interpolate_path(self, path: np.ndarray) -> np.ndarray:
        """
        Interpolate path with self._interpolate_interval
        :param path: path to be interpolated size: (N, 2)
        :return: interpolated path
        """
        interpolated_path = [path[0]]
        for i in range(len(path) - 1):
            current_pos = path[i]
            next_pos = path[i + 1]
            distance = np.linalg.norm(next_pos - current_pos)
            num_interpolate = int(distance / self._interpolate_interval)
            for j in range(num_interpolate):
                interpolated_path.append(
                    current_pos + (j + 1) * (next_pos - current_pos) / num_interpolate
                )
        return np.array(interpolated_path)

    def _downsample_path(self, path: np.ndarray, interval: float) -> np.ndarray:
        """
        Downsample path with interval
        """
        down_sampled_path = [path[0]]
        for i in range(len(path) - 1):
            current_pos = path[i]
            next_pos = path[i + 1]
            distance = np.linalg.norm(next_pos - current_pos)
            if distance > interval:
                down_sampled_path.append(next_pos)
        return np.array(down_sampled_path)

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

    def _dijkstra(
        self,
        road_map: List[List[int]],
        samples: List[np.ndarray],
        start_index: int,
        goal_index: int,
    ) -> Tuple[List[int], float]:
        """
        Dijkstra's algorithm.
        """
        # initialize
        start_node = Node(pos=samples[start_index], cost=0.0, parent_index=-1)
        goal_node = Node(pos=samples[goal_index], cost=0.0, parent_index=-1)

        open_set, closed_set = dict(), dict()
        open_set[start_index] = start_node

        path_found = True
        while True:
            if not open_set:
                # cannot find path
                path_found = False
                break

            # get the node with the lowest cost
            current_index = min(open_set, key=lambda o: open_set[o].cost)
            current_node = open_set[current_index]

            if current_index == goal_index:
                # path is found
                goal_node.parent_index = current_node.parent_index
                goal_node.cost = current_node.cost
                break

            # remove the item from the open set
            del open_set[current_index]
            # add it to the closed set
            closed_set[current_index] = current_node

            # expand search grid based on motion model
            for neighbor in road_map[current_index]:
                dist = np.linalg.norm(current_node.pos - samples[neighbor])
                node = Node(
                    pos=samples[neighbor],
                    cost=current_node.cost + dist,
                    parent_index=current_index,
                )

                if neighbor in closed_set:
                    continue
                if neighbor in open_set:
                    if open_set[neighbor].cost > node.cost:
                        open_set[neighbor].cost = node.cost
                        open_set[neighbor].parent_index = current_index
                else:
                    open_set[neighbor] = node

        if not path_found:
            # Cannot found path
            return None

        # build a path
        path = [samples[goal_index]]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            path.append(samples[parent_index])
            parent_index = closed_set[parent_index].parent_index

        # plot path and samples
        # import matplotlib.pyplot as plt
        # for i, sample in enumerate(samples):
        #     plt.plot(sample[0], sample[1], "o")
        #     for j in road_map[i]:
        #         plt.plot(
        #             [sample[0], samples[j][0]],
        #             [sample[1], samples[j][1]],
        #             "o",
        #             color="black",
        #         )
        #         for j in road_map[i]:
        #             plt.plot(
        #                 [sample[0], samples[j][0]],
        #                 [sample[1], samples[j][1]],
        #                 "-",
        #                 color="gray",
        #             )
        # for i in range(1, len(path)):
        #     plt.plot([path[i - 1][0], path[i][0]], [path[i - 1][1], path[i][1]], "r-")
        # plt.show()
        # exit()

        return np.array(path[::-1])
