"""
    RRT base class
    Kohei Honda, 2022
"""

from __future__ import annotations

from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt

from navigation_stack_py.global_planner.global_planner_base import GlobalPlannerBase
from navigation_stack_py.utils import MapHandler, ParameterHandler


class Tree:
    def __init__(self, dim: int = 2):
        self.vertices: np.ndarray = np.empty((0, dim))
        self.vertices_count: int = 0
        self.edges: dict[int, int] = {}  # {vertex_id: parent_id}
        self.costs: dict[int, float] = {0: 0}  # {vertex_id: cost}

    def plot(self, start: np.ndarray, goal: np.ndarray):
        plt.plot(self.vertices[:, 0], self.vertices[:, 1], "o")
        for v_id, p_id in self.edges.items():
            plt.plot(
                [self.vertices[v_id, 0], self.vertices[p_id, 0]],
                [self.vertices[v_id, 1], self.vertices[p_id, 1]],
                "-",
                color="gray",
            )
        plt.plot(start[0], start[1], "ro")
        plt.plot(goal[0], goal[1], "go")
        plt.show()


class RRTBase(GlobalPlannerBase):
    def __init__(self, params: ParameterHandler) -> None:
        super().__init__()
        self._cost_map: MapHandler = None
        self._area_limits: np.ndarray = None
        self._tree: Tree = None

        self._inflation_radius = params.robot_radius + params.inflation_radius
        self._max_iter = 10000
        self._max_dist = 0.5  # max distance to sample point [m]
        self._interpolate_interval = (
            0.1  # interval nodes to interpolate path when reconstruct [m]
        )
        self._collision_check_sample_num = (
            10  # number of samples to check collision between two vertices
        )
        self._rng = np.random.default_rng(seed=0)

        self._reset()

    def _reset(self):
        self._tree = Tree()
        self._vertices_count = 0

    def set_costmap(self, cost_map: MapHandler):
        self._cost_map = cost_map.get_inflated_map(
            inflation_radius=self._inflation_radius
        )
        self._area_limits = self._cost_map.get_area_limits()

    def get_costmap(self) -> MapHandler:
        return self._cost_map

    def _reconstruct_path(
        self, init_pos: np.ndarray, goal_pos: np.ndarray
    ) -> np.ndarray:
        """
        Reconstruct path from start to goal
        :param init_pos: initial position
        :param goal_pos: goal position
        :return: path from start to goal
        """
        path = [goal_pos]
        current_vertex_index = self._tree.vertices_count - 1
        while current_vertex_index != 0:
            next_vertex_index = self._tree.edges[current_vertex_index]
            path.append(self._tree.vertices[next_vertex_index])
            current_vertex_index = next_vertex_index
        path.append(init_pos)
        path.reverse()
        return np.array(path)

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

    def _add_vertex(self, vertex: np.ndarray):
        """
        Add vertex to corresponding tree
        :param tree_index: tree to which vertex is added
        :param vertex: vertex to be added
        """
        self._tree.vertices = np.vstack((self._tree.vertices, vertex))
        self._tree.vertices_count += 1

    def _add_edge(self, parent_index: int, child_index: int):
        """
        Add edge to corresponding tree
        :param tree_index: tree to which edge is added
        :param parent_pos: parent vertex
        :param child_index: child index of edge
        """
        self._tree.edges[child_index] = parent_index

    def _connect_tree(self, parent_index: int, child_pos: np.ndarray):
        """
        Connect two vertices in corresponding tree
        :param tree_index: tree to which vertices are connected
        :param child_index: child vertex
        """
        self._add_vertex(child_pos)
        self._add_edge(parent_index, self._tree.vertices_count - 1)

    def _get_nearest_vertex(self, pos: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Get nearest vertex to pos
        :param pos: position
        :return: nearest vertex to pos and its index
        """
        distances = np.linalg.norm(self._tree.vertices - pos, axis=1)
        nearest_vertex_index = np.argmin(distances)
        return self._tree.vertices[nearest_vertex_index], nearest_vertex_index

    def _get_new_vertex(
        self, nearest_vertex: np.ndarray, sample_pos: np.ndarray
    ) -> np.ndarray:
        """
        Get new vertex that is max_dist away from nearest_vertex
        :param nearest_vertex: nearest vertex to sample_pos
        :param sample_pos: sample position
        :return: new vertex
        """
        new_vertex = nearest_vertex + self._max_dist * (
            sample_pos - nearest_vertex
        ) / np.linalg.norm(sample_pos - nearest_vertex)
        return new_vertex

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

        # check existence of vertex in vertices
        if np.any(np.all(self._tree.vertices == new_vertex, axis=1)):
            return False

        return True

    def _check_goal_reachable(self, goal_pos: np.ndarray, new_pos: np.ndarray) -> bool:
        """
        Check if goal is reachable
        :param goal_pos: goal position
        :return: True if goal is reachable, False otherwise
        """

        if np.linalg.norm(goal_pos - new_pos) > self._max_dist:
            # too far from goal
            return False
        else:
            poses = np.linspace(new_pos, goal_pos, self._collision_check_sample_num)
            return not self._is_collision(poses, radius=self._inflation_radius)
