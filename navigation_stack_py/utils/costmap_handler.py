"""2D Cost Map handler for navigation stack.
Author: Kohei Honda
Affiliation: OMRON SINIC X
"""
from __future__ import annotations
from typing import Tuple, List

import numpy as np
import os
import cv2
from contextlib import redirect_stdout
from matplotlib import pyplot as plt
from CMap2D import CMap2D, gridshow, path_from_dijkstra_field


class MapHandler:
    """Cost Map wrapper for navigation stack.
    Occupancy map: each cell has value 0 ~ 254.
    """

    OCC_MAX_VAL = 1.0
    OCC_MIN_VAL = 0
    OCC_INFLATION_VAL = 0.5

    def __init__(self) -> None:
        self._cmap2d: CMap2D = None
        self._occupancy_map: np.ndarray = None
        self._sdf_map: np.ndarray = None

    def load_map_from_file(self, map_file_path: str) -> None:
        """Load map from file.
        Args:
             map_file_path: yaml file for map. e.g. '../maps/default.yaml'
        """

        # devide folder path and file name
        folder_path, file_name = os.path.split(map_file_path)

        if map_file_path.endswith(".yaml"):
            file_name = file_name.replace(".yaml", "")

        # load map
        # redirect stdout to avoid print
        with open(os.devnull, "w") as f, redirect_stdout(f):
            self._cmap2d = CMap2D(folder_path, file_name)

        # get occupancy and sdf map
        self._set_occupancy_map_as_binary()
        self._sdf_map = self._cmap2d.as_sdf()

    def load_map_from_occupancy_map(
        self, occupancy_map: np.ndarray, origin: np.ndarray, resolution: float
    ) -> None:
        """Load map from occupancy map.
        Args:
             occupancy_map: occupancy map as np.ndarray
        """
        self._occupancy_map = self._saturate_occupancy_map(occupancy_map)
        self._cmap2d = CMap2D()
        self._cmap2d.from_array(self._occupancy_map, origin, resolution)
        self._sdf_map = self._cmap2d.as_sdf()

    def _saturate_occupancy_map(self, occupancy_map: np.ndarray) -> np.ndarray:
        # saturation
        saturate_map = np.clip(occupancy_map, self.OCC_MIN_VAL, self.OCC_MAX_VAL)
        return saturate_map

    def _scale_occupancy_map(
        self, occupancy_map: np.ndarray, scale: float
    ) -> np.ndarray:
        return occupancy_map * scale

    def _set_occupancy_map_as_binary(self) -> None:
        if self._cmap2d is None:
            raise ValueError("cmap2d is not set")
        occ = self._cmap2d.occupancy()

        # convert to binary map
        self._occupancy_map = (occ > 0.5).astype(np.int32) * self.OCC_MAX_VAL

    def get_inflated_map(self, inflation_radius: float = None) -> "MapHandler":
        if self._cmap2d is None:
            raise ValueError("cmap2d is not set")

        if inflation_radius is None:
            raise ValueError("inflation_radius is not set")

        # First, set occupancy map as binary
        self._set_occupancy_map_as_binary()

        # make map inflation
        inflated_map: np.ndarray = self._occupancy_map.copy()

        for i in range(self._occupancy_map.shape[0]):
            for j in range(self._occupancy_map.shape[1]):
                if (
                    abs(self._sdf_map[i, j]) <= inflation_radius
                    and self._occupancy_map[i, j] == 0
                ):
                    inflated_map[i, j] = self.OCC_INFLATION_VAL

        new_map = MapHandler()
        new_map.load_map_from_occupancy_map(
            inflated_map, self.get_origin(), self.get_resolution()
        )

        return new_map

    def get_inflation_layer(self) -> np.ndarray:
        layer = self._occupancy_map.copy()
        layer = (layer == self.OCC_INFLATION_VAL).astype(np.int32)

        return layer

    def set_cmap2d(self, cmap2d: CMap2D) -> None:
        """Set cmap2d.
        Args:
             cmap2d: CMap2D object
        """
        self._cmap2d = cmap2d
        self._set_occupancy_map_as_binary()
        self._sdf_map = self._cmap2d.as_sdf()

    def copy(self) -> "MapHandler":
        new_map = MapHandler()
        new_map.set_cmap2d(self._cmap2d)
        return new_map

    def merge_map(self, map: "MapHandler") -> "MapHandler":
        """Merge another map via occupancy, occupancies are added simply.
        Args: map: merged map
        Return:
               new map handler
        """
        if not isinstance(map, MapHandler):
            raise TypeError("map must be MapHandler")

        if (
            self.get_origin()[0] != map.get_origin()[0]
            or self.get_origin()[1] != map.get_origin()[1]
        ):
            raise ValueError("origin of maps must be same")

        if self.get_resolution() != map.get_resolution():
            raise ValueError("resolution must be same")

        # filtered map
        filtered_map = self._saturate_occupancy_map(map.get_map_as_np("occupancy"))

        # new map
        new_cmap2d = CMap2D()
        new_occupancy_map = self._saturate_occupancy_map(
            self._occupancy_map + filtered_map
        )
        new_cmap2d.from_array(
            new_occupancy_map, self.get_origin(), self.get_resolution()
        )

        # new map handler
        new_map_hander = MapHandler()
        new_map_hander.set_cmap2d(new_cmap2d)
        return new_map_hander

    def view_map(self, map_type: str) -> plt.imshow:
        """View map.
        Args:
             map_type: 'occupancy' or 'sdf'
        """

        # check map type
        if map_type != "occupancy" and map_type != "sdf":
            raise ValueError('map_type must be "occupancy" or "sdf"')

        plt.figure(map_type)
        plt.title(map_type)
        plt.tick_params(
            axis="both",
            which="both",
            bottom=False,
            top=False,
            labelbottom=False,
            right=False,
            left=False,
            labelleft=False,
        )

        if map_type == "occupancy":
            return gridshow(self._occupancy_map)
        else:
            return gridshow(self._sdf_map)

    def get_map_as_np(self, map_type: str) -> np.ndarray:
        """Get map as np.ndarray.
        Args:
             map_type: 'occupancy' or 'sdf'
        Returns:
             map as np.ndarray
        """

        # check map type
        if map_type != "occupancy" and map_type != "sdf":
            raise ValueError('map_type must be "occupancy" or "sdf"')

        if map_type == "occupancy":
            return np.array(self._occupancy_map)
        else:
            return np.array(self._sdf_map)

    def get_resolution(self) -> float:
        """Get resolution.
        Returns:
             resolution of map [m]
        """
        return self._cmap2d.resolution()

    def get_origin(self) -> np.ndarray:
        """Get origin.
        Returns:
             origin (x, y) in global coordinate
        """
        return self._cmap2d.origin_xy()

    def get_area_limits(self) -> np.ndarray:
        """Get area limits.
        Returns:
             area limits (x_min, x_max, y_min, y_max) in global coordinate
        """
        origin = self.get_origin()
        map_size = self.get_map_size()
        x_min = origin[0]
        x_max = origin[0] + map_size[0]
        y_min = origin[1]
        y_max = origin[1] + map_size[1]

        return np.array([x_min, x_max, y_min, y_max])

    def get_image_size(self) -> np.ndarray:
        return self._occupancy_map.shape

    def get_map_size(self) -> np.ndarray:
        resolution = self.get_resolution()
        return (
            self._occupancy_map.shape[0] * resolution,
            self._occupancy_map.shape[1] * resolution,
        )

    def meter2pixel(self, meter: float) -> int:
        return int(meter / max(self.get_resolution(), 0.00001))

    def meter2pixel_float(self, meter: float) -> float:
        return float(meter / max(self.get_resolution(), 0.00001))

    def pixel2meter(self, pixel: int) -> float:
        return float(pixel * self.get_resolution())

    def pose2index(self, pose: np.ndarray) -> np.ndarray:
        """Convert pose to index.
        Args:
             pose: pose (x, y) in global coordinate, yaw is optional
        Returns:
             integer index (i, j)
        """

        if not isinstance(pose, np.ndarray):
            raise TypeError("pose must be np.ndarray")

        if len(pose) != 2:
            raise ValueError("pose must be (x, y) as np.ndarray")

        # convert pose to index
        xy = np.array([pose])
        in_ij_coordinates = self._cmap2d.xy_to_ij(xy)

        return in_ij_coordinates[0]

    def pose2index_float(self, pose: np.ndarray) -> np.ndarray:
        """Convert pose to index.
        Args:
             pose: pose (x, y) in global coordinate, yaw is optional
        Returns:
             float index (i, j)
        """

        if not isinstance(pose, np.ndarray):
            raise TypeError("pose must be np.ndarray")

        if len(pose) != 2:
            raise ValueError("pose must be (x, y) as np.ndarray")

        # convert pose to index
        xy = np.array([pose])
        in_ij_coordinates = self._cmap2d.xy_to_floatij(xy)

        return in_ij_coordinates[0]

    def pose_array2index_array(self, pose_array: np.ndarray) -> np.ndarray:
        """Convert pose array to index array.
        Args:
             pose_array: (N, 2) pose array in global coordinate
        Returns:
             Integer index array (N, 2)
        """

        if not isinstance(pose_array, np.ndarray):
            raise TypeError("pose_array must be np.ndarray")

        if len(pose_array.shape) != 2:
            raise ValueError("pose_array must be (N, 2) as np.ndarray")

        # convert pose array to index array
        in_ij_coordinates = self._cmap2d.xy_to_ij(pose_array)

        return in_ij_coordinates

    def pose_array2index_array_float(self, pose_array: np.ndarray) -> np.ndarray:
        """Convert pose array to index array.
        Args:
             pose_array: (N, 2) pose array in global coordinate
        Returns:
             float index array (N, 2)
        """

        if not isinstance(pose_array, np.ndarray):
            raise TypeError("pose_array must be np.ndarray")

        if len(pose_array.shape) != 2:
            raise ValueError("pose_array must be (N, 2) as np.ndarray")

        # convert pose array to index array
        in_ij_coordinates = self._cmap2d.xy_to_floatij(pose_array)

        return in_ij_coordinates

    def index2pose(self, index: np.ndarray) -> np.ndarray:
        """Convert index to pose.
        Args:
             index: integer index (i, j)
        Returns:
             pose (x, y) in global coordinate
        """

        if not isinstance(index, np.ndarray):
            raise TypeError("index must be np.ndarray")

        if len(index) != 2:
            raise ValueError("index must be (i, j) as np.ndarray")

        # convert index to pose
        in_ij_coordinates = np.array([index])
        xy = self._cmap2d.ij_to_xy(in_ij_coordinates)

        return xy[0]

    def index_array2pose_array(self, index_array: np.ndarray) -> np.ndarray:
        """Convert index array to pose array.
        Args:
             index_array: (N, 2) index array
        Returns:
             pose array (N, 2)
        """

        if not isinstance(index_array, np.ndarray):
            raise TypeError("index_array must be np.ndarray")

        if len(index_array.shape) != 2:
            raise ValueError("index_array must be (N, 2) as np.ndarray")

        # convert index array to pose array
        in_ij_coordinates = np.array([index_array])
        xy = self._cmap2d.ij_to_xy(in_ij_coordinates)

        return xy

    def get_obs_dist(self, pose: np.ndarray) -> float:
        """Check collision using prepared SDF map
        Args:
            pose: (x, y) pose array in global coordinate
        Returns:
            float: Distance to nearest obstacle
        """
        if not isinstance(pose, np.ndarray):
            raise TypeError("pose must be np.ndarray")

        if len(pose) != 2:
            raise ValueError("pose must be (x, y) as np.ndarray")

        # convert pose to index
        in_ij_coordinates = self.pose2index(pose)

        # get distance to nearest obstacle
        sdf_dist = self._sdf_map[in_ij_coordinates[0], in_ij_coordinates[1]]

        return sdf_dist

    def get_obs_dist_array(self, pose_array: np.ndarray) -> np.ndarray:
        """Check collision using prepared SDF map
        Args:
            pose_array: (N, 2) pose array in global coordinate
        Returns:
            distance array (N)
        """
        if not isinstance(pose_array, np.ndarray):
            raise TypeError("pose_array must be np.ndarray")

        if len(pose_array.shape) != 2:
            raise ValueError("pose_array must be (N, 2) as np.ndarray")

        # convert pose array to index array
        in_ij_coordinates_array = self.pose_array2index_array(pose_array)

        # get distance to nearest obstacle
        sdf_dist_array = self._sdf_map[
            in_ij_coordinates_array[:, 0], in_ij_coordinates_array[:, 1]
        ]

        return sdf_dist_array

    def check_collision(self, pose: np.ndarray, radius: float) -> bool:
        """
        True if the position is in collision
        """
        dist = self.get_obs_dist(pose)
        if dist + self.get_resolution() < radius:
            return True
        else:
            return False
        
    def check_out_of_map(self, pos: np.ndarray, radius) -> bool:
        """
        True if the position is out of bound
        """
        area_limit = self.get_area_limits()
        
        is_in_x = area_limit[0] + radius < pos[0] < area_limit[1] - radius
        is_in_y = area_limit[2] + radius < pos[1] < area_limit[3] - radius
        
        return not (is_in_x and is_in_y)

    def clip(self, pos: np.ndarray) -> np.ndarray:
        
        area_limit = self.get_area_limits()
        
        x = np.clip(pos[0], area_limit[0], area_limit[1])
        y = np.clip(pos[1], area_limit[2], area_limit[3])

        return np.array([x, y])

    def construct_dijkstra_map(
        self,
        goal_pose: np.ndarray,
        mask: np.ndarray = None,
        extra_costs=None,
        inv_value=None,
        connectedness: int = 8,
    ):
        if not isinstance(goal_pose, np.ndarray):
            raise TypeError("goal_pose must be np.ndarray")

        if len(goal_pose) != 2:
            raise ValueError("goal_pose must be (x, y) as np.ndarray")

        goal_ij = self.pose2index(goal_pose)

        if mask is not None:
            mask = mask.astype(np.uint8)

        if extra_costs is not None:
            extra_costs = extra_costs.astype(np.float32)

        dijkstra_map = self._cmap2d.dijkstra(
            goal_ij, mask, extra_costs, inv_value, connectedness
        )

        return dijkstra_map

    def compute_shortest_path_dijkstra(
        self, start_pose: np.ndarray, dijkstra_map: np.ndarray, connectedness: int = 8
    ) -> Tuple:
        """Compute shortest path using Dijkstra's algorithm
        Args:
            start_pose: (x, y) pose array in global coordinate
            dijkstra_map: Dijkstra map
        Returns:
            path_xy: (N, 2) shortest path in global coordinate
            path_ij: (N, 2) shortest path in index coordinate
            cost: jump cost (N) of the path
        """

        if not isinstance(start_pose, np.ndarray):
            raise TypeError("start_pose must be np.ndarray")

        if not isinstance(dijkstra_map, np.ndarray):
            raise TypeError("dijkstra_map must be np.ndarray")

        if len(start_pose) != 2:
            raise ValueError("start_pose must be (x, y) as np.ndarray, type")

        # convert pose to index
        start_ij = self.pose2index(start_pose)

        # compute shortest path
        result = path_from_dijkstra_field(dijkstra_map, start_ij, connectedness)

        path_ij = result[0]
        cost = result[1]

        path_xy = self.index_array2pose_array(result[0])

        return path_xy, path_ij, cost

    def compute_lidar_scan_map(
        self, pos: np.ndarray, angles: np.ndarray, ranges: np.ndarray
    ) -> np.ndarray:
        """Compute lidar scan map by ray casting
        Args:
            pos: (x, y) pose array in global coordinate
            angles: (N) angle array in radian, NOTE: values converted to float.32
            ranges: (N) range array, NOTE: values converted to float.32
        Returns:
            scan: (N, 2) lidar scan in map coordinate: -1: unknown, !-1: exist ray
        """

        if not isinstance(pos, np.ndarray):
            raise TypeError("pos must be np.ndarray")

        if not isinstance(angles, np.ndarray):
            raise TypeError("angles must be np.ndarray")

        if not isinstance(ranges, np.ndarray):
            raise TypeError("ranges must be np.ndarray")

        if len(pos) != 2:
            raise ValueError("pos must be (x, y) as np.ndarray")

        if len(angles) != len(ranges):
            raise ValueError("angles and ranges must be same length")

        # convert pose to index
        pos_ij = self.pose2index(pos)

        # compute lidar scan map by ray casting
        angles_float32 = angles.astype(np.float32)
        ranges_float32 = ranges.astype(np.float32)
        scan_map = self._cmap2d.lidar_visibility_map_ij(
            pos_ij, angles_float32, ranges_float32
        )

        return scan_map

    def get_scan_points_map(
        self, pos: np.ndarray, angles: np.ndarray, ranges: np.ndarray
    ) -> "MapHandler":
        """Get detected map by ray casting
        Args:
            pos: (x, y) pose array in global coordinate
            angles: (N) angle array in radian, NOTE: values converted to float.32
            ranges: (N) range array, NOTE: values converted to float.32
        Returns:
            detected map: (N, 2) detected map in map coordinate: OCC_MIN_VAL: unknown, OCC_MAX_VAL: detected
        """

        if not isinstance(pos, np.ndarray):
            raise TypeError("pos must be np.ndarray")

        if not isinstance(angles, np.ndarray):
            raise TypeError("angles must be np.ndarray")

        if not isinstance(ranges, np.ndarray):
            raise TypeError("ranges must be np.ndarray")

        if len(pos) != 2:
            raise ValueError("pos must be (x, y) as np.ndarray")

        if len(angles) != len(ranges):
            raise ValueError("angles and ranges must be same length")

        # compute lidar scan map by ray casting
        scan_occ = self.compute_lidar_scan_map(pos, angles, ranges)

        # Regularize base cost map: 1: obstacle, 0: free
        base_occ = (self._occupancy_map > 0.5).astype(int)

        # Regularize scan map: 1: ray, 0: unknown
        scan_occ = (scan_occ >= 1.0).astype(int)

        detected_occ = self._scale_occupancy_map(base_occ * scan_occ, self.OCC_MAX_VAL)

        scan_map = MapHandler()
        scan_map.load_map_from_occupancy_map(
            detected_occ, self.get_origin(), self.get_resolution()
        )

        return scan_map

    def calc_relative_pose(
        self, source_xy: np.ndarray, source_yaw: float, target_ij: np.ndarray
    ) -> np.ndarray:
        """
        Args:
            source_xy: (x, y) pose array in global coordinate
            source_yaw: yaw angle in radian
            target_ij: (i, j) pose array in map coordinate
        Returns:
            relative_pose: (x, y, yaw) pose array in global coordinate
        """
        if not isinstance(source_xy, np.ndarray):
            raise TypeError("source must be np.ndarray")

        if not isinstance(source_yaw, float):
            raise TypeError("source_yaw must be float")

        if not isinstance(target_ij, np.ndarray):
            raise TypeError("target must be np.ndarray")

        if len(source_xy) != 2:
            raise ValueError("source must be (x, y) as np.ndarray")

        if len(target_ij) != 2:
            raise ValueError("target must be (x, y) as np.ndarray")

        # convert pose to index
        source_ij = self.pose2index_float(source_xy)

        # relative pose
        relative_pose_index = target_ij - source_ij
        dx = self.pixel2meter(relative_pose_index[0])
        dy = self.pixel2meter(relative_pose_index[1])
        dyaw = np.arctan2(dy, dx) - source_yaw

        relative_pose = np.array([dx, dy, dyaw])

        return relative_pose
    
    def get_points(self) -> np.ndarray:
        """
        Get occupancy points in global coordinate
        Return
            points: (N, 2) points in global coordinate
        """
        occ = self.get_map_as_np("occupancy")
        
        # get points with value OCC_MAX_VAL
        points_indices = np.argwhere(occ == self.OCC_MAX_VAL)
        
        # convert to global coordinate
        points = self.index_array2pose_array(points_indices)    

        return points[0]

    def calc_relative_pose_to_scan_points(
        self, source_xy: np.ndarray, source_yaw: float, FOV: np.ndarray = None
    ) -> np.ndarray:
        """
        Calculate relative distance for all pixels with value OCC_MAX_VAL
        Args:
            source_xy: (x, y) pose array in global coordinate
            source_yaw: yaw angle in radian
            FOV: [min, max] field of view in radian, center of FOV is source_yaw
        Return
            relative_poses: (N, 3) relative pose array in global coordinate
        """
        if not isinstance(source_xy, np.ndarray):
            raise TypeError("source must be np.ndarray")

        if not isinstance(source_yaw, float):
            raise TypeError("source_yaw must be float")

        if not isinstance(FOV, np.ndarray) and FOV is not None:
            raise TypeError("FOV must be np.ndarray")

        if len(source_xy) != 2:
            raise ValueError("source must be (x, y) as np.ndarray")

        if FOV is not None:
            if len(FOV) != 2:
                raise ValueError("FOV must be (min, max) as np.ndarray")

        if FOV is None:
            FOV = np.array([-np.pi, np.pi])

        occ = self.get_map_as_np("occupancy")
        
        _relative_poses: list[np.ndarray] = []
        for i in range(occ.shape[0]):
            for j in range(occ.shape[1]):
                if occ[i, j] == self.OCC_MAX_VAL:
                    # calc relative pose
                    relative_pose = self.calc_relative_pose(
                        source_xy, source_yaw, np.array([i, j])
                    )
                    # check FOV
                    if FOV[0] <= relative_pose[2] <= FOV[1]:
                        _relative_poses.append(relative_pose)

        # sort by yaw: smallest first
        _relative_poses = sorted(_relative_poses, key=lambda x: x[2])

        # list to numpy array
        relative_poses = np.array(_relative_poses)

        return relative_poses
    
    def get_polygons(self) -> List[np.ndarray]:
        """
        Get polygons indices positions from occupancy map
        return: list of polygons indices positions in xy coordinate
        """
        # Get contours
        contours, _ = cv2.findContours(
            self._occupancy_map.astype(np.uint8),
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Convert contours to polygons
        polygons = []
        for contour in contours:
            polygon = contour.reshape(-1, 2)
            polygons.append(polygon)
            
        # Convert map coordinate to global coordinate
        polygons_indices_global = []
        for polygon in polygons:
            # convert to numpy array
            polygon_indices = np.array(polygon)
            # convert to global coordinate
            polygon_xy = self.index_array2pose_array(polygon_indices)
            polygons_indices_global.append(polygon_xy[0])
        
        return polygons_indices_global
