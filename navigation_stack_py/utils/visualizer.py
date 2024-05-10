from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, List, Deque

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import io

if TYPE_CHECKING:
    from navigation_stack_py.common import RobotObservation, AgentState, MovingObstacle
    from navigation_stack_py.utils import MapHandler


def cmap_with_transparency(cmap: cm) -> cm:
    cmap_data = cmap(np.arange(cmap.N))
    cmap_data[0, 3] = 0.0
    customized_cmap = colors.ListedColormap(cmap_data)
    return customized_cmap


def convert_to_image_array(
    fig: plt.Figure, axes: plt.Axes, image_size: np.ndarray
) -> np.ndarray:
    # setting
    axes.set_xlim([0, image_size[0]])
    axes.set_ylim([image_size[1], 0])
    fig.tight_layout()
    axes.axes.get_xaxis().set_visible(False)
    axes.axes.get_yaxis().set_visible(False)
    plt.axis("off")

    # convert
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format="raw")
    plt.close(fig)
    io_buf.seek(0)
    img_arr = np.reshape(
        np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
        newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1),
    )
    io_buf.close()

    return img_arr[:, :, :3]


def render(
    static_map: MapHandler,
    obstacle_map: MapHandler,
    obstacles_history: Deque(List[MovingObstacle]),
    robot_observation: RobotObservation,
    robot_traj: np.ndarray,
    goal_state: AgentState,
    robot_radius: float,
    inflation_layer: np.ndarray = None,
    global_path: np.ndarray = None,
    local_path_list: list = None,
    local_path_best_index: int = None,
    sub_goal_index: int = None,
    visualize_local_path: bool = False,
    is_replan: bool = False,
) -> np.ndarray:
    """
    Render the robot navigation situation.
    return: image of the bird view of the robot navigation situation.
    """
    # fig, axes = plt.subplots(1, 1, figsize=(20, 20)) # high resolution
    fig, axes = plt.subplots(1, 1, figsize=(8, 8))

    # tableau color
    color_map = colors.TABLEAU_COLORS

    # image of 2d static map
    axes.imshow(
        1 - static_map.get_map_as_np("occupancy").T,
        cmap="gray",
        vmin=0,
        vmax=1,
        alpha=1.0,
    )

    # plot only trajectory of obstacles
    # Define the color stops
    stops = [(0, "tab:orange"), (1, "white")]
    # Create the colormap
    linear_cmap = colors.LinearSegmentedColormap.from_list(
        "mycmap", stops, N=len(obstacles_history)
    )
    for obstacles, i in zip(obstacles_history, range(len(obstacles_history))):
        for obstacle in obstacles:
            if obstacle.linear_vel[0] == 0.0 and obstacle.linear_vel[1] == 0.0:
                color = color_map["tab:blue"]
            else:
                # color = color_map['tab:orange']
                color = linear_cmap(i / len(obstacles_history))
            obstacle_in_ij = static_map.pose2index_float(obstacle.pos)
            axes.add_patch(
                plt.Circle(obstacle_in_ij, 2.0, color=color, fill=True, linewidth=0)
            )

    # body of obstacles
    latest_obstacles = obstacles_history[0]
    for obstacle in latest_obstacles:
        obstacle_in_ij = static_map.pose2index_float(obstacle.pos)
        if obstacle.linear_vel[0] == 0.0 and obstacle.linear_vel[1] == 0.0:
            color = color_map["tab:blue"]
        else:
            color = color_map["tab:orange"]

        # body
        radius = obstacle.size[0]
        viz_radius = static_map.meter2pixel_float(radius)
        axes.add_patch(
            plt.Circle(
                obstacle_in_ij, viz_radius, color=color, fill=False, linewidth=15
            )
        )

    # inflation
    if inflation_layer is not None:
        inflation_cmap = cmap_with_transparency(cm.get_cmap("gray"))
        axes.imshow(inflation_layer.T, cmap=inflation_cmap, vmin=0, vmax=2, alpha=0.3)

    # robot color switched when collision
    robot_color: str = color_map["tab:green"]
    if is_replan:
        robot_color = color_map["tab:red"]

    if robot_observation.is_collision:
        robot_color = "yellow"

    # image of robot position with circle
    # viz_radius = int(robot_radius / static_map.get_resolution())
    viz_radius = static_map.meter2pixel_float(robot_radius)
    robot_in_ij = static_map.pose2index_float(robot_observation.state.pos)
    axes.add_patch(
        plt.Circle(robot_in_ij, viz_radius, color=robot_color, fill=False, linewidth=15)
    )

    # image of robot heading
    robot_heading_ij = static_map.pose2index_float(
        robot_observation.state.pos
        + np.array(
            [np.cos(robot_observation.state.yaw), np.sin(robot_observation.state.yaw)]
        ).T
        * robot_radius
    )
    axes.plot(
        [robot_in_ij[0], robot_heading_ij[0]],
        [robot_in_ij[1], robot_heading_ij[1]],
        color=robot_color,
    )

    # image of robot trajectory
    linewidth = static_map.meter2pixel_float(1.0)
    traj_in_ij = static_map.pose_array2index_array_float(robot_traj)
    axes.plot(traj_in_ij[:, 0], traj_in_ij[:, 1], color="b", linewidth=linewidth)

    # image of goal position with arrow
    goal_in_ij = static_map.pose2index_float(goal_state.pos)
    goal_heading_ij = static_map.pose2index_float(
        goal_state.pos
        + np.array([np.cos(goal_state.yaw), np.sin(goal_state.yaw)]).T * robot_radius
    )
    axes.add_patch(
        plt.Arrow(
            goal_in_ij[0],
            goal_in_ij[1],
            goal_heading_ij[0] - goal_in_ij[0],
            goal_heading_ij[1] - goal_in_ij[1],
            color=color_map["tab:orange"],
            width=viz_radius,
        )
    )

    # image of global path
    if global_path is not None:
        axes.plot(
            global_path[:, 0],
            global_path[:, 1],
            "x",
            color=color_map["tab:red"],
        )

    # image of local planner predicted path
    if local_path_list is not None and local_path_best_index is not None:
        ### visualize all local paths
        if visualize_local_path:
            for local_path in local_path_list:
                local_paths_ij = static_map.pose_array2index_array_float(local_path)
                axes.plot(
                    local_paths_ij[:, 0],
                    local_paths_ij[:, 1],
                    color=color_map["tab:blue"],
                    linewidth=1.0,
                )
        ### visualize best local path
        local_path_ij = static_map.pose_array2index_array_float(
            local_path_list[local_path_best_index]
        )
        axes.plot(
            local_path_ij[:, 0],
            local_path_ij[:, 1],
            color=color_map["tab:orange"],
            linewidth=2.0,
        )

    # image of scan points from lidar
    scan_cmap = cmap_with_transparency(cm.get_cmap("Reds"))
    scan_points = robot_observation.scan_points.get_map_as_np("occupancy")
    axes.imshow(scan_points.T, cmap=scan_cmap, vmin=0, vmax=1, alpha=1.0)

    # image of sub goal
    if sub_goal_index is not None:
        sub_goal_radius = static_map.meter2pixel_float(0.2)
        axes.add_patch(
            plt.Circle(
                global_path[sub_goal_index],
                sub_goal_radius,
                color="orange",
                fill=True,
                linewidth=3,
            )
        )

    # convert to numpy array
    image_size = static_map.get_image_size()
    image_array = convert_to_image_array(fig, axes, image_size)

    return image_array


def render_separate(
    static_map: MapHandler,
    obstacles: List[AgentState],
    robot_observation: RobotObservation,
    robot_traj: np.ndarray,
    goal_state: AgentState,
    robot_radius: float,
    inflation_layer: np.ndarray = None,
    is_replan: bool = False,
    time_step: int = 0,
    max_time_step: int = 500,
    is_end: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Render the robot navigation situation with separate static map, obstacle map, and robot trajectory.
    return: image of static map, image of obstacle map, image of robot trajectory
    """
    static_fig, static_axes = plt.subplots(1, 1, figsize=(20, 20))
    robot_fig, robot_axes = plt.subplots(1, 1, figsize=(20, 20))
    traj_fig, traj_axes = plt.subplots(1, 1, figsize=(20, 20))
    obstacle_fig, obstacle_axes = plt.subplots(1, 1, figsize=(20, 20))
    obs_traj_fig, obs_traj_axes = plt.subplots(1, 1, figsize=(20, 20))

    # tableau color
    color_map = colors.TABLEAU_COLORS

    ### static map
    # image of 2d static map
    static_axes.imshow(
        1 - static_map.get_map_as_np("occupancy").T,
        cmap="gray",
        vmin=0,
        vmax=1,
        alpha=1.0,
    )

    ### obstacles
    # image of 2d obstacle map with gradient color
    for obstacle in obstacles:
        obstacle_in_ij = static_map.pose2index_float(obstacle.pos)
        if obstacle.linear_vel[0] == 0.0 and obstacle.linear_vel[1] == 0.0:
            body_color = color_map["tab:blue"]
            traj_color = color_map["tab:blue"]
        else:
            body_color = color_map["tab:orange"]
            traj_color = color_map["tab:brown"]

        # traj
        offset = 0.5
        # obstacle_color = cm.get_cmap('Blues')(min(1.0, time_step / max_time_step + offset))
        obs_traj_axes.add_patch(
            plt.Circle(obstacle_in_ij, 1.0, color=traj_color, fill=True, linewidth=0)
        )

        # body
        radius = obstacle.size[0]
        viz_radius = static_map.meter2pixel_float(radius)
        obstacle_axes.add_patch(
            plt.Circle(
                obstacle_in_ij, viz_radius, color=body_color, fill=False, linewidth=10
            )
        )

    ### robot
    # image of robot position with circle
    viz_radius = static_map.meter2pixel_float(robot_radius)
    fill = False
    robot_color = color_map["tab:green"]
    robot_in_ij = static_map.pose2index_float(robot_observation.state.pos)
    robot_axes.add_patch(
        plt.Circle(robot_in_ij, viz_radius, color=robot_color, fill=fill, linewidth=10)
    )

    # image of robot trajectory
    radius = 1.0
    fill = True
    offset = 0.5
    # color = cm.get_cmap('Greens')(min(1.0, time_step / max_time_step + offset))
    traj_color = color_map["tab:olive"]
    traj_axes.add_patch(
        plt.Circle(robot_in_ij, radius, color=traj_color, fill=fill, linewidth=0)
    )

    # if replan, draw a red circle
    if is_replan:
        traj_axes.add_patch(
            plt.Circle(
                robot_in_ij, 1.2, color=color_map["tab:red"], fill=False, linewidth=6
            )
        )

    # draw goal
    goal_in_ij = static_map.pose2index_float(goal_state.pos)
    robot_axes.add_patch(
        plt.Circle(
            goal_in_ij, 2.0, color=color_map["tab:orange"], fill=True, linewidth=3
        )
    )

    image_size = static_map.get_image_size()
    static_image_array = convert_to_image_array(static_fig, static_axes, image_size)
    robot_image_array = convert_to_image_array(robot_fig, robot_axes, image_size)
    traj_image_array = convert_to_image_array(traj_fig, traj_axes, image_size)
    obstacle_image_array = convert_to_image_array(
        obstacle_fig, obstacle_axes, image_size
    )
    obstacle_traj_image_array = convert_to_image_array(
        obs_traj_fig, obs_traj_axes, image_size
    )

    # check image size
    if (
        static_image_array.shape != obstacle_image_array.shape
        or static_image_array.shape != robot_image_array.shape
    ):
        raise ValueError("Image size is not the same!")

    return (
        static_image_array,
        obstacle_image_array,
        robot_image_array,
        traj_image_array,
        obstacle_traj_image_array,
    )
