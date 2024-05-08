from typing import List
import fire
import numpy as np
import pygame
import os
import pickle
import cv2
import glob
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
from navigation_stack_py.gym_env import NavigationStackEnv
from navigation_stack_py.common import MovingObstacle, MotionModel, AgentState


def rhu(number: float, decimals=0) -> int:
    return int(Decimal(number).quantize(Decimal("0"), rounding=ROUND_HALF_UP))


def main(traj_data_pkl: str, map_pkl: str = None, save_path: str = None) -> None:
    # simulation settings
    robot_radius = 1.0  # meters
    line_width = 10  # robot line width
    goal_radius = 1.0  # meters
    control_time_interval = 0.1  # seconds

    # Window settings
    window_width = 1400  # pixels
    window_height = 1400  # pixels
    center = (150, 150)
    scale = 45  # pixels per meter
    render_fps = 10  # frames per second
    font_size = 100  # pixels
    font_location = (20, 20)  # pixels

    # colors
    black = [20, 20, 40]
    lightblue = [0, 120, 255]
    darkblue = [0, 40, 160]
    red = [255, 0, 0]
    # orange = [255, 100, 0]
    white = [255, 255, 255]
    # blue = [0, 0, 255]
    lightgrey = [230, 230, 230]

    # tableu colors
    green = [44, 150, 44]
    orange = [255, 127, 14]
    olive = [188, 189, 34]
    blue = [31, 119, 180]

    # get directory path of traj_data_pkl
    dir = os.path.dirname(traj_data_pkl)
    name = os.path.basename(traj_data_pkl)

    if map_pkl is None:
        map_pkl = os.path.join(dir, "map.pkl")

    if save_path is None:
        save_path = os.path.join(dir, name.replace(".pkl", ".mp4"))

    # Prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(save_path, fourcc, render_fps, (window_width, window_height))

    with open(traj_data_pkl, "rb") as f:
        # load data
        traj_data = pickle.load(f)

    with open(map_pkl, "rb") as f:
        # load map as list of polygon
        map_polygons = pickle.load(f)

    pygame.init()
    pygame.display.set_caption("Navigation")
    pygame.display.init()
    window = pygame.display.set_mode((window_width, window_height))
    font = pygame.font.Font(None, font_size)

    clock = pygame.time.Clock()
    canvas = pygame.Surface((window_width, window_height))
    canvas.fill(white)

    robot_traj = []
    replan_history = []

    length = len(traj_data)
    for i in range(length):
        canvas.fill(white)

        data = traj_data[i]

        scan = data["scan"]
        # inflation area for scans
        for point in scan:
            pygame.draw.circle(
                surface=canvas,
                color=lightgrey,
                center=(
                    rhu(point[0] * scale + center[0]),
                    rhu(point[1] * scale + center[1]),
                ),
                radius=robot_radius * scale,
                width=0,
            )
        # inflation area for static map
        for polygon in map_polygons:
            # edge
            edges = [
                (polygon[i], polygon[(i + 1) % len(polygon)])
                for i in range(len(polygon))
            ]
            interpolated_points = []
            for edge in edges:
                interpolated_points += list(
                    np.linspace(edge[0], edge[1], num=10, endpoint=False)
                )
            for point in interpolated_points:
                pygame.draw.circle(
                    surface=canvas,
                    color=lightgrey,
                    center=(
                        rhu(point[0] * scale + center[0]),
                        rhu(point[1] * scale + center[1]),
                    ),
                    radius=robot_radius * scale,
                    width=0,
                )

        # goal
        goal = data["goal"]
        pygame.draw.circle(
            surface=canvas,
            color=red,
            center=(
                rhu(goal[0] * scale + center[0]),
                rhu(goal[1] * scale + center[1]),
            ),
            radius=goal_radius / 2.0 * scale,
            width=2,
        )

        # draw map
        for polygon in map_polygons:
            pygame.draw.polygon(
                surface=canvas,
                color=black,
                points=[node * scale + center for node in polygon],
                width=0,
            )

        robot = data["robot_state"]
        robot_traj.append(robot.pos)
        if data["is_replan"]:
            replan_history.append(True)
        else:
            replan_history.append(False)

        if robot_traj != [] and len(robot_traj) > 2:
            # draw robot trajectory
            
            for i, pos in enumerate(robot_traj):
                if replan_history[i]:
                    color = red
                    size = 0.2
                else:
                    color = lightblue
                    size = 0.1
                pygame.draw.circle(
                    surface=canvas,
                    color=color,
                    center=(
                        rhu(pos[0] * scale + center[0]),
                        rhu(pos[1] * scale + center[1]),
                    ),
                    radius= size * scale,
                    width=0,
                )

        # draw robot
        # body
        pygame.draw.circle(
            surface=canvas,
            color=green,
            center=(
                rhu(robot.pos[0] * scale + center[0]),
                rhu(robot.pos[1] * scale + center[1]),
            ),
            radius=robot_radius * scale,
            width=line_width,
        )

        # heading line
        # start_pos = robot.pos
        # end_pos = robot.pos + robot_radius * np.array(
        #     [np.cos(robot.yaw), np.sin(robot.yaw)]
        # )
        # pygame.draw.line(
        #     surface=canvas,
        #     color=green,
        #     start_pos=(
        #         rhu(start_pos[0] * scale + center[0]),
        #         rhu(start_pos[1] * scale + center[1]),
        #     ),
        #     end_pos=(
        #         rhu(end_pos[0] * scale + center[0]),
        #         rhu(end_pos[1] * scale + center[1]),
        #     ),
        #     width=line_width,
        # )

        # wheels
        wheel_length = robot_radius * 0.4
        wheel_width = robot_radius * 0.2
        # left wheel
        lw_center = robot.pos + (robot_radius - wheel_width / 2.0) * np.array(
            [-np.sin(robot.yaw), np.cos(robot.yaw)]
        )
        # polygon points
        left_wheel = [
            lw_center
            + wheel_length * np.array([np.cos(robot.yaw), np.sin(robot.yaw)])
            + wheel_width * np.array([-np.sin(robot.yaw), np.cos(robot.yaw)]),
            lw_center
            + wheel_length * np.array([np.cos(robot.yaw), np.sin(robot.yaw)])
            - wheel_width * np.array([-np.sin(robot.yaw), np.cos(robot.yaw)]),
            lw_center
            - wheel_length * np.array([np.cos(robot.yaw), np.sin(robot.yaw)])
            - wheel_width * np.array([-np.sin(robot.yaw), np.cos(robot.yaw)]),
            lw_center
            - wheel_length * np.array([np.cos(robot.yaw), np.sin(robot.yaw)])
            + wheel_width * np.array([-np.sin(robot.yaw), np.cos(robot.yaw)]),
        ]

        pygame.draw.polygon(
            surface=canvas,
            color=green,
            points=[node * scale + center for node in left_wheel],
            width=0,
        )

        # right wheel
        rw_center = robot.pos + (robot_radius - wheel_width / 2.0) * np.array(
            [np.sin(robot.yaw), -np.cos(robot.yaw)]
        )
        right_wheel = [
            rw_center
            + wheel_length * np.array([np.cos(robot.yaw), np.sin(robot.yaw)])
            + wheel_width * np.array([np.sin(robot.yaw), -np.cos(robot.yaw)]),
            rw_center
            + wheel_length * np.array([np.cos(robot.yaw), np.sin(robot.yaw)])
            - wheel_width * np.array([np.sin(robot.yaw), -np.cos(robot.yaw)]),
            rw_center
            - wheel_length * np.array([np.cos(robot.yaw), np.sin(robot.yaw)])
            - wheel_width * np.array([np.sin(robot.yaw), -np.cos(robot.yaw)]),
            rw_center
            - wheel_length * np.array([np.cos(robot.yaw), np.sin(robot.yaw)])
            + wheel_width * np.array([np.sin(robot.yaw), -np.cos(robot.yaw)]),
        ]

        pygame.draw.polygon(
            surface=canvas,
            color=green,
            points=[node * scale + center for node in right_wheel],
            width=0,
        )

        # if replan, draw red circle
        if data["is_replan"]:
            pygame.draw.circle(
                surface=canvas,
                color=red,
                center=(
                    rhu(robot.pos[0] * scale + center[0]),
                    rhu(robot.pos[1] * scale + center[1]),
                ),
                radius=(robot_radius) * scale,
                width=line_width,
            )

        # draw obstacles
        for obstacle in data["obstacles"]:
            pos = obstacle.pos
            if obstacle.motion_model == MotionModel.POINT_MASS_MODEL:
                color = blue
            else:
                color = orange
            pygame.draw.circle(
                surface=canvas,
                color=color,
                center=(
                    rhu(pos[0] * scale + center[0]),
                    rhu(pos[1] * scale + center[1]),
                ),
                radius=obstacle.size[0] * scale,
                # width=line_width,
            )

        # scan points
        for point in scan:
            pygame.draw.circle(
                surface=canvas,
                color=olive,
                center=(
                    rhu(point[0] * scale + center[0]),
                    rhu(point[1] * scale + center[1]),
                ),
                radius=0.2 * scale,
                width=0,
            )

        # reference path
        reference_path = data["global_path"]
        for pos in reference_path:
            pygame.draw.circle(
                surface=canvas,
                color=red,
                center=(
                    rhu(pos[0] * scale + center[0]),
                    rhu(pos[1] * scale + center[1]),
                ),
                radius=0.05 * scale,
                width=0,
            )

        # local paths
        local_paths = data["local_path_list"]
        best_index = data["local_path_best_index"]

        # all paths
        if local_paths is not None:
            for path in local_paths:
                pygame.draw.lines(
                    surface=canvas,
                    color=lightblue,
                    points=[
                        (
                            rhu(pos[0] * scale + center[0]),
                            rhu(pos[1] * scale + center[1]),
                        )
                        for pos in path
                    ],
                    closed=False,
                    width=rhu(0.03 * scale),
                )

            # best local path
            best_path = local_paths[best_index]
            pygame.draw.lines(
                surface=canvas,
                color=orange,
                points=[
                    (rhu(pos[0] * scale + center[0]), rhu(pos[1] * scale + center[1]))
                    for pos in best_path
                ],
                closed=False,
                width=rhu(0.05 * scale),
            )

        window.blit(canvas, (0, 0))

        # time stamp
        time = i * control_time_interval
        text = font.render("Time: " + str(round(time, 1)) + " s", True, (0, 0, 0))
        window.blit(text, font_location)

        pygame.event.pump()
        pygame.display.update()

        # save image
        screen_array = pygame.surfarray.array3d(window)
        screen_array = screen_array.transpose([1, 0, 2])
        screen_array = cv2.cvtColor(screen_array, cv2.COLOR_RGB2BGR)
        out.write(screen_array)

        clock.tick(render_fps)

    out.release()
    cv2.destroyAllWindows()

    pygame.display.quit()
    pygame.quit()


def run_all(dir: str):
    # get all pkl file path in the dir recursively without map.pkl
    pkl_files = glob.glob(dir + "/**/*.pkl", recursive=True)

    # remove map.pkl
    pkl_files = [pkl_file for pkl_file in pkl_files if "map.pkl" not in pkl_file]

    for pkl_file in pkl_files:
        print("Processing: ", pkl_file)
        main(pkl_file)


if __name__ == "__main__":
    fire.Fire()
    # fire.Fire(main)
    # fire.Fire(run_all)
