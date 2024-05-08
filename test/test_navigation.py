import matplotlib.animation as animation
from matplotlib import pyplot as plt
from navigation_stack_py import global_planner
from navigation_stack_py.simulator import Simulator
from navigation_stack_py.utils import visualizer, ParameterHandler
from navigation_stack_py.global_planner import (
    DijkstraPlanner,
    RRTPlanner,
    RRTStarPlanner,
)
from navigation_stack_py.local_planner import DWAPlanner, RandomizedMPCPlanner

# Parameters
config_path = "config/navigation/square/default.yaml"

params = ParameterHandler()
params.init(config_path)

# Simulator
simulator = Simulator()
static_map = simulator.reset(params)

# Global planner
# global_planner = DijkstraPlanner(params)
# global_planner = RRTPlanner(params)
global_planner = RRTStarPlanner(params)

# Local planner
local_planner = DWAPlanner(params)
# local_planner = RandomizedMPCPlanner(params)

# Set goal randomly
goal_state = params.goal_state

save_animation = False
save_folder = "/home/honda/Videos/"

frames = []

max_frame = 200

for _ in range(max_frame):

    # Observation
    robot_obs = simulator.get_observation()

    # Calculate global path
    global_planner.set_costmap(robot_obs.static_map_with_scan)
    reference_path_ij, reference_path_xy = global_planner.make_plan(
        robot_obs.state.pos, goal_state.pos
    )

    # Calculate local path and control command
    local_planner.set_costmap(robot_obs.static_map_with_scan)
    # local_planner.set_costmap(robot_obs.scan_points)
    local_planner_output = local_planner.compute_velocity_command(
        robot_obs, goal_state, reference_path_xy
    )

    # Update simulator
    obstacle_map, robot_traj = simulator.step(local_planner_output.control_command)

    # Get inflated map
    inflated_map = global_planner.get_costmap()
    # inflated_map = local_planner.get_costmap()

    # Rendering
    image_arr = visualizer.render(
        static_map=static_map,
        obstacle_map=obstacle_map,
        inflation_layer=inflated_map.get_inflation_layer(),
        robot_observation=robot_obs,
        robot_traj=robot_traj,
        goal_state=goal_state,
        robot_radius=params.robot_radius,
        global_path=reference_path_ij,
        local_path_list=local_planner_output.predict_path_list,
        local_path_best_index=local_planner_output.best_index,
        sub_goal_index=local_planner_output.sub_goal_index,
    )

    frames.append([plt.imshow(image_arr)])

    if save_animation:
        pass
    else:
        # For faster rendering
        plt.pause(0.01)
        plt.clf()

    if local_planner_output.is_goal_reached:
        print("Goal reached!")
        break

# save animation from image list
if save_animation:
    anim = animation.ArtistAnimation(plt.gcf(), frames, interval=100)
    anim.save(save_folder + "animation.gif", fps=10, writer="ffmpeg")

plt.close()
