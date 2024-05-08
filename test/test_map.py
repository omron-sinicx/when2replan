from navigation_stack_py.utils import MapHandler
from CMap2D import gridshow
from matplotlib import pyplot as plt
import numpy as np
import pytest
import fire

"""
For pytest
"""


@pytest.fixture
def map_handler_fixture():
    map_path = "../maps/square/map.yaml"
    map_handler = MapHandler()
    map_handler.load_map_from_file(map_path)
    return map_handler


def test_get_map(map_handler_fixture):
    occpupancy = map_handler_fixture.get_map_as_np("occupancy")
    sdf = map_handler_fixture.get_map_as_np("sdf")
    assert occpupancy.shape == (128, 128)
    assert sdf.shape == (128, 128)
    assert map_handler_fixture.get_image_size() == (128, 128)

    assert map_handler_fixture.get_resolution() == pytest.approx(0.2)


def test_calc_relative_pose(map_handler_fixture):
    source_xy = np.array([0.0, 0.0])
    source_yaw = 0.0
    target_ij = np.array([1, 0])
    relative_pose = map_handler_fixture.calc_relative_pose(
        source_xy, source_yaw, target_ij
    )
    assert relative_pose[:2] == pytest.approx([-12.8, -12.8], abs=0.5)
    assert relative_pose[2] == pytest.approx(-np.pi * 0.75, abs=0.1)

    source_xy = np.array([1.0, 2.0])
    source_yaw = np.pi * 0.5
    target_xy = np.array([-3.0, 4.0])
    target_ij = map_handler_fixture.pose2index(target_xy)

    correct_xy = target_xy - source_xy
    correct_yaw = np.arctan2(correct_xy[1], correct_xy[0]) - source_yaw
    relative_pose = map_handler_fixture.calc_relative_pose(
        source_xy, source_yaw, target_ij
    )
    assert relative_pose[:2] == pytest.approx(correct_xy, abs=0.5)
    assert relative_pose[2] == pytest.approx(correct_yaw, abs=0.1)


"""
For example of usage
e.g. python3 test_map.py view_map
"""

map_path = "../maps/square/map.yaml"


def map_size():
    map_handler = MapHandler()
    map_handler.load_map_from_file(map_path)

    image_shape = map_handler.get_image_size()
    print(image_shape)

    map_size = map_handler.get_map_size()
    print(map_size)


def view_map():
    map_handler = MapHandler()
    map_handler.load_map_from_file(map_path)

    map_handler.view_map("occupancy")
    map_handler.view_map("sdf")

    plt.show()


def convert_coordinate():
    map_handler = MapHandler()
    map_handler.load_map_from_file(map_path)
    pose = np.array([0.0, 0.0])

    # one pose to one coordinate
    index = map_handler.pose2index(pose)
    print(index)

    pose_arr = np.array([[0.0, 0.0], [1.0, 2.0], [4.0, 5.0]])

    index_arr = map_handler.pose_array2index_array(pose_arr)
    print(index_arr)

    pose = map_handler.index2pose(index)
    print(pose)

    pose_arr = map_handler.index_array2pose_array(index_arr)
    print(pose_arr)


def get_nearest_dist():
    map_handler = MapHandler()
    map_handler.load_map_from_file(map_path)
    pose = np.array([0.0, 0.0])
    radius = 5.0

    dist = map_handler.get_obs_dist(pose)

    print(dist)

    print(map_handler.check_collision(pose, radius))


def compute_dijkstra():
    map_handler = MapHandler()
    map_handler.load_map_from_file(map_path)
    map_handler.load_map_from_file(map_path)
    goal_pose = np.array([10.0, 10.0])
    mask = None
    extra_costs = None

    dijk_map = map_handler.construct_dijkstra_map(goal_pose, mask, extra_costs)

    print(type(dijk_map))

    start_pose = np.array([0.0, 0.0])

    shortest_path_xy, shortest_path_ij, cost = map_handler.compute_shortest_path_dijkstra(
        start_pose, dijk_map
    )

    print(shortest_path_xy)


def compute_scan():
    map_handler = MapHandler()
    map_handler.load_map_from_file(map_path)

    pos = np.array([0.0, 0.0])

    angle_min = -np.pi / 2
    angle_max = np.pi

    scan_point_num = 360

    angles = np.linspace(angle_min, angle_max, scan_point_num)

    range = 10  # [m]
    ranges = np.ones(scan_point_num) * range

    scan = map_handler.compute_lidar_scan_map(pos, angles, ranges)

    map_handler.view_map("occupancy")

    plt.figure("scan")
    plt.title("scan")
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
    gridshow(scan)

    np.set_printoptions(threshold=np.inf)
    # print(scan)
    index = map_handler.pose2index(pos)
    print(map_handler.get_map_as_np("occupancy")[index[0]][index[1]])

    plt.show()


def detect_obstacle():
    map_handler = MapHandler()
    map_handler.load_map_from_file(map_path)
    pos = np.array([10.0, 10.0])

    angle_min = -np.pi
    angle_max = np.pi

    scan_point_num = 360

    angles = np.linspace(angle_min, angle_max, scan_point_num)

    range = 10  # [m]
    ranges = np.ones(scan_point_num) * range

    static_map = map_handler.get_map_as_np("occupancy")

    detected_map = map_handler.get_detected_map(pos, angles, ranges)

    fig, axes = plt.subplots(1, 1, figsize=(8, 8))

    axes.imshow(detected_map.T, cmap="Reds", vmin=0, vmax=1)

    index = map_handler.pose2index(pos)
    print(static_map[index[0]][index[1]])

    plt.show()


def merge_map():
    base_map = MapHandler()
    base_map.load_map_from_file(map_path)
    obs_map = MapHandler()
    obs_map.load_map_from_file("../maps/star.yaml")

    new_map = base_map.merge_map(obs_map)

    base_map.view_map("occupancy")

    plt.show()

    obs_map.view_map("occupancy")

    plt.show()

    new_map.view_map("occupancy")

    plt.show()


def inflation_map():
    base_map = MapHandler()
    base_map.load_map_from_file(map_path)

    base_map.set_occupancy_map_with_inflation(1.0)

    fig, axes = plt.subplots(1, 1, figsize=(8, 8))

    axes.imshow(base_map.get_map_as_np("occupancy").T, cmap="Reds", vmin=0, vmax=1)

    plt.show()


if __name__ == "__main__":
    fire.Fire()
