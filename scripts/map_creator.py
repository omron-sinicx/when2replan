# This script creates a map with a defined number of rooms and a defined number of doors.
# Original script is in https://github.com/ignc-research/arena-rosnav

from urllib.parse import SplitResult
import numpy as np
import matplotlib.pyplot as plt
import datetime
from PIL import Image
import os
import yaml
import random
import argparse


def create_yaml_files(map_name, dir_path, resolution):
    map_yaml = {
        "image": "{0}.pgm".format(map_name),
        "resolution": resolution,  # m/pixel
        "origin": [0.0, 0.0, 0.0],  # [-x,-y,0.0]
        "negate": 0,
        "occupied_thresh": 0.65,
        "free_thresh": 0.196,
    }
    with open(dir_path + "/{}/map.yaml".format(map_name), "w") as outfile:
        yaml.dump(map_yaml, outfile, sort_keys=False, default_flow_style=None)

    # world_yaml_properties = {"properties": {
    #     "velocity_iterations": 10, "position_iterations": 10}}
    # world_yaml_layers = {"layers": [
    #     {"name": "static", "map": "map.yaml", "color": [0, 1, 0, 1]}
    # ]}
    # with open(dir_path+"/{}/map.world.yaml".format(map_name), 'w') as outfile:
    #     # somehow the first part must be with default_flow_style=False
    #     yaml.dump(world_yaml_properties, outfile,
    #               sort_keys=False, default_flow_style=False)
    #     # 2nd part must be with default_flow_style=None
    #     yaml.dump(world_yaml_layers, outfile,
    #               sort_keys=False, default_flow_style=None)


# create pgm file from occupancy map (1:occupied, 0:free) and the necessary yaml files
def make_image(map, maptype, map_name, resolution, save_dir):
    now = datetime.datetime.now()
    img = Image.fromarray(((map - 1) ** 2 * 255).astype("uint8"))  # monochromatic image

    if map_name == "":
        map_name = maptype
    # get path for current file, does not work if os.chdir() was used, NOTE: change directory if needed
    current_dir_path = os.path.dirname(os.path.realpath(__file__))
    project_dir_path = os.path.join(current_dir_path, "..")
    map_dir = os.path.join(project_dir_path, save_dir)
    os.makedirs(map_dir, exist_ok=True)
    try:
        # create directory based on mapname where this script is located
        os.mkdir(os.path.join(map_dir, map_name))
    except:
        pass
    # save map in map directory
    img.save(map_dir + "/{0}/{0}.pgm".format(map_name))

    # create corresponding yaml files
    create_yaml_files(map_name, map_dir, resolution)


# create empty map with format given by height,width and initialize empty tree
def initialize_map(height, width, type="indoor"):
    if type == "outdoor":
        map = np.tile(1, [height, width])
        margin = 0  # If the wall is too thin, pymap2d's lidar scan simulation may show through
        # https://github.com/danieldugas/pymap2d/blob/master/CMap2D.pyx#L1485
        map[slice(margin, height - margin), slice(margin, width - margin)] = 0
        return map
    else:
        return np.tile(1, [height, width])


def insert_root_node(map, tree):  # create root node in center of map
    root_node = [int(np.floor(map.shape[0] / 2)), int(np.floor(map.shape[1] / 2))]
    map[root_node[0], root_node[1]] = 0
    tree.append(root_node)


# sample position from map within boundary and leave tolerance for corridor width
def sample(map, corridor_radius):
    random_x = np.random.choice(
        range(corridor_radius + 2, map.shape[0] - corridor_radius - 1, 1)
    )
    random_y = np.random.choice(
        range(corridor_radius + 2, map.shape[1] - corridor_radius - 1, 1)
    )
    return [random_x, random_y]


def specify(map, position):
    x = position[0]
    y = position[1]

    # range check
    max_x = map.shape[0] - 1
    max_y = map.shape[1] - 1
    if x < 0 or x > max_x or y < 0 or y > max_y:
        raise ValueError("Position out of range")

    return [x, y]


# find nearest node according to L1 norm
def find_nearest_node(random_position, tree):
    nearest_node = []
    min_distance = np.inf
    for node in tree:
        distance = sum(np.abs(np.array(random_position) - np.array(node)))
        if distance < min_distance:
            min_distance = distance
            nearest_node = node
    return nearest_node


# insert new node into the map and tree
def insert_new_node(random_position, tree, map):
    map[random_position[0], random_position[1]] = 0
    tree.append(random_position)


def get_constellation(node1, node2):
    # there are two relevant constellations for the 2 nodes, which must be considered when creating the horizontal and vertical path
    # 1: lower left and upper right
    # 2: upper left and lower right
    # each of the 2 constellation have 2 permutations which must be considered as well
    constellation1 = {
        # x1>x2 and y1<y2
        "permutation1": node1[0] > node2[0] and node1[1] < node2[1],
        "permutation2": node1[0] < node2[0] and node1[1] > node2[1],
    }  # x1<x2 and y1>y2
    if constellation1["permutation1"] or constellation1["permutation2"]:
        return 1
    else:
        return 2


def create_path(node1, node2, corridor_radius, map):
    coin_flip = np.random.random()
    # x and y coordinates must be sorted for usage with range function
    x1, x2 = sorted([node1[0], node2[0]])
    y1, y2 = sorted([node1[1], node2[1]])
    if get_constellation(node1, node2) == 1:  # check which constellation
        # randomly determine the curvature of the path (right turn/left turn)
        if coin_flip >= 0.5:
            map[
                slice(x1 - corridor_radius, x1 + corridor_radius + 1),
                range(y1 - corridor_radius, y2 + 1 + corridor_radius, 1),
            ] = 0  # horizontal path
            map[
                range(x1 - corridor_radius, x2 + 1 + corridor_radius, 1),
                slice(y1 - corridor_radius, y1 + corridor_radius + 1),
            ] = 0  # vertical path
        else:
            map[
                slice(x2 - corridor_radius, x2 + corridor_radius + 1),
                range(y1 - corridor_radius, y2 + 1 + corridor_radius, 1),
            ] = 0  # horizontal path
            map[
                range(x1 - corridor_radius, x2 + 1 + corridor_radius, 1),
                slice(y2 - corridor_radius, y2 + corridor_radius + 1),
            ] = 0  # vertical path
    else:
        # randomly determine the curvature of the path (right turn/left turn)
        if coin_flip >= 0.5:
            map[
                slice(x1 - corridor_radius, x1 + corridor_radius + 1),
                range(y1 - corridor_radius, y2 + 1 + corridor_radius, 1),
            ] = 0  # horizontal path
            map[
                range(x1 - corridor_radius, x2 + 1 + corridor_radius, 1),
                slice(y2 - corridor_radius, y2 + corridor_radius + 1),
            ] = 0  # vertical path
        else:
            map[
                slice(x2 - corridor_radius, x2 + corridor_radius + 1),
                range(y1 - corridor_radius, y2 + 1 + corridor_radius, 1),
            ] = 0  # horizontal path
            map[
                range(x1 - corridor_radius, x2 + 1 + corridor_radius, 1),
                slice(y1 - corridor_radius, y1 + corridor_radius + 1),
            ] = 0  # vertical path


def create_rooms(map, tree, room_number, room_width, room_height, no_overlap):
    rooms_created = 0
    rooms_list = []
    room_vertical_radius = room_width // 2
    room_horizontal_radius = room_height // 2
    distance = 2 * (room_vertical_radius**2 + room_horizontal_radius**2) ** 0.5
    for room in range(1, room_number + 1):
        if rooms_created == room_number:
            break
        for i, node in enumerate(random.sample(tree, len(tree))):
            x1 = node[0] - room_horizontal_radius
            x2 = node[0] + room_horizontal_radius
            y1 = node[1] - room_vertical_radius
            y2 = node[1] + room_vertical_radius

            # if room is out of map
            if x1 < 0 or x2 > map.shape[1] or y1 < 0 or y2 > map.shape[0]:
                if (i == len(tree) - 1) and (room > rooms_created):
                    print("No valid position for room " + str(room) + " found.")
                continue

            if rooms_created == room_number:
                break
            if no_overlap:
                if len(rooms_list) == 0:
                    try:
                        map[x1, y1]
                        map[x1, y2]
                        map[x2, y1]
                        map[x2, y2]
                        map[slice(x1, x2), slice(y1, y2)] = 0
                        rooms_created += 1
                        rooms_list.append(node)
                        print(
                            "Room "
                            + str(room)
                            + " created at position "
                            + str(node[0])
                            + ","
                            + str(node[1])
                        )
                        break
                    except:
                        pass
                else:
                    for room_node in rooms_list:
                        collision = False
                        if (
                            np.linalg.norm(np.array(node) - np.array(room_node))
                            < distance
                        ):
                            if (i == len(tree) - 1) and (room > rooms_created):
                                print(
                                    "No valid position for room "
                                    + str(room)
                                    + " found."
                                )
                            collision = True
                            break
                    if collision:
                        continue
                    try:
                        map[x1, y1]
                        map[x1, y2]
                        map[x2, y1]
                        map[x2, y2]
                        map[slice(x1, x2), slice(y1, y2)] = 0
                        rooms_created += 1
                        rooms_list.append(node)
                        print(
                            "Room "
                            + str(room)
                            + " created at position "
                            + str(node[0])
                            + ","
                            + str(node[1])
                        )
                        break
                    except:
                        pass

            else:
                try:
                    map[x1, y1]
                    map[x1, y2]
                    map[x2, y1]
                    map[x2, y2]
                    map[slice(x1, x2), slice(y1, y2)] = 0
                    rooms_created += 1
                    rooms_list.append(node)
                    print(
                        "Room "
                        + str(room)
                        + " created at position "
                        + str(node[0])
                        + ","
                        + str(node[1])
                    )
                    break
                except:
                    pass


def create_indoor_map(
    height,
    width,
    corridor_radius,
    iterations,
    room_number,
    room_width,
    room_height,
    no_overlap,
):
    tree = []  # initialize empty tree
    map = initialize_map(height, width)
    insert_root_node(map, tree)
    for i in range(iterations):  # create as many paths/nodes as defined in iteration
        random_position = sample(map, corridor_radius)
        # nearest node must be found before inserting the new node into the tree, else nearest node will be itself
        nearest_node = find_nearest_node(random_position, tree)
        insert_new_node(random_position, tree, map)
        create_path(random_position, nearest_node, corridor_radius, map)
    create_rooms(map, tree, room_number, room_width, room_height, no_overlap)
    return map


def create_outdoor_map(height, width, obstacle_number, obstacle_extra_radius):
    map = initialize_map(height, width, type="outdoor")
    for i in range(obstacle_number):
        random_position = sample(map, obstacle_extra_radius)
        map[
            slice(
                random_position[0] - obstacle_extra_radius,
                random_position[0] + obstacle_extra_radius + 1,
            ),  # create 1 pixel obstacles with extra radius if specified
            slice(
                random_position[1] - obstacle_extra_radius,
                random_position[1] + obstacle_extra_radius + 1,
            ),
        ] = 1
    return map


def create_simplest_map(
    height: float, width: float, obstacle_positions: list, obstacle_sizes: list
):
    map = initialize_map(height, width, type="outdoor")
    for position in obstacle_positions:
        pos = specify(map=map, position=position)
        map[
            slice(
                pos[0] - obstacle_sizes[obstacle_positions.index(position)][0],
                pos[0] + obstacle_sizes[obstacle_positions.index(position)][1] + 1,
            ),  # create 1 pixel obstacles with extra radius if specified
            slice(
                pos[1] - obstacle_sizes[obstacle_positions.index(position)][1],
                pos[1] + obstacle_sizes[obstacle_positions.index(position)][1] + 1,
            ),
        ] = 1
    return map


def create_uniform_map(height: int, width: int, obstacle_num: list, obstacle_size: int):
    map = initialize_map(height, width, type="outdoor")
    # マップサイズに対して障害物を均一に配置する
    obstacle_positions = []
    # 縦方向の障害物の数
    obstacle_num_width = obstacle_num[0]
    obstacle_num_height = obstacle_num[1]
    obstacle_interval_y = int(height // obstacle_num_width)
    obstacle_interval_x = int(width // obstacle_num_height)

    obstacle_position_y = obstacle_interval_y // 2
    for i in range(obstacle_num_width):
        # 横方向の障害物の配置位置
        obstacle_position_x = obstacle_interval_x // 2
        for j in range(obstacle_num_height):
            obstacle_positions.append([obstacle_position_x, obstacle_position_y])
            obstacle_position_x += obstacle_interval_x
        obstacle_position_y += obstacle_interval_y
    for position in obstacle_positions:
        map[
            slice(
                position[0] - obstacle_size, position[0] + obstacle_size + 1
            ),  # create 1 pixel obstacles with extra radius if specified
            slice(position[1] - obstacle_size, position[1] + obstacle_size + 1),
        ] = 1
    return map


def generate_random_maps(map_config_path: str, random_map_num: int, seed: int):
    with open(map_config_path, "r") as f:
        config = yaml.safe_load(f)

    height = config["common"]["height"]
    width = config["common"]["width"]
    resolution = config["common"]["resolution"]
    corridor_radius = config["indoor"]["corridor_radius"]
    iterations = config["indoor"]["iterations"]
    room_number = config["indoor"]["room_number"]
    room_width = config["indoor"]["room_width"]
    room_height = config["indoor"]["room_height"]
    no_overlap = config["indoor"]["no_overlap"]
    obstacle_number = config["outdoor"]["obstacle_number"]
    obstacle_extra_radius = config["outdoor"]["obstacle_extra_radius"]
    save_dir = config["common"]["save_dir"]
    outdoor_map_rate = config["random"]["outdoor_map_rate"]

    outdoor_map_num = int(random_map_num * outdoor_map_rate)
    indoor_map_num = random_map_num - outdoor_map_num

    random.seed(seed)
    np.random.seed(seed)
    # generate indoor maps
    for i in range(indoor_map_num):
        map_name = "indoor_" + str(i)

        indoor_map = create_indoor_map(
            height,
            width,
            corridor_radius,
            iterations,
            room_number,
            room_width,
            room_height,
            no_overlap,
        )
        make_image(indoor_map, None, map_name, resolution, save_dir)

    # generate outdoor maps
    for i in range(outdoor_map_num):
        map_name = "outdoor_" + str(i)
        outdoor_map = create_outdoor_map(
            height, width, obstacle_number, obstacle_extra_radius
        )
        make_image(outdoor_map, None, map_name, resolution, save_dir)


"""
    args:
        config_name: config file name in config folder
"""
if __name__ == "__main__":
    # arguments: config name
    parser = argparse.ArgumentParser(description="Create 2D map as pgm format")
    parser.add_argument("config", type=str, help="config file name")

    # map generation parameters loaded from config.yaml
    dir_path = os.path.dirname(os.path.realpath(__file__))
    project_path = os.path.join(dir_path, "..")
    config_dir = os.path.join(project_path, "config/map_creator")
    config_name = parser.parse_args().config + ".yaml"
    config_path = os.path.join(config_dir, config_name)
    with open(config_path) as f:
        config = yaml.safe_load(f)
    height = config["common"]["height"]
    width = config["common"]["width"]
    resolution = config["common"]["resolution"]
    maptype = config["common"]["maptype"]
    map_name = config["common"]["map_name"]
    save_dir = os.path.join(project_path, config["common"]["save_dir"])

    # create outdoor map
    if maptype == "outdoor":
        obstacle_number = config["outdoor"]["obstacle_number"]
        obstacle_extra_radius = config["outdoor"]["obstacle_extra_radius"]
        outdoor_map = create_outdoor_map(
            height, width, obstacle_number, obstacle_extra_radius
        )
        make_image(outdoor_map, maptype, map_name, resolution, save_dir)
    # create indoor map
    elif maptype == "indoor":
        corridor_radius = config["indoor"]["corridor_radius"]
        iterations = config["indoor"]["iterations"]
        room_number = config["indoor"]["room_number"]
        room_width = config["indoor"]["room_width"]
        room_height = config["indoor"]["room_height"]
        no_overlap = config["indoor"]["no_overlap"]
        indoor_map = create_indoor_map(
            height,
            width,
            corridor_radius,
            iterations,
            room_number,
            room_width,
            room_height,
            no_overlap,
        )
        make_image(indoor_map, maptype, map_name, resolution, save_dir)
    # create
    elif maptype == "simplest":
        obstacle_positions = config["simplest"]["obstacle_positions"]
        obstacle_sizes = config["simplest"]["obstacle_sizes"]
        simplest_map = create_simplest_map(
            height, width, obstacle_positions, obstacle_sizes
        )
        make_image(simplest_map, maptype, map_name, resolution, save_dir)
    elif maptype == "uniform":
        obstacle_num = config["uniform"]["obstacle_num"]
        obstacle_size = config["uniform"]["obstacle_size"]
        uniform_map = create_uniform_map(height, width, obstacle_num, obstacle_size)
        make_image(uniform_map, maptype, map_name, resolution, save_dir)
