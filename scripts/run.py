from typing import List

from enum import Enum, auto
import gym
import yaml
import os
import time
import pickle
import numpy as np
import matplotlib.animation as animation
from matplotlib import pyplot as plt
from navigation_stack_py.gym_env import NavigationStackEnv, NavigationStackGoalEnv
from navigation_stack_py.utils import DataLogger

from gym import envs
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from navigation_stack_py.rl_modules.dqn.dqn import DQN
from utils import HyperParameters

from expert import ExpertActor

from omegaconf import OmegaConf, DictConfig
import hydra


def synthesis_image_arrays(
    base_image: np.ndarray, additional_image: np.ndarray
) -> np.ndarray:
    """
    Synthesis image arrays, base_image + additional_image; overlapped area is overwritten by additional_image
    params: base_image (N, N, 3)
    params: additional_image (N, N, 3)
    """
    image = base_image.copy()

    # Overwrite base_image with additional_image if additional_image is not white
    image[additional_image != 255] = additional_image[additional_image != 255]

    return image


def run(method_name: str, params: HyperParameters, seed: int):
    print("Running {}...".format(method_name))
    print("Seed: {}".format(seed))
    set_random_seed(seed)
    env = gym.make(
        params.env_id,
        navigation_config=params.navigation_config,
        scenario_list=params.navigation_scenario_list,
        seed=seed,
        visualize_mode=params.visualize_mode,
        save_fig=params.save_figure,
        rl_config=params.rl_config,
        save_visualized_data=True
    )
    env.seed(seed)
    env.action_space.seed(seed)
    env = Monitor(env, allow_early_resets=True)

    frames = []
    if method_name == "rl_based_replan":
        model_path = params.saved_model_path
        # check existence of model
        if not os.path.exists(model_path):
            raise Exception("Model does not exist: {}".format(model_path))
        if params.rl_algorithm == "DQN":
            model = DQN.load(model_path, env=env, seed=seed)
        elif params.rl_algorithm == "PPO":
            model = PPO.load(model_path)
        elif params.rl_algorithm == "A2C":
            model = A2C.load(model_path)
        else:
            raise Exception("Unknown model: {}".format(params.rl_algorithm))

    save_dir = os.path.join(params.log_dir, str(seed))
    os.makedirs(save_dir, exist_ok=True)
    csv_name = method_name + ".csv"
    csv_logger = DataLogger(os.path.join(save_dir, csv_name))

    average_calculation_time = 0.0
    max_calculation_time = 0.0

    if method_name == "expert":
        expert = ExpertActor(seed=seed, max_hz=env.get_max_global_hz())

    obs = env.reset()

    # save map with pickle
    static_map_polygons = getattr(env, "get_static_map_polygons")()
    map_poligon_path = os.path.join(save_dir, "map.pkl")

    f = open(map_poligon_path, "wb")
    pickle.dump(static_map_polygons, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

    traj_data_list = []
    for i in range(env.get_max_steps()):
        start_time = time.time()

        if method_name == "rl_based_replan":
            action, _ = model.predict(obs, deterministic=True)
        elif method_name == "expert":
            action = expert.act(obs)
        else:
            action = getattr(env, method_name)()
        elapsed_time = time.time() - start_time

        average_calculation_time = average_calculation_time * i / (
            i + 1
        ) + elapsed_time / (i + 1)
        max_calculation_time = max(max_calculation_time, elapsed_time)

        obs, reward, done, info = env.step(action)

        # data log
        data = info["data"]

        # log q_values
        if method_name == "rl_based_replan":
            if params.rl_algorithm == "DQN":
                q_value = model.get_q_value(obs)
                for i in range(len(q_value[0])):
                    data["q_value_{}".format(i)] = q_value[0][i].item()

        csv_logger.log(data)

        # visualize data
        visualized_data_list = info["visualized_data_list"]

        traj_data_list.extend(visualized_data_list)
        
        if params.view_animation:
            image_arr_buffer = env.render(mode="rgb_array")
            for image_arr in image_arr_buffer:
                plt.axis("off")
                plt.imshow(image_arr)
                plt.pause(0.01)
                plt.clf()
        elif params.save_animation:
            plt.axis("off")
            frames.append([plt.imshow(image_arr)])

            if method_name == "manual_based_replan":
                # show animation in real time
                plt.pause(0.01)

        if done:
            break

    if params.save_animation:
        anim = animation.ArtistAnimation(plt.gcf(), frames, interval=100)
        movie_name = method_name + "_" + str(seed) + "." + params.movie_type
        save_path = os.path.join(save_dir, movie_name)
        anim.save(save_path, fps=10, writer="ffmpeg")

    if params.save_figure:
        (
            static_map_frames,
            obstacle_map_frames,
            robot_frames,
            traj_frames,
            obstacle_traj_frames,
        ) = env.get_layer_buffers()
        plt.axis("off")
        image = static_map_frames[-1]

        for i in range(0, len(robot_frames), 60):
            image = synthesis_image_arrays(image, robot_frames[i])

        for traj_frame in traj_frames:
            image = synthesis_image_arrays(image, traj_frame)

        for i in range(0, len(obstacle_map_frames), 150):
            image = synthesis_image_arrays(image, obstacle_map_frames[i])

        for i in range(0, len(obstacle_traj_frames), 10):
            image = synthesis_image_arrays(image, obstacle_traj_frames[i])

        image = synthesis_image_arrays(image, static_map_frames[-1])

        fig_type = "png"
        fig_name = method_name + "_" + str(seed) + "." + fig_type
        save_path = os.path.join(save_dir, fig_name)
        plt.imshow(image)
        plt.savefig(save_path)
        plt.clf()

    csv_logger.save()

    # save traj data with pickle for visualization
    traj_data_name = method_name + "_data.pkl"
    f = open(os.path.join(save_dir, traj_data_name), "wb")
    pickle.dump(traj_data_list, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

    print("Average calculation time [s]: {}".format(average_calculation_time))
    print("Max calculation time [s]: {}".format(max_calculation_time))


def run_all(params: HyperParameters):
    for seed in params.seeds:
        for method_name in params.method_list:
            run(method_name, params, seed)


@hydra.main(config_name="default.yaml", config_path="../config/run", version_base=None)
def main(cfg: DictConfig):
    # arg pars
    config = OmegaConf.to_container(cfg, resolve=True)
    params = HyperParameters(config=config, mode="run")
    run_all(params)


if __name__ == "__main__":
    main()
