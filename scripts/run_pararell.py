from enum import Enum, auto
from random import seed
import gym
import yaml
import numpy as np
import os
import time
import argparse
import matplotlib.animation as animation
from matplotlib import pyplot as plt
from navigation_stack_py.gym_env import NavigationStackEnv
from navigation_stack_py.utils import DataLogger

from gym import envs
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.utils import set_random_seed

from omegaconf import OmegaConf, DictConfig
import hydra


class RunParameters:
    model_dir: str = None
    log_dir: str = None
    navigation_config: dict = None
    scenario: dict = None
    env_id: str = None
    model_name: str = None
    rl_model: str = None
    seeds: list = None
    save_animation: bool = None
    movie_type: str = None
    method_list: list = None

    def __init__(self, config: yaml):
        current_path = os.path.dirname(os.path.abspath(__file__))
        project_path = os.path.dirname(current_path)
        self.navigation_config = config["navigation"]
        scenario_path = os.path.join(project_path, config["run"]["scenario"])
        self.scenario = yaml.safe_load(open(scenario_path, "r"))
        self.rl_model = config["run"]["rl_model"]
        self.model_name = config["run"]["model_name"]
        self.env_id = config["run"]["env_id"]
        self.method_list = config["run"]["method_list"]

        model_dir_name = config["run"]["model_dir"]
        self.model_dir = os.path.join(project_path, model_dir_name)
        rl_config_path = os.path.join(project_path, config["run"]["rl_config"])
        self.rl_config = yaml.safe_load(open(rl_config_path, "r"))
        save_dir_name = config["run"]["log_save_dir"]
        self.log_dir = os.path.join(project_path, save_dir_name)
        os.makedirs(self.log_dir, exist_ok=True)
        self.seeds = config["run"]["seeds"]
        self.save_animation = config["run"]["save_animation"]
        self.visualize_mode = config["run"]["visualize_mode"]
        self.movie_type = config["run"]["movie_type"]


class Mode(Enum):
    MODEL = "model"
    TIMER_BASELINE = "timer-triggered"
    EVENT_TRIGGERED_BASELINE = "event-triggered"
    HYBRID_BASELINE = "hybrid-triggered"
    RANDOM_BASELINE = "random"
    MANUAL = "manual"


def run(params: RunParameters, seed: int):
    frames = []
    model_path = os.path.join(params.model_dir, params.model_name)
    # check existence of model
    if not os.path.exists(model_path):
        raise Exception("Model does not exist: {}".format(model_path))

    if params.rl_model == "DQN":
        model = DQN.load(model_path)
    elif params.rl_model == "PPO":
        model = PPO.load(model_path)
    elif params.rl_model == "A2C":
        model = A2C.load(model_path)
    else:
        raise Exception("Unknown model: {}".format(params.rl_model))

    env_rl = gym.make(
        params.env_id,
        navigation_config=params.navigation_config,
        scenario=params.scenario,
        seed=seed,
        visualize_mode=params.visualize_mode,
        rl_config=params.rl_config,
        is_training=False,
    )

    env_timer = gym.make(
        params.env_id,
        navigation_config=params.navigation_config,
        scenario=params.scenario,
        seed=seed,
        visualize_mode=params.visualize_mode,
        rl_config=params.rl_config,
        is_training=False,
    )

    env_event = gym.make(
        params.env_id,
        navigation_config=params.navigation_config,
        scenario=params.scenario,
        seed=seed,
        visualize_mode=params.visualize_mode,
        rl_config=params.rl_config,
        is_training=False,
    )

    env_random = gym.make(
        params.env_id,
        navigation_config=params.navigation_config,
        scenario=params.scenario,
        seed=seed,
        visualize_mode=params.visualize_mode,
        rl_config=params.rl_config,
        is_training=False,
    )

    obs_rl = env_rl.reset()
    _ = env_timer.reset()
    _ = env_event.reset()
    _ = env_random.reset()
    for i in range(env_rl.get_max_steps()):
        action_rl, _ = model.predict(obs_rl, deterministic=True)
        action_timer = env_timer.timer_baseline()
        action_event = env_event.event_triggered_baseline()
        action_random = env_random.random_baseline()

        obs_rl, _, done_rl, _ = env_rl.step(action_rl)
        _, _, done_timer, _ = env_timer.step(action_timer)
        _, _, done_event, _ = env_event.step(action_event)
        _, _, done_random, _ = env_random.step(action_random)

        image_rl = env_rl.render(mode="rgb_array")
        image_timer = env_timer.render(mode="rgb_array")
        image_event = env_event.render(mode="rgb_array")
        image_random = env_random.render(mode="rgb_array")

        image_up = np.concatenate((image_rl, image_timer), axis=1)
        image_down = np.concatenate((image_event, image_random), axis=1)
        composed_image = np.concatenate((image_up, image_down), axis=0)

        if params.save_animation:
            frames.append([plt.imshow(composed_image)])
        else:
            plt.imshow(composed_image)
            plt.pause(0.01)
            plt.clf()

        if done_rl and done_timer and done_event and done_random:
            break

    if params.save_animation:
        anim = animation.ArtistAnimation(plt.gcf(), frames, interval=100)
        movie_name = "parallel" + "_" + str(seed) + "." + params.movie_type
        save_dir = os.path.join(params.log_dir, str(seed))
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, movie_name)
        anim.save(save_path, fps=10, writer="ffmpeg")


def run_all(params: RunParameters):
    for seed in params.seeds:
        print("Running seed: {}".format(seed))
        set_random_seed(seed)
        run(params, seed)


@hydra.main(config_name="default.yaml", config_path="../config/run", version_base=None)
def main(cfg: DictConfig):
    # arg pars
    config = OmegaConf.to_container(cfg, resolve=True)
    params = RunParameters(config)

    run_all(params)


if __name__ == "__main__":
    main()
