"""
Part of this script is copied from https://github.com/DLR-RM/stable-baselines3
"""

import warnings
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import yaml
import gym
import numpy as np

from stable_baselines3.common import base_class
from stable_baselines3.common.vec_env import VecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor

from navigation_stack_py.gym_env import NavigationStackEnv
from navigation_stack_py.utils import DataLogger


class HyperParameters:
    def __init__(self, config: dict, mode: str):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        project_dir = os.path.join(current_dir, "..")
        root_log_dir = os.path.join(project_dir, config[mode]["log_root_dir"])

        ## Common parameters ##
        log_save_dir = config[mode]["log_save_dir"]
        self.log_dir = os.path.join(root_log_dir, log_save_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        self.model_dir = os.path.join(root_log_dir, config[mode]["model_dir"])
        os.makedirs(self.model_dir, exist_ok=True)
        self.saved_model_path = os.path.join(self.model_dir, config[mode]["model_name"])
        self.env_id = config[mode]["env_id"]
        self.rl_algorithm = config[mode]["rl_algorithm"]
        rl_config_path = os.path.join(project_dir, config[mode]["rl_config"])
        self.rl_config = yaml.safe_load(open(rl_config_path, "r"))
        self.navigation_scenario_path_list = []
        for navigation_scenario_path in config[mode]["navigation_scenarios"]:
            self.navigation_scenario_path_list.append(
                os.path.join(project_dir, navigation_scenario_path)
            )
        self.navigation_scenario_list = []
        for navigation_scenario_path in self.navigation_scenario_path_list:
            self.navigation_scenario_list.append(
                yaml.safe_load(open(navigation_scenario_path, "r"))
            )
        self.navigation_config = config["navigation"]

        # unique parameters
        if mode == "train":
            self.num_processes = config[mode]["num_processes"]
            self.seed = config[mode]["seed"]
            self.num_test_episodes = config[mode]["num_test_episodes"]
            self.is_generate_random_map = config[mode]["is_generate_random_map"]
            self.random_map_num = config[mode]["random_map_num"]
            self.map_config = os.path.join(project_dir, config[mode]["map_config"])
            self.train_mode = config[mode]["train_mode"]
            self.total_timesteps = config[mode]["total_timesteps"]
            self.tensor_board_path = os.path.join(
                self.log_dir, config[mode]["tensor_board_path"]
            )
            self.model_save_freq = config[mode]["model_save_freq"]
            self.tensorboard_check_freq = config[mode]["tensorboard_check_freq"]
            self.verbose = config[mode]["verbose"]
            self.dqn_params = config[mode]["dqn_params"]
            self.her_params = config[mode]["her_params"]
            self.prb_params = config[mode]["prb_params"]
        elif mode == "eval":
            self.num_processes = config[mode]["num_processes"]
            self.seed = config[mode]["seed"]
            self.is_generate_random_map = config[mode]["is_generate_random_map"]
            self.random_map_num = config[mode]["random_map_num"]
            self.map_config = os.path.join(project_dir, config[mode]["map_config"])
            self.eval_method_list = config[mode]["eval_method_list"]
            self.save_result = config[mode]["save_result"]
            self.num_eval_episodes = config[mode]["num_eval_episodes"]
        elif mode == "run":
            self.method_list = config[mode]["method_list"]
            self.seeds = config[mode]["seeds"]
            self.save_animation = config["run"]["save_animation"]
            self.save_figure = config["run"]["save_figure"]
            self.view_animation = config[mode]["view_animation"]
            self.visualize_mode = config["run"]["visualize_mode"]
            self.movie_type = config["run"]["movie_type"]
        elif mode == "tune_baselines":
            self.method = config[mode]["method"]
            self.num_eval_episodes = config[mode]["num_eval_episodes"]
            self.num_processes = config[mode]["num_processes"]
            self.seed = config[mode]["seed"]
        else:
            raise ValueError("mode should be train or eval")


def _make_env(params: HyperParameters, seed: int) -> Callable:
    """
    Make multiple environments
    """

    def _init() -> gym.Env:
        set_random_seed(seed)
        env = gym.make(
            params.env_id,
            navigation_config=params.navigation_config,
            scenario_list=params.navigation_scenario_list,
            seed=seed,
            rl_config=params.rl_config,
        )
        env.seed(seed)
        env.action_space.seed(seed)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = Monitor(env, allow_early_resets=True)
        return env

    return _init


def make_proc_env(params: HyperParameters, rank: int, num_env: int) -> VecEnv:
    """
    Make subprocessed vectorized environments
    """
    return SubprocVecEnv(
        [_make_env(params=params, seed=rank + i) for i in range(num_env)]
    )


def evaluate_policy(
    method: Union["base_class.BaseAlgorithm", str],
    params: HyperParameters,
    n_eval_episodes: int,
    num_workers: int,
    rank: int = 0,
) -> List[DataLogger]:
    """
    Evaluate policy with ``n_eval_episodes``.
    param method: RL model or method name
    param n_eval_episodes: number of episodes to evaluate
    param num_workers: number of parallel workers
    param rank: offset of seed
    return data_logger_list [data_logger], len=n_eval_episodes
    """
    # evaluate seed list
    seed_list = [rank + i for i in range(n_eval_episodes)]

    # divide seed list into batch list
    seed_batch_list = [
        seed_list[i : i + num_workers] for i in range(0, len(seed_list), num_workers)
    ]

    # data logger list for each seed
    data_logger_list = [DataLogger() for _ in range(n_eval_episodes)]

    progress = 0

    for seed_batch in seed_batch_list:
        n_envs = len(seed_batch)
        envs = make_proc_env(params, rank=seed_batch[0], num_env=n_envs)

        episode_counts = np.zeros(n_envs, dtype="int")
        episode_count_targets = np.ones(n_envs, dtype="int")

        observations = envs.reset()
        episode_starts = np.ones((envs.num_envs,), dtype=bool)

        while (episode_counts < episode_count_targets).any():
            if isinstance(method, base_class.BaseAlgorithm):
                actions, _ = method.predict(observations, deterministic=True)
            elif isinstance(method, str):
                actions = envs.env_method(method)
            else:
                raise ValueError(
                    "method must be either a RL model or a method name implemented in the environment"
                )
            observations, rewards, dones, infos = envs.step(actions)
            for i in range(n_envs):
                if episode_counts[i] < episode_count_targets[i]:
                    # unpack values so that the callback can access the local variables
                    done = dones[i]
                    info = infos[i]
                    data = info["data"]
                    index_in_seed_list = seed_batch[i] - rank
                    data_logger_list[index_in_seed_list].log(data)
                    episode_starts[i] = done

                    if dones[i]:
                        progress += 1
                        episode_counts[i] += 1
                        print(f"progress: {progress}/{n_eval_episodes}")

    return data_logger_list
