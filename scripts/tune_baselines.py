from __future__ import annotations

from collections import defaultdict
from typing import Callable, Tuple, Union
import time
import gym
import os
import numpy as np
import yaml
import datetime
import csv
from omegaconf import OmegaConf, DictConfig
import hydra
import mlflow
import pandas as pd
from matplotlib import pyplot as plt
from navigation_stack_py.gym_env import NavigationStackEnv
from navigation_stack_py.utils import DataLogger
from navigation_stack_py.rl_modules.dqn.dqn import DQN

from map_creator import generate_random_maps
from utils import HyperParameters, evaluate_policy


@hydra.main(
    config_name="default.yaml",
    config_path="../config/tune_baselines",
    version_base=None,
)
def main(cfg: DictConfig) -> float:
    # mlflow setup
    mlflow.set_tracking_uri("file://" + hydra.utils.get_original_cwd() + "/mlruns")

    mlflow.set_experiment("tune_baselines")

    config = OmegaConf.to_container(cfg, resolve=True)
    params = HyperParameters(config=config, mode="tune_baselines")

    time_stamp = str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    run_name = time_stamp
    for navigation_scenario_path in params.navigation_scenario_path_list:
        name = (
            navigation_scenario_path.split("/")[-2]
            + "_"
            + navigation_scenario_path.split("/")[-1].split(".")[0]
        )
        run_name += "_" + name
    with mlflow.start_run(run_name=run_name):
        # parameter logging
        mlflow.log_params(config["tune_baselines"])
        mlflow.log_params(config["navigation"])
        global_planner = config["navigation"]["common"]["global_planner"]
        local_planner = config["navigation"]["common"]["local_planner"]
        mlflow.log_param("global_planner", global_planner)
        mlflow.log_param("local_planner", local_planner)
        method = config["tune_baselines"]["method"]
        mlflow.log_param("method", method)

        # eval method
        log_list = evaluate_policy(
            method=method,
            params=params,
            n_eval_episodes=params.num_eval_episodes,
            num_workers=params.num_processes,
            rank=params.seed,
        )

        reward_list = []
        for data_logger in log_list:
            reward_list.append(data_logger.sum("reward"))

        mean_reward = np.mean(reward_list)
        std_reward = np.std(reward_list)

        mlflow.log_metric("mean_reward", mean_reward)
        mlflow.log_metric("std_reward", std_reward)

    return mean_reward


if __name__ == "__main__":
    main()
