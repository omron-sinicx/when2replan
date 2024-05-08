import os
import gym
import numpy as np
import time
import random
import argparse
import yaml
import datetime
from typing import Callable, Any, Dict, Tuple
import tensorboard
import matplotlib.pyplot as plt

from navigation_stack_py.gym_env import NavigationStackEnv
from gym import envs
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import PPO, A2C, DDPG, SAC, TD3
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.logger import Figure, HParam
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video
from imitation.algorithms import bc
from imitation.data.types import Trajectory
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper

from navigation_stack_py.rl_modules.dqn.dqn import DQN
from navigation_stack_py.rl_modules.common.buffers import PrioritizedReplayBuffer
from navigation_stack_py.rl_modules.her.her_replay_buffer import HerReplayBuffer

from map_creator import generate_random_maps
from expert import ExpertActor
from utils import HyperParameters, make_proc_env

from omegaconf import OmegaConf, DictConfig
import hydra
import mlflow


class TensorboardCallback(BaseCallback):
    def __init__(self, eval_env: gym.Env, freq: int, verbose: int):
        super().__init__(verbose=verbose)
        self._eval_env = eval_env
        self._freq = freq
        self._Window = 100  # window size for moving average

        # length of episode
        self._num_episodes = 0
        self._ave_stuck_time_list = []
        self._max_stuck_time_list = []
        self._num_update_global_planner_list = []
        self._num_oscillation_list = []
        self._SGT_list = []  # Success weighted by normalized goal time
        self._SPL_list = []  # Success weighted by normalized path length

        # length = goal num
        self._goal_time_list = []
        self._ave_stuck_time_goal_list = []
        self._max_stuck_time_goal_list = []
        self._reach_max_time_list = []

        # length = collision num
        self._collision_list = []

    # This func is called every steps, but if not end of episode, return as soon as possible
    def record(self):
        # check end of episode
        dones = self.locals["dones"]

        # if no end of episode, return
        if not any(dones):
            return

        # get data from environment
        infos = self.locals["infos"]
        data_list = [info["data"] for info in infos]

        ave_stuck_list = [data["average_stuck_time"] for data in data_list]
        max_stuck_list = [data["max_stuck_time"] for data in data_list]
        num_update_global_planner_list = [
            data["num_update_global_planner"] for data in data_list
        ]
        num_oscillation_list = [data["num_oscillation"] for data in data_list]
        SGT_list = [data["SGT"] for data in data_list]
        SPL_list = [data["SPL"] for data in data_list]
        collision_list = [data["collision"] for data in data_list]
        goal_time_list = [data["goal_time"] for data in data_list]
        reach_max_time_list = [data["reach_max_time"] for data in data_list]
        for i, done in enumerate(dones):
            if done:
                self._num_episodes += 1
                self._ave_stuck_time_list.append(ave_stuck_list[i])
                self._max_stuck_time_list.append(max_stuck_list[i])
                self._num_update_global_planner_list.append(
                    num_update_global_planner_list[i]
                )
                self._num_oscillation_list.append(num_oscillation_list[i])
                self._SPL_list.append(SPL_list[i])
                self._SGT_list.append(SGT_list[i])
                self._reach_max_time_list.append(reach_max_time_list[i])
                if collision_list[i] == 1:
                    self._collision_list.append(1)
                if goal_time_list[i] != -1:
                    self._goal_time_list.append(goal_time_list[i])
                    self._ave_stuck_time_goal_list.append(ave_stuck_list[i])
                    self._max_stuck_time_goal_list.append(max_stuck_list[i])

    def set_tensorboard(self):
        if self._num_episodes == 0:
            pass
        else:
            window_size = min(self._Window, self._num_episodes)
            self.logger.record("metric/num_episodes", self._num_episodes)

            # average actual goal time in recent episodes
            if len(self._goal_time_list) != 0:
                ave_goal_time = np.mean(self._goal_time_list[-window_size:])
                self.logger.record("metric/average_goal_time", ave_goal_time)

            # SGT in recent episodes
            ave_SGT = np.mean(self._SGT_list[-window_size:])
            min_SGT = np.min(self._SGT_list[-window_size:])
            max_SGT = np.max(self._SGT_list[-window_size:])
            self.logger.record("metric/average_SGT", ave_SGT)
            self.logger.record("metric/min_SGT", min_SGT)
            self.logger.record("metric/max_SGT", max_SGT)

            # average SPL in recent episodes
            ave_SPL = np.mean(self._SPL_list[-window_size:])
            self.logger.record("metric/average_SPL", ave_SPL)

            if len(self._reach_max_time_list) != 0:
                reach_max_time = np.mean(self._reach_max_time_list[-window_size:])
                self.logger.record("metric/reach_max_time", reach_max_time)

            # average success rate in all episodes
            goal_num = len(self._goal_time_list)
            success_rate = float(goal_num) / float(self._num_episodes)
            self.logger.record("metric/success_rate", success_rate)

            # collision rate in all episodes
            collision_num = len(self._collision_list)
            collision_rate = float(collision_num) / float(self._num_episodes)
            self.logger.record("metric/collision_rate", collision_rate)

            # stuck time in recent episodes
            ave_stuck_time = np.mean(self._ave_stuck_time_list[-window_size:])
            max_stuck_time = np.mean(self._max_stuck_time_list[-window_size:])
            self.logger.record("metric/average_stuck_time", ave_stuck_time)
            self.logger.record("metric/max_stuck_time", max_stuck_time)

            # num of update global planner in recent episodes
            num_update_global_planner = np.mean(
                self._num_update_global_planner_list[-window_size:]
            )
            self.logger.record(
                "metric/num_update_global_planner", num_update_global_planner
            )

            # num of oscillation in recent episodes
            num_oscillation = np.mean(self._num_oscillation_list[-window_size:])
            self.logger.record("metric/num_oscillation", num_oscillation)

            # plot data with goal time score
            figure = plt.figure()
            ax = figure.add_subplot()
            ax.scatter(self._num_oscillation_list, self._SGT_list, label="average SGT")
            ax.scatter(self._num_oscillation_list, self._SPL_list, label="average SPL")
            figure.legend()
            plt.ylabel("SGT/SPL")
            plt.xlabel("Num of oscillation")

            # Close the figure after logging it
            self.logger.record(
                "figure",
                Figure(figure, close=True),
                exclude=("stdout", "log", "json", "csv"),
            )
            plt.close()

    def _on_step(self) -> bool:
        # Record variables at every step
        self.record()

        # Set log for tensorboard every check_freq steps
        if self.n_calls % self._freq == 0:
            self.set_tensorboard()

        return True


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}"
                    )

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}.zip")
                    self.model.save(self.save_path)

        return True


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


def train(params: HyperParameters):
    # This is faster than DummyVecEnv for the environment
    env = make_proc_env(params=params, rank=params.seed, num_env=params.num_processes)

    model = None

    policy_kwags = dict(net_arch=params.dqn_params["net_arch"])

    if params.dqn_params["learning_schedule"] == "linear":
        learning_rate = linear_schedule(params.dqn_params["learning_rate"])
    elif params.dqn_params["learning_schedule"] == "constant":
        learning_rate = params.dqn_params["learning_rate"]
    else:
        raise ValueError("Learning schedule not supported")

    if params.train_mode == "continue":
        raise NotImplementedError
        if params.rl_algorithm == "DQN":
            model = DQN.load(
                params.saved_model_path,
                verbose=params.verbose,
                env=env,
                tensorboard_log=params.tensor_board_path,
                policy_kwags=policy_kwags,
                gradient_steps=-1,
                learning_rate=learning_rate,
                buffer_size=int(params.dqn_params["buffer_size"]),
                learning_starts=int(params.dqn_params["learning_starts"]),
                batch_size=int(params.dqn_params["batch_size"]),
                train_freq=params.dqn_params["train_freq"],
                target_update_interval=int(params.dqn_params["target_update_interval"]),
            )
        elif params.rl_algorithm == "DQN-PRB":
            prioritized_replay_kwargs = dict(
                prioritized_error=params.prb_params["prioritized_error"],
                alpha=params.prb_params["alpha"],
                initial_beta=params.prb_params["initial_beta"],
                beta_increment=params.prb_params["beta_increment"],
            )
            model = DQN.load(
                env,
                policy_kwargs=policy_kwags,
                verbose=params.verbose,
                replay_buffer_class=PrioritizedReplayBuffer,
                prioritized_replay_kwargs=prioritized_replay_kwargs,
                tensorboard_log=params.tensor_board_path,
                gradient_steps=-1,
                learning_rate=learning_rate,
                buffer_size=int(params.dqn_params["buffer_size"]),
                learning_starts=int(params.dqn_params["learning_starts"]),
                batch_size=int(params.dqn_params["batch_size"]),
                train_freq=params.dqn_params["train_freq"],
                target_update_interval=int(params.dqn_params["target_update_interval"]),
            )
        else:
            raise ValueError(f"RL model {params.rl_algorithm} is not supported")
    elif params.train_mode == "new":
        # new training
        if params.rl_algorithm == "DQN":
            model = DQN(
                "MlpPolicy",
                env,
                policy_kwargs=policy_kwags,
                verbose=params.verbose,
                tensorboard_log=params.tensor_board_path,
                gradient_steps=-1,
                learning_rate=learning_rate,
                buffer_size=int(params.dqn_params["buffer_size"]),
                learning_starts=int(params.dqn_params["learning_starts"]),
                batch_size=int(params.dqn_params["batch_size"]),
                train_freq=params.dqn_params["train_freq"],
                target_update_interval=int(params.dqn_params["target_update_interval"]),
                seed=params.seed,
            )
        elif params.rl_algorithm == "DQN-PRB":
            prioritized_replay_kwargs = dict(
                prioritized_error=params.prb_params["prioritized_error"],
                alpha=params.prb_params["alpha"],
                initial_beta=params.prb_params["initial_beta"],
                beta_increment=params.prb_params["beta_increment"],
            )
            model = DQN(
                "MlpPolicy",
                env,
                policy_kwargs=policy_kwags,
                verbose=params.verbose,
                replay_buffer_class=PrioritizedReplayBuffer,
                prioritized_replay_kwargs=prioritized_replay_kwargs,
                tensorboard_log=params.tensor_board_path,
                gradient_steps=-1,
                learning_rate=learning_rate,
                buffer_size=int(params.dqn_params["buffer_size"]),
                learning_starts=int(params.dqn_params["learning_starts"]),
                batch_size=int(params.dqn_params["batch_size"]),
                train_freq=params.dqn_params["train_freq"],
                target_update_interval=int(params.dqn_params["target_update_interval"]),
                seed=params.seed,
            )
        else:
            raise ValueError(f"RL model {params.rl_algorithm} is not supported")
    else:
        raise ValueError(f"Train mode {params.train_mode} is not supported")

    # callbacks
    ## save model callback
    check_point_dir = os.path.join(params.model_dir, "checkpoints")
    checkpoint_callback = CheckpointCallback(
        save_freq=params.model_save_freq,
        save_path=check_point_dir,
        verbose=params.verbose,
    )
    ## tensorboard callback
    tensorboard_callback = TensorboardCallback(
        eval_env=env, freq=params.tensorboard_check_freq, verbose=params.verbose
    )

    callback_list = CallbackList([checkpoint_callback, tensorboard_callback])

    model.learn(total_timesteps=params.total_timesteps, callback=callback_list)

    model.save(params.saved_model_path)

    # save model to mlflow
    mlflow.log_artifact(params.saved_model_path)

    return model


@hydra.main(
    config_name="default.yaml", config_path="../config/train", version_base=None
)
def main(cfg: DictConfig):
    # mlflow setup
    mlflow.set_tracking_uri("file://" + hydra.utils.get_original_cwd() + "/mlruns")

    mlflow.set_experiment("train")

    # get params
    train_config = OmegaConf.to_container(cfg, resolve=True)
    params = HyperParameters(config=train_config, mode="train")

    # generate random maps
    if params.is_generate_random_map:
        generate_random_maps(params.map_config, params.random_map_num, params.seed)

    print("RL algorithm: ", params.rl_algorithm)
    print("Train mode: ", params.train_mode)
    print("Seed: ", params.seed)
    print("Total timesteps: ", params.total_timesteps)

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
        mlflow.log_params(train_config["train"])
        mlflow.log_params(train_config["navigation"])
        global_planner = train_config["navigation"]["common"]["global_planner"]
        local_planner = train_config["navigation"]["common"]["local_planner"]
        mlflow.log_param("global_planner", global_planner)
        mlflow.log_param("local_planner", local_planner)

        # train
        model = train(params)

        # evaluate
        eval_seed = 1234
        eval_env = make_proc_env(
            params=params, rank=eval_seed, num_env=params.num_processes
        )
        mean_reward, std_reward = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=params.num_test_episodes,
            render=False,
            deterministic=True,
            callback=None,
            reward_threshold=None,
            return_episode_rewards=False,
        )

        # log metrics
        mlflow.log_metric("mean_reward", mean_reward)
        mlflow.log_metric("std_reward", std_reward)

    return mean_reward


if __name__ == "__main__":
    main()
