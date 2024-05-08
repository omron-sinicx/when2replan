import os
import gym
import numpy as np
import time
import random
import argparse
import yaml
from typing import Callable, Any, Dict, Tuple
import tensorboard
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

from navigation_stack_py.gym_env import NavigationStackEnv
from gym import envs
import torch as th
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
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video
from imitation.algorithms import bc
from imitation.data.types import Trajectory
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
import pickle

from navigation_stack_py.rl_modules.dqn.dqn import DQN
from navigation_stack_py.rl_modules.common.buffers import PrioritizedReplayBuffer
from navigation_stack_py.rl_modules.her.her_replay_buffer import HerReplayBuffer

from map_creator import generate_random_maps
from expert import ExpertActor

from omegaconf import OmegaConf, DictConfig
import hydra


class TrainParameters:
    navigation_config_path: str = None
    process_type: str = None
    train_mode: str = None
    total_timesteps: int = None
    env_id: str = None
    num_cpu: int = None
    log_dir: str = None
    saved_model_path: str = None
    tensor_board_path: str = None
    check_freq: int = None
    verbose: int = None
    seed: int
    net_arch: dict = None
    is_generate_random_map: bool = None
    random_map_num: int = None
    map_config: str = None
    rl_model: str = None
    rl_config: str = None
    ppo_params: None
    dqn_params = None
    her_params = None
    prb_params = None
    imitation_config: None
    expert_traj_path: str = None
    bc_policy_path: str = None

    def __init__(self, train_config: dict):
        self.navigation_config_path = train_config["navigation_config"]
        self.rl_model = train_config["rl_model"]
        self.process_type = train_config["process_type"]
        self.train_mode = train_config["train_mode"]
        # load as int
        self.total_timesteps = train_config["total_timesteps"]
        self.env_id = train_config["env_id"]
        self.num_cpu = train_config["num_cpu"]
        current_dir = os.path.dirname(os.path.realpath(__file__))
        project_dir = os.path.join(current_dir, "..")
        print(project_dir)
        root_log_dir = os.path.join(project_dir, train_config["log_dir"])
        self.train_log_dir = os.path.join(root_log_dir, "train")
        self.model_dir = os.path.join(root_log_dir, "model")
        os.makedirs(self.train_log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        self.saved_model_path = os.path.join(self.model_dir, train_config["model_name"])
        self.tensor_board_path = os.path.join(
            self.train_log_dir, train_config["tensor_board_path"]
        )
        self.model_save_freq = train_config["model_save_freq"]
        self.tensorboard_check_freq = train_config["tensorboard_check_freq"]
        self.verbose = train_config["verbose"]
        self.is_generate_random_map = train_config["is_generate_random_map"]
        self.seed = train_config["seed"]
        self.net_arch = train_config["net_arch"]
        self.random_map_num = train_config["random_map_num"]
        self.map_config = os.path.join(project_dir, train_config["map_config"])
        self.rl_config = os.path.join(project_dir, train_config["rl_config"])
        self.ppo_params = train_config["ppo_params"]
        self.dqn_params = train_config["dqn_params"]
        self.her_params = train_config["her_params"]
        self.prb_params = train_config["prb_params"]
        self.imitation_config = train_config["imitation_config"]
        self.expert_traj_path = os.path.join(
            self.train_log_dir, self.imitation_config["trajectory_path"]
        )
        self.bc_policy_path = os.path.join(
            self.model_dir, self.imitation_config["bc_policy_path"]
        )


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
        # on_step() is not called when episode is done
        # So, I extend the length of episode by 1
        # is_end() is called when episode is done + 1 step
        end_list = self.training_env.env_method("is_end")

        # if no end of episode, return
        if not any(end_list):
            return

        # get data from environment
        data_list = self.training_env.env_method("copy_data")

        # record value when done
        done_list = [data["done"] for data in data_list]

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
        for i, done in enumerate(done_list):
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


class HParamCallback(BaseCallback):
    def __init__(self):
        """
        Saves the hyperparameters and metrics at the start of the training, and logs them to TensorBoard.
        """
        super().__init__()

    def _on_training_start(self) -> None:
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "learning rate": self.model.learning_rate,
            "gamma": self.model.gamma,
            "learning_starts": self.model.learning_starts,
            "target_update_interval": self.model.target_update_interval,
            "batch_size": self.model.batch_size,
        }
        # define the metrics that will appear in the `HPARAMS` Tensorboard tab by referencing their tag
        # Tensorbaord will find & display metrics from the `SCALARS` tab
        metric_dict = {"rollout/ep_len_mean": 0, "train/value_loss": 0}
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
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


# NOTE: It is too slow and not stable now
class VideoRecorderCallback(BaseCallback):
    def __init__(
        self,
        eval_env: gym.Env,
        render_freq: int,
        n_eval_episodes: int = 1,
        deterministic: bool = True,
    ):
        """
        Records a video of an agent's trajectory traversing ``eval_env`` and logs it to TensorBoard

        :param eval_env: A gym environment from which the trajectory is recorded
        :param render_freq: Render the agent's trajectory every eval_freq call of the callback.
        :param n_eval_episodes: Number of episodes to render
        :param deterministic: Whether to use deterministic or stochastic policy
        """
        super().__init__()
        self._eval_env = eval_env
        self._render_freq = render_freq
        self._n_eval_episodes = n_eval_episodes
        self._deterministic = deterministic

    def _on_step(self) -> bool:
        if self.n_calls % self._render_freq == 0:
            screens = []

            def grab_screens(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
                """
                Renders the environment in its current state, recording the screen in the captured `screens` list

                :param _locals: A dictionary containing all local variables of the callback's scope
                :param _globals: A dictionary containing all global variables of the callback's scope
                """
                screen = self._eval_env.render(mode="rgb_array")
                # PyTorch uses CxHxW vs HxWxC gym (and tensorflow) image convention
                screens.append(screen.transpose(2, 0, 1))

            evaluate_policy(
                self.model,
                self._eval_env,
                callback=grab_screens,
                n_eval_episodes=self._n_eval_episodes,
                deterministic=self._deterministic,
            )
            self.logger.record(
                "trajectory/video",
                Video(th.ByteTensor([screens]), fps=10),
                exclude=("stdout", "log", "json", "csv"),
            )
        return True


def make_env(params: TrainParameters, rank: int, seed: int = 0) -> Callable:
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """

    def _init() -> gym.Env:
        env = gym.make(
            params.env_id,
            navigation_config=params.navigation_config_path,
            seed=seed + rank,
            rl_config=params.rl_config,
        )
        env.seed(seed + rank)
        env.action_space.seed(seed + rank)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = Monitor(env, params.train_log_dir, allow_early_resets=True)
        return env

    set_random_seed(seed)
    return _init


def train_single_proc(params: TrainParameters):
    env = gym.make(
        params.env_id,
        navigation_config=params.navigation_config_path,
        seed=params.seed,
        rl_config=params.rl_config,
    )
    env = Monitor(env, params.train_log_dir, allow_early_resets=True)

    # Create action noise because TD3 and DDPG use a deterministic policy
    # n_actions = env.action_space.shape[-1]
    # action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    # model = TD3('MlpPolicy', env, action_noise=action_noise, verbose=0)

    model = None
    if params.train_mode == "continue":
        # Continue training from the last best model
        if params.rl_model == "DQN":
            model = DQN.load(
                params.saved_model_path,
                verbose=params.verbose,
                env=env,
                tensorboard_log=params.tensor_board_path,
            )
        elif params.rl_model == "PPO":
            model = PPO.load(
                params.saved_model_path,
                verbose=params.verbose,
                env=env,
                tensorboard_log=params.tensor_board_path,
            )
        else:
            raise ValueError(f"RL model {params.rl_model} is not supported")
    elif params.train_mode == "new":
        # New training
        if params.rl_model == "DQN":
            model = DQN(
                "MlpPolicy",
                env,
                verbose=params.verbose,
                tensorboard_log=params.tensor_board_path,
            )
        elif params.rl_model == "PPO":
            model = PPO(
                "MlpPolicy",
                env,
                verbose=params.verbose,
                tensorboard_log=params.tensor_board_path,
                seed=params.seed,
            )
        else:
            raise ValueError(f"RL model {params.rl_model} is not supported")
    elif params.train_mode == "imitation":
        # load pre-trained model by imitation learning
        if params.rl_model != "PPO":
            raise ValueError(
                f"RL model {params.rl_model} is supported for imitation learning. PPO is only supported."
            )
        bc_policy = load_bc_model(params)
        model = PPO(
            bc_policy,
            verbose=params.verbose,
            env=env,
            tensorboard_log=params.tensor_board_path,
        )
    else:
        raise ValueError(f"Train mode {params.train_mode} is not supported")

    checkpoint_callback = SaveOnBestTrainingRewardCallback(
        check_freq=params.model_save_freq,
        log_dir=params.train_log_dir,
        verbose=params.verbose,
    )
    tensorboard_callback = TensorboardCallback(
        eval_env=env, freq=params.tensorboard_check_freq, verbose=params.verbose
    )
    # _evaluate_env = gym.make(params.env_id, navigation_config = params.navigation_config_path,
    #                         seed=params.seed+10000, rl_config = params.rl_config,
    #                         visualize_mode = "birdeye", is_training = False)
    # evaluate_env = Monitor(_evaluate_env, params.train_log_dir, allow_early_resets=True)
    # video_callback = VideoRecorderCallback(evaluate_env, render_freq=100)

    callback_list = CallbackList([checkpoint_callback, tensorboard_callback])

    model.learn(total_timesteps=params.total_timesteps, callback=callback_list)

    model.save(params.saved_model_path)
    del model  # remove to demonstrate saving and loading


def train_parallel_proc(params: TrainParameters):
    # This is faster than DummyVecEnv for the environment
    env = SubprocVecEnv(
        [
            make_env(params=params, rank=i, seed=params.seed)
            for i in range(params.num_cpu)
        ]
    )

    model = None

    policy_kwags = dict(net_arch=params.net_arch)

    if params.train_mode == "continue":
        # continue training from the last best model
        if params.rl_model == "DQN":
            model = DQN.load(
                params.saved_model_path,
                verbose=params.verbose,
                env=env,
                tensorboard_log=params.tensor_board_path,
            )
        elif params.rl_model == "PPO":
            model = PPO.load(
                params.saved_model_path,
                verbose=params.verbose,
                env=env,
                tensorboard_log=params.tensor_board_path,
            )
        elif params.rl_model == "A2C":
            model = A2C.load(
                params.saved_model_path,
                verbose=params.verbose,
                env=env,
                tensorboard_log=params.tensor_board_path,
            )
        else:
            raise ValueError(f"RL model {params.rl_model} is not supported")
    elif params.train_mode == "new":
        # new training
        if params.rl_model == "DQN":
            model = DQN(
                "MlpPolicy",
                env,
                policy_kwargs=policy_kwags,
                verbose=params.verbose,
                tensorboard_log=params.tensor_board_path,
                gradient_steps=-1,
                learning_rate=params.dqn_params["learning_rate"],
                buffer_size=int(params.dqn_params["buffer_size"]),
                learning_starts=int(params.dqn_params["learning_starts"]),
                batch_size=int(params.dqn_params["batch_size"]),
                train_freq=params.dqn_params["train_freq"],
                target_update_interval=int(params.dqn_params["target_update_interval"]),
            )
        elif params.rl_model == "DQN-HER":
            her_kwargs = dict(
                n_sampled_goal=params.her_params["n_sampled_goal"],
                goal_selection_strategy=params.her_params["goal_selection_strategy"],
                online_sampling=True,
            )

            model = DQN(
                "MultiInputPolicy",
                env,
                policy_kwargs=policy_kwags,
                verbose=params.verbose,
                replay_buffer_class=HerReplayBuffer,
                replay_buffer_kwargs=her_kwargs,
                tensorboard_log=params.tensor_board_path,
                gradient_steps=-1,
                learning_rate=params.dqn_params["learning_rate"],
                buffer_size=int(params.dqn_params["buffer_size"]),
                learning_starts=int(params.dqn_params["learning_starts"]),
                batch_size=int(params.dqn_params["batch_size"]),
                train_freq=params.dqn_params["train_freq"],
                target_update_interval=int(params.dqn_params["target_update_interval"]),
            )
        elif params.rl_model == "DQN-PRB":
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
                learning_rate=params.dqn_params["learning_rate"],
                buffer_size=int(params.dqn_params["buffer_size"]),
                learning_starts=int(params.dqn_params["learning_starts"]),
                batch_size=int(params.dqn_params["batch_size"]),
                train_freq=params.dqn_params["train_freq"],
                target_update_interval=int(params.dqn_params["target_update_interval"]),
            )
        elif params.rl_model == "PPO":
            model = PPO(
                "MlpPolicy",
                env,
                verbose=params.verbose,
                tensorboard_log=params.tensor_board_path,
                seed=params.seed,
                learning_rate=params.ppo_params["learning_rate"],
                n_steps=params.ppo_params["n_steps"],
                batch_size=params.ppo_params["batch_size"],
                n_epochs=params.ppo_params["n_epochs"],
                gamma=params.ppo_params["gamma"],
                gae_lambda=params.ppo_params["gae_lambda"],
                clip_range=params.ppo_params["clip_range"],
                normalize_advantage=params.ppo_params["normalize_advantage"],
                ent_coef=params.ppo_params["ent_coef"],
                vf_coef=params.ppo_params["vf_coef"],
                max_grad_norm=params.ppo_params["max_grad_norm"],
                use_sde=params.ppo_params["use_sde"],
                sde_sample_freq=params.ppo_params["sde_sample_freq"],
            )
        elif params.rl_model == "A2C":
            model = A2C(
                "MlpPolicy",
                env,
                verbose=params.verbose,
                tensorboard_log=params.tensor_board_path,
            )
        else:
            raise ValueError(f"RL model {params.rl_model} is not supported")
    elif params.train_mode == "imitation":
        # load pre-trained model by imitation learning
        if params.rl_model != "PPO":
            raise ValueError(
                f"RL model {params.rl_model} is supported for imitation learning. PPO is only supported."
            )
        bc_policy = load_bc_model(params)
        model = PPO(
            bc_policy,
            verbose=params.verbose,
            env=env,
            tensorboard_log=params.tensor_board_path,
        )
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
    hyperparams_callback = HParamCallback()

    callback_list = CallbackList(
        [checkpoint_callback, tensorboard_callback, hyperparams_callback]
    )

    model.learn(total_timesteps=params.total_timesteps, callback=callback_list)

    model.save(params.saved_model_path)
    del model  # remove to demonstrate saving and loading


def expert_trajectory(params: TrainParameters, seed: int) -> Tuple[Trajectory, int]:
    env = gym.make(
        params.env_id,
        navigation_config=params.navigation_config_path,
        seed=seed,
        rl_config=params.rl_config,
    )
    env.seed(params.seed)
    expert = ExpertActor(seed=seed, max_hz=env.get_max_global_hz())
    obs = env.reset()
    observations = [obs]
    actions = []
    infos = []
    done = False
    sum_reward = 0
    while not done:
        action = expert.act(obs)
        obs, reward, done, info = env.step(action)
        sum_reward += reward
        actions.append(action)
        observations.append(obs)
        infos.append(info)
        if done:
            ts = Trajectory(
                obs=np.array(observations),
                acts=np.array(actions),
                infos=np.array(infos),
                terminal=True,
            )
            break
    print("seed: {}, reward: {}".format(seed, sum_reward))
    return ts


def expert_trajectory_collection(params: TrainParameters) -> None:
    # Expert Trajectory Collection
    num_record_episode = params.imitation_config["num_episodes"]
    trajectories = []
    if params.process_type == "Single":
        pass
        for i in range(num_record_episode):
            ts = expert_trajectory(params, i)
            trajectories.append(ts)
    elif params.process_type == "Parallel":
        _ts_list = []
        with ProcessPoolExecutor(max_workers=params.num_cpu) as executor:
            for i in range(num_record_episode):
                _ts = executor.submit(expert_trajectory, params, i)
                _ts_list.append(_ts)
        for i in range(num_record_episode):
            trajectories.append(_ts_list[i].result())
    else:
        raise ValueError(f"Process type {params.process_type} is not supported")

    # save trajectories
    with open(params.expert_traj_path, "wb") as f:
        pickle.dump(trajectories, f)


def imitation_learning(params: TrainParameters) -> None:
    # load trajectories
    with open(params.expert_traj_path, "rb") as f:
        expert_trajectories = pickle.load(f)
    transitions = rollout.flatten_trajectories(expert_trajectories)
    env = gym.make(
        params.env_id,
        navigation_config=params.navigation_config_path,
        seed=params.seed,
        rl_config=params.rl_config,
    )
    env = Monitor(env, params.train_log_dir, allow_early_resets=True)
    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
    )
    # before evaluation
    print("evaluate before BC training")
    before_bc_reward, _ = evaluate_policy(
        bc_trainer.policy,
        env,
        n_eval_episodes=params.imitation_config["num_expert_eval_episodes"],
        deterministic=True,
        render=False,
    )

    # Train a policy with BC
    bc_trainer.train(
        n_epochs=params.imitation_config["num_train_epochs"], progress_bar=True
    )

    # After evaluation
    print("evaluate after BC training")
    after_bc_reward, _ = evaluate_policy(
        bc_trainer.policy,
        env,
        n_eval_episodes=params.imitation_config["num_expert_eval_episodes"],
        deterministic=True,
        render=False,
    )

    # IL result
    print(f"Before training, reward is {before_bc_reward}")
    print(f"After training, reward is {after_bc_reward}")

    # save bc policy
    bc_trainer.save_policy(params.bc_policy_path)


def load_bc_model(params: TrainParameters) -> ActorCriticPolicy:
    bc_trainer = bc.reconstruct_policy(params.bc_policy_path)

    # save policy as PPO model
    class CopyPolicy(ActorCriticPolicy):
        def __new__(cls, *args, **kwargs):
            return bc_trainer

    return CopyPolicy


def pre_train(params: TrainParameters) -> None:
    # Expert Trajectory Collection
    if params.imitation_config["is_collect_expert_data"]:
        print("Collecting expert trajectories")
        expert_trajectory_collection(params)
    else:
        if not os.path.exists(params.expert_traj_path):
            raise ValueError(
                f"Expert trajectory path {params.expert_traj_path} does not exist."
            )
        else:
            print(f"Expert trajectory {params.expert_traj_path} is used.")

    # Imitation learning with Behavior Cloning
    if params.imitation_config["is_pre_train"]:
        print("Imitation learning with Behavior Cloning")
        imitation_learning(params)
    else:
        if not os.path.exists(params.bc_policy_path):
            raise ValueError(
                f"Imitation model path {params.bc_policy_path} does not exist."
            )
        else:
            print(f"Imitation model {params.bc_policy_path} is used.")


@hydra.main(config_name="train/default", config_path="../config", version_base=None)
def main(cfg: DictConfig):
    train_config = OmegaConf.to_container(cfg.train, resolve=True)

    params = TrainParameters(train_config=train_config)

    # generate random maps
    if params.is_generate_random_map:
        generate_random_maps(params.map_config, params.random_map_num, params.seed)

    # Pre training with imitation learning
    if params.train_mode == "imitation":
        print("Pre-training with imitation learning")
        pre_train(params)

    print("RL algorithm: ", params.rl_model)
    print("Train mode: ", params.train_mode)
    print("Navigation scenario: ", params.navigation_config_path)
    print("Seed: ", params.seed)
    print("Total timesteps: ", params.total_timesteps)
    if params.process_type == "Single":
        train_single_proc(params)
    elif params.process_type == "Parallel":
        train_parallel_proc(params)
    else:
        raise ValueError("Unknown train mode")


if __name__ == "__main__":
    main()
