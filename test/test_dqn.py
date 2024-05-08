import gym

import tensorboard
from typing import Callable

from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from navigation_stack_py.rl_modules.dqn.dqn import DQN
from navigation_stack_py.rl_modules.common.buffers import (
    ReplayBuffer,
    PrioritizedReplayBuffer,
)

import mlflow


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


# replay_buffer_cls = ReplayBuffer
replay_buffer_cls = PrioritizedReplayBuffer
prioritized_error = "q_error"  # "q_error" or "td_error"
alpha = 0.6
beta = 0.4
beta_increment = 0.00001

env_id = "CartPole-v1"
num_cpu = 2  # Number of processes to use
total_time_steps = 50000
learning_rate = linear_schedule(2e-3)
batch_size = 64
buffer_size = 100000
learning_starts = 1000
gamma = 0.99
target_update_interval = 10
train_freq = 256
gradient_steps = 128
exploration_fraction = 0.16
exploration_final_eps = 0.04
policy_kwargs = dict(net_arch=[256, 256])


def main():

    mlflow.set_experiment("test")

    # Create the vectorized environment
    env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv)
    set_random_seed(0)

    prioritized_replay_kwargs = dict(
        prioritized_error=prioritized_error,
        alpha=alpha,
        beta=beta,
        beta_increment=beta_increment,
    )
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        replay_buffer_class=replay_buffer_cls,
        replay_buffer_kwargs=None,
        prioritized_replay_kwargs=prioritized_replay_kwargs,
        tensorboard_log="./dqn_cartpole_tensorboard/",
        gradient_steps=-1,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        target_update_interval=target_update_interval,
        exploration_final_eps=exploration_final_eps,
        batch_size=batch_size,
        gamma=gamma,
        train_freq=train_freq,
        exploration_fraction=exploration_fraction,
        policy_kwargs=policy_kwargs,
        learning_starts=learning_starts,
    )

    with mlflow.start_run():

        model.learn(total_timesteps=total_time_steps)

        model_path = "./test_model.zip"
        model.save(model_path)

        # save model as artifact to mlflow
        mlflow.log_artifact(model_path)


if __name__ == "__main__":
    main()
