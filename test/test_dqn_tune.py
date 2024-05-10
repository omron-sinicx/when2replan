"""
Part of this script is copied from https://github.com/DLR-RM/stable-baselines3
"""

import gym

import tensorboard

from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from navigation_stack_py.rl_modules.dqn.dqn import DQN
from navigation_stack_py.rl_modules.common.buffers import (
    ReplayBuffer,
    PrioritizedReplayBuffer,
)

from omegaconf import OmegaConf, DictConfig
import hydra
import mlflow


@hydra.main(config_name="config.yaml", config_path=".", version_base=None)
def main(cfg: DictConfig) -> None:
    # replay_buffer_cls = ReplayBuffer
    replay_buffer_cls = PrioritizedReplayBuffer
    prioritized_error = cfg.priority.prioritized_error
    alpha = cfg.priority.alpha
    beta = cfg.priority.beta
    beta_increment = cfg.priority.beta_increment

    env_id = "CartPole-v1"
    num_cpu = cfg.common.num_cpu
    total_time_steps = cfg.common.total_time_steps
    learning_rate = cfg.common.learning_rate
    batch_size = cfg.common.batch_size
    buffer_size = cfg.common.buffer_size
    learning_starts = cfg.common.learning_starts
    gamma = cfg.common.gamma
    target_update_interval = cfg.common.target_update_interval
    train_freq = cfg.common.train_freq
    gradient_steps = cfg.common.gradient_steps
    exploration_fraction = cfg.common.exploration_fraction
    exploration_final_eps = cfg.common.exploration_final_eps
    policy_kwargs = dict(net_arch=cfg.common.net_arch)

    # set mlflow
    mlflow.set_tracking_uri("file://" + hydra.utils.get_original_cwd() + "/mlruns")

    experiment_name = cfg.name.setting_name
    mlflow.set_experiment(experiment_name)

    run_name = cfg.name.run_name
    with mlflow.start_run(run_name=run_name):
        # Parameters logging from cfg
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))

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
            gradient_steps=gradient_steps,
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

        model.learn(total_timesteps=total_time_steps)

        # evaluate policy
        eval_env = make_vec_env(
            env_id, n_envs=num_cpu, seed=10, vec_env_cls=SubprocVecEnv
        )
        mean_reward, std_reward = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=10,
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
