import gym

import tensorboard

from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from navigation_stack_py.rl_modules.common.env.bit_flipping_goal_env import (
    BitFlippingGoalEnv,
)
from navigation_stack_py.rl_modules.dqn.dqn import DQN
from navigation_stack_py.rl_modules.her.her_replay_buffer import HerReplayBuffer
from navigation_stack_py.rl_modules.her.custom_her_replay_buffer import (
    CustomHerReplayBuffer,
)

# settings
seed = 0
n_envs = 2
replay_buffer_cls = CustomHerReplayBuffer  # HerReplayBuffer or CustomHerReplayBuffer

# Bit-Flipping hyper params
N_BITS = 6
# https://github.com/DLR-RM/stable-baselines3/pull/704#issuecomment-1015497249
target_update_interval = 500
exploration_final_eps = 0.02
n_sampled_goal = 2
learning_rate = 5e-4
train_freq = 1
# gradient_steps = 1
gradient_steps = -1  # NOTE: should be -1 if use multiple envs
# https://github.com/DLR-RM/stable-baselines3/pull/439#issuecomment-961796799
learning_starts = 500
batch_size = 32
buffer_size = 10000

# total_time_steps = 10000
total_time_steps = 3000


# Available strategies (cf paper): future, final, episode
goal_selection_strategy = "final"  # equivalent to GoalSelectionStrategy.FUTURE

# NOTE: offline sampling is not supported on multiple envs
online_sampling = True
# Time limit for the episodes
max_episode_length = 4


def env_fn():
    return BitFlippingGoalEnv(
        n_bits=N_BITS, continuous=False, max_steps=max_episode_length
    )


def main():
    env = make_vec_env(env_fn, n_envs, vec_env_cls=SubprocVecEnv, seed=seed)

    model = DQN(
        "MultiInputPolicy",
        env,
        replay_buffer_class=replay_buffer_cls,
        # Parameters for HER
        replay_buffer_kwargs=dict(
            n_sampled_goal=n_sampled_goal,
            goal_selection_strategy=goal_selection_strategy,
            online_sampling=online_sampling,
        ),
        verbose=1,
        tensorboard_log="./dqn_her_bit_flipping_tensorboard/",
        target_update_interval=target_update_interval,
        exploration_final_eps=exploration_final_eps,
        learning_rate=learning_rate,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        learning_starts=learning_starts,
        batch_size=batch_size,
        buffer_size=buffer_size,
    )

    model.learn(total_timesteps=total_time_steps)


if __name__ == "__main__":
    main()
