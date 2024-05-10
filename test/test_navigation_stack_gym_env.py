import gym
from navigation_stack_py.gym_env import NavigationStackEnv

from gym import envs

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy


config_path = "config/navigation/square/random.yaml"
env = gym.make("NavigationStackEnv-v0", navigation_config=config_path)

print(env.action_space)
print(env.observation_space)

# Check the environment is valid
check_env(env)

# n_steps = 1000

# obs = env.reset()
# for _ in range(n_steps):
#     obs, reward, done, info = env.step(1)
#     env.render()
#     if done:
#         print("Goal reached!", "reward=", reward)
#         break
