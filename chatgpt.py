import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common import make_vec_env
import numpy as np

class CustomEnv(gym.Env):
    def __init__(self, num_agents):
        super(CustomEnv, self).__init__()
        self.num_agents = num_agents
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(num_agents,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)  # Example action space, modify as needed
        self.shared_reward = 0.0  # Initialize shared reward

    def reset(self):
        # Reset the environment and return initial observations for each agent
        self.shared_reward = 0.0
        return np.zeros(self.num_agents)

    def step(self, actions):
        # Take a step in the environment given actions from each agent
        # Update the state and calculate the reward
        # Modify the following lines to implement your own environment dynamics and reward structure
        rewards = [1.0 if action == 1 else 0.0 for action in actions]
        self.shared_reward += sum(rewards)
        observations = np.zeros(self.num_agents)
        done = False  # Modify based on your termination conditions
        info = {"shared_reward": self.shared_reward}
        return observations, sum(rewards), done, info

# Number of agents
num_agents = 3

# Create a function to make the environments
def make_env(env_class, env_args):
    return lambda: env_class(**env_args)

# Create a list of environments
env_args = {"num_agents": num_agents}
envs = [make_env(CustomEnv, env_args) for _ in range(num_agents)]

# Use DummyVecEnv for a single process or SubprocVecEnv for multiple processes
# env = DummyVecEnv(envs)
env = SubprocVecEnv(envs)

# Create the PPO model
model = PPO("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=10000)

# Save the trained model
model.save("multi_agent_model")

# Optionally, you can load the model later
# model = PPO.load("multi_agent_model")

# You can now use the trained model to make predictions or continue t