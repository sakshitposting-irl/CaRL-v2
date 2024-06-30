'''
Racecar driver, Reinforcement Learning based car driving.
According to the documentation, the problem is solved on a cumulative_reward of >= 900
'''

# Import all dependencies
import gym
from stable_baselines3 import PPO 
from stable_baselines3.common.vec_env import DummyVecEnv 
from stable_baselines3.common.evaluation import evaluate_policy 
import os
# pip install box2d-py 
from gym.envs import box2d

# env created
env_name = "CarRacing-v2"
env = gym.make(env_name)

env = DummyVecEnv([lambda: env])
model = PPO("CnnPolicy", env, verbose=1)

# Train the agent
model.learn(total_timesteps=100000, progress_bar=True)

# Evaluate the trained model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

# Test the trained model
env = gym.make(env_name, render_mode="human")  # Enable rendering for testing
env = DummyVecEnv([lambda: env])

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
