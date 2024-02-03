import gymnasium as gym
import numpy as np

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from envs.frc_multiagent import FRC, Robot

import yaml

from stable_baselines3 import DQN

def mask_fn(env: FRC) -> np.ndarray:
    return env.action_mask()

# Load robots from yaml file
with open("robots.yaml", "r") as file:
    loadedYaml = yaml.load(file, Loader=yaml.FullLoader)
robots = []
for idx, i in enumerate(loadedYaml["robots"]):
    robots.append(Robot(idx, i["amp_cycle_time"], i["speaker_cycle_time"], i["can_score_amp"], i["can_score_speaker"]))
    print(i)
print(list(map(lambda r: r.CAN_SCORE_SPEAKER, robots)))
env = FRC(robots=robots)

# load the model
model = MaskablePPO.load("3_robot_ppo_frc_150000.zip", env=env, verbose=1, device="cpu")
# run a test
max_reward = 0 
actions_taken = []
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(20):
    total_reward = 0
    for i in range(360):
        action, _state = model.predict(obs, deterministic=True, action_masks=mask_fn(env))
        obs, reward, done, info = vec_env.step(action)
        total_reward += reward
    if total_reward > max_reward:
        max_reward = total_reward
        print(f"New max reward: {max_reward}")
        actions_taken = info[0]["actions_taken"]
    print(".", end="", flush=True)

a = actions_taken
output = ""
for t,b in enumerate(list(zip(*a.values()))):
    output += f"{t}: {b}\n"
print(output)
with open("results.txt", "w") as file:
    file.write(output)
