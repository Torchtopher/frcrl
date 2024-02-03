import gymnasium as gym
import numpy as np

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from envs.frc_multiagent import FRC, Robot

import yaml

from stable_baselines3 import DQN

def mask_fn(env: FRC) -> np.ndarray:
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.
    #print(env._actions_taken)
    #print(env._action_mask())
    #print(env._get_obs())
    return env.action_mask()


with open("robots.yaml", "r") as file:
    loadedYaml = yaml.load(file, Loader=yaml.FullLoader)
robots = []
for idx, i in enumerate(loadedYaml["robots"]):
    robots.append(Robot(idx, i["amp_cycle_time"], i["speaker_cycle_time"], i["can_score_amp"], i["can_score_speaker"]))
    print(i)
print(list(map(lambda r: r.CAN_SCORE_SPEAKER, robots)))
env = FRC(robots=robots)

# vectorized environments allow for batched actions
env = ActionMasker(env, mask_fn)  # Wrap to enable masking

# MaskablePPO behaves the same as SB3's PPO unless the env is wrapped
# with ActionMasker. If the wrapper is detected, the masks are automatically
# retrieved and used when learning. Note that MaskablePPO does not accept
# a new action_mask_fn kwarg, as it did in an earlier draft.
# faster on cpu lmao
# (probably due to gpu copy times)
model = MaskablePPO("MultiInputPolicy", env, verbose=1, device="cpu")
# we want to overfit because we are only optimzing for one specific environment
model.learn(total_timesteps=150_000)
print("Done training")
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(120 * len(env.robots)):
    action, _state = model.predict(obs, deterministic=True, action_masks=mask_fn(env))
    obs, reward, done, info = vec_env.step(action)

# save the model
model.save("3_robot_ppo_frc_150000")

print(info)
a = info[0]["actions_taken"]
output = ""
for t,b in enumerate(list(zip(*a.values()))):
    output += f"{t}: {b}\n"
print(output)
with open("results.txt", "w") as file:
    file.write(output)