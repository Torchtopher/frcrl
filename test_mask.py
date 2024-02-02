import gymnasium as gym
import numpy as np

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from frc import FRC

from stable_baselines3 import DQN

def mask_fn(env: FRC) -> np.ndarray:
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.
    #print(env._actions_taken)
    #print(env._action_mask())
    #print(env._get_obs())
    return env.action_mask()


env = FRC()
# vectorized environments allow for batched actions
env = ActionMasker(env, mask_fn)  # Wrap to enable masking

# MaskablePPO behaves the same as SB3's PPO unless the env is wrapped
# with ActionMasker. If the wrapper is detected, the masks are automatically
# retrieved and used when learning. Note that MaskablePPO does not accept
# a new action_mask_fn kwarg, as it did in an earlier draft.
model = MaskablePPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=10000000)
print("Done training")
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(120):
    action, _state = model.predict(obs, deterministic=True, action_masks=mask_fn(env))
    obs, reward, done, info = vec_env.step(action)
print(info)

# save the model
model.save("ppo_frc_100000")
# cycle*6 + score_amp + cycle*6 + score_amp 