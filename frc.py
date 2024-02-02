
from typing import List, Tuple, Union
import gymnasium as gym
import numpy as np
# spaces
from gymnasium import spaces


class FRC(gym.Env):
    """
    FRC2024 environment involve a grid world, in which multiple robots attempt to maximize points

    Robot agenst select one of five actions ∈ {}.
    Each agent's observation includes its:
        - agent ID (1)
        - position with in grid (2)
        - number of steps since beginning (1)
        - number of agents and tree strength for each cell in agent view (2 * `np.prod(tuple(2 * v + 1 for v in agent_view))`).
    All values are scaled down into range ∈ [0, 1].

    Only the agents who are involved in cutting down the tree are rewarded with `tree_cutdown_reward`.
    The environment is terminated as soon as all trees are cut down or when the number of steps reach the `max_steps`.

    Upon rendering, we show the grid, where each cell shows the agents (blue) and tree (green) with their current strength.

    Args:
        n_agents: number of frc
        full_observable: flag whether agents should receive observation for all other agents
        step_cost: reward receive in each time step
        tree_cutdown_reward: reward received by agents who cut down the tree
        max_steps: maximum steps (1sec interval) before the environment is terminated (game is over)

    Attributes:
        _agents: list of all agents. The index in this list is also the ID of the agent
        _agent_map: tree dimensional numpy array of indicators where the agents are located
        _tree_map: two dimensional numpy array of strength of the trees
        _total_episode_reward: array with accumulated rewards for each agent.
        _agent_dones: list with indicater whether the agent is done or not.
        _base_img: base image with grid
        _viewer: viewer for the rendered image
    """
    metadata = {'render.modes': ['text']}

    def __init__(self,
                 step_cost: float = -0.1, amp_reward: float = 1.0, speaker_reward: float = 2.0,
                 amped_speaker_reward: float = 5.0, max_steps: int = 120):

        self.AMP_CYCLE_T = 6
        #self.AMP_ACC = AMP_ACC 
        self.SPEAKER_CYCLE_T = 6 
        #self.SPEAKER_ACC = SPEAKER_ACC 
        self.CAN_SCORE_AMP = True 
        self.CAN_SCORE_SPEAKER = True
        self.time_spent_cycling = 0
        self.amp_time_left = 0
        self.notes_in_amp = 0

        self._step_cost = step_cost
        self._step_count = None
        self._amp_reward = amp_reward
        self._speaker_reward = speaker_reward
        self._amped_speaker_reward = amped_speaker_reward
        self._max_steps = max_steps
        self._actions_taken = []
        self._total_episode_reward = None

        # action space is, [wait,  move_cycle, score_amp, score_speaker, activate_amp]
        self.action_space = spaces.Discrete(5)
        self._action_ids = {
            'wait': 0,
            'cycle': 1,
            'score_amp': 2,
            'score_speaker': 3,
            'activate_amp': 4,
        }
        # invert the dictionary
        self._action_names = {v: k for k, v in self._action_ids.items()}

        # obs space is [time_left, total_cycle_t, speaker_cycle_t, amp_cycle_t, can_score_amp, can_score_speaker, amp_time_left, notes_in_amp]
        self.observation_space = spaces.Tuple((
                                            spaces.Box(low=0, high=max_steps+1, shape=(1,), dtype=np.int32), # time_left
                                            spaces.Box(low=0, high=max_steps, shape=(1,), dtype=np.int32), # total_cycle_t
                                            spaces.Box(low=self.SPEAKER_CYCLE_T, high=self.SPEAKER_CYCLE_T, shape=(1,), dtype=np.float32), # speaker_cycle_t
                                            spaces.Box(low=self.AMP_CYCLE_T, high=self.AMP_CYCLE_T, shape=(1,), dtype=np.float32), # amp_cycle_t
                                            spaces.Discrete(2), # can_score_amp
                                            spaces.Discrete(2), # can_score_speaker
                                            spaces.Box(low=0, high=13, shape=(1,), dtype=np.int32), # amp_time_left
                                            spaces.Discrete(3) # notes_in_amp
                                            ))
        
        # make observation_space a dict for easy access
        self.observation_space = spaces.Dict({
                                        "time_left": spaces.Box(low=0, high=max_steps+1, shape=(1,), dtype=np.int32),
                                        "total_cycle_t": spaces.Box(low=0, high=max_steps, shape=(1,), dtype=np.int32),
                                        "speaker_cycle_t": spaces.Box(low=self.SPEAKER_CYCLE_T, high=self.SPEAKER_CYCLE_T, shape=(1,), dtype=np.float32),
                                        "amp_cycle_t": spaces.Box(low=self.AMP_CYCLE_T, high=self.AMP_CYCLE_T, shape=(1,), dtype=np.float32),
                                        "can_score_amp": spaces.Discrete(2),
                                        "can_score_speaker": spaces.Discrete(2),
                                        "amp_time_left": spaces.Box(low=0, high=13, shape=(1,), dtype=np.int32),
                                        "notes_in_amp": spaces.Discrete(3)
                                    })

    def action_mask(self):
        # mask is a list of 1s and 0s, where 1 means the action is allowed
        mask = [0, 1, 1, 1, 1]
        # action space is, [wait,  move_cycle, score_amp, score_speaker, activate_amp]
        if self.time_spent_cycling < self.SPEAKER_CYCLE_T or not self.CAN_SCORE_SPEAKER:
            mask[3] = 0
        if self.time_spent_cycling < self.AMP_CYCLE_T or not self.CAN_SCORE_AMP:
            mask[2] = 0
        if self.notes_in_amp != 2:
            mask[4] = 0
        return mask

    def _get_info(self):
        return {"actions_taken": self._actions_taken, "total_episode_reward": self._total_episode_reward}
        
    def _get_obs(self):
        # obs space is [time_left, total_cycle_t, speaker_cycle_t, amp_cycle_t, can_score_amp, can_score_speaker, amp_time_left, notes_in_amp]
        #obs = (self._max_steps - self._step_count, self.time_spent_cycling, self.SPEAKER_CYCLE_T, self.AMP_CYCLE_T, self.CAN_SCORE_AMP, self.CAN_SCORE_SPEAKER, self.amp_time_left, self.notes_in_amp)
        # return dict
        obs = {
            "time_left": np.array([self._max_steps - self._step_count], dtype=np.int32),
            "total_cycle_t": np.array([self.time_spent_cycling], dtype=np.int32),
            "speaker_cycle_t": np.array([self.SPEAKER_CYCLE_T], dtype=np.float32),
            "amp_cycle_t": np.array([self.AMP_CYCLE_T], dtype=np.float32),
            "can_score_amp": np.array([self.CAN_SCORE_AMP], dtype=np.int32),
            "can_score_speaker": np.array([self.CAN_SCORE_SPEAKER], dtype=np.int32),
            "amp_time_left": np.array([self.amp_time_left], dtype=np.int32),
            "notes_in_amp": np.array([self.notes_in_amp], dtype=np.int32)
        }
        return obs
    

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self.time_spent_cycling = 0
        self._step_count = 0
        self._actions_taken = []
        self._total_episode_reward = 0 
        self.amp_time_left = 0
        self.notes_in_amp = 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()
        #print(f"RESET: {observation}")
        return observation, info


    # probably add printouts here? Not anything to render
    def render(self, mode='human'):
        
        assert (self._step_count is not None), \
            "Call reset before using render method."
        print("RENDER METHOD NOT IMPLEMENTED")

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        action_str = self._action_names[action]
        self._actions_taken.append(action_str)
        if action_str == 'wait':
            reward = self._step_cost
        elif action_str == 'cycle':
            self.time_spent_cycling += 1
            if self.time_spent_cycling > max(self.SPEAKER_CYCLE_T, self.AMP_CYCLE_T):
                reward = self._step_cost
            else:
                reward = 0
        elif action_str == 'score_amp':

            #if self.CAN_SCORE_AMP and self.time_spent_cycling >= self.AMP_CYCLE_T:
            reward = self._amp_reward
            self.notes_in_amp += 1

            self.time_spent_cycling = 0
        elif action_str == 'score_speaker':
            assert self.CAN_SCORE_SPEAKER, "Speaker cannot be scored at this time"
            assert self.time_spent_cycling >= self.SPEAKER_CYCLE_T, "Speaker cannot be scored at this time"
            #if self.CAN_SCORE_SPEAKER and self.time_spent_cycling >= self.SPEAKER_CYCLE_T:
            if self.amp_time_left > 0:
                reward = self._amped_speaker_reward
                #print("AMPED SPEAKER SCORED!!")
            else:
                reward = self._speaker_reward

            self.time_spent_cycling = 0

        elif action_str == 'activate_amp':
            #if self.notes_in_amp == 2:
            self.notes_in_amp = 0
            self.amp_time_left = 13
            reward = self._amp_reward
            #print("AMP ACTIVATED!!")
        else:
            raise ValueError(f"Invalid action: {action}")
        
        self._step_count += 1
        terminated = self._step_count >= self._max_steps
        self._total_episode_reward += reward
        if self.amp_time_left > 0:
            self.amp_time_left -= 1
        
        if self.notes_in_amp > 2:
            self.notes_in_amp = 2

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    ''' 
    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
    '''
