
from typing import List, Tuple, Union
import gymnasium as gym
import numpy as np
# spaces
from gymnasium import spaces

class Robot:
    time_spent_cycling = 0
    AMP_CYCLE_T = 0
    AMP_ACC = 0
    SPEAKER_CYCLE_T = 0
    SPEAKER_ACC = 0
    CAN_SCORE_AMP = False
    CAN_SCORE_SPEAKER = False
    id = 0
    
    def __init__(self, id: int, amp_cycle_t: int, speaker_cycle_t: int, can_score_amp: bool, can_score_speaker: bool):
        self.id = id
        self.AMP_CYCLE_T = amp_cycle_t
        #self.AMP_ACC = AMP_ACC 
        self.SPEAKER_CYCLE_T = speaker_cycle_t
        #self.SPEAKER_ACC = SPEAKER_ACC 
        self.CAN_SCORE_AMP = can_score_amp 
        self.CAN_SCORE_SPEAKER = can_score_speaker
        #print(f"Robot created with AMP_CYCLE_T: {self.AMP_CYCLE_T}, SPEAKER_CYCLE_T: {self.SPEAKER_CYCLE_T}, CAN_SCORE_AMP: {self.CAN_SCORE_AMP}, CAN_SCORE_SPEAKER: {self.CAN_SCORE_SPEAKER}")
        self.time_spent_cycling = 0
        

class FRC(gym.Env):

    metadata = {'render.modes': ['text']}

    def __init__(self, robots=None,
                 step_cost: float = -0.11, amp_reward: float = 1.0, speaker_reward: float = 2.0,
                 amped_speaker_reward: float = 5.0, max_steps: int = 120):
        assert robots is not None, "Must pass in robots"
        self.number_of_robots = len(robots)
        self.robots = robots
        self.amp_time_left = 0
        self.notes_in_amp = 0

        self._step_cost = step_cost
        self._step_count = None
        self._amp_reward = amp_reward
        self._speaker_reward = speaker_reward
        self._amped_speaker_reward = amped_speaker_reward
        self._max_steps = max_steps * self.number_of_robots
        self.notes_scored_while_amp_active = 0
        self._actions_taken = {"amp_active": []}
        for i in range(self.number_of_robots):
            self._actions_taken[i] = []

        self._total_episode_reward = None

        # action space is, [wait,  move_cycle, score_amp, score_speaker, activate_amp, score_speaker_amped]
        self.action_space = spaces.Discrete(6)
        self._action_ids = {
            'wait': 0,
            'cycle': 1,
            'score_amp': 2,
            'score_speaker': 3,
            'activate_amp': 4,
            'score_speaker_amped': 5
        }
        # invert the dictionary
        self._action_names = {v: k for k, v in self._action_ids.items()}
        
        # make observation_space a dict for easy access
        self.observation_space = spaces.Dict({
                                        # global states
                                        "time_left": spaces.Box(low=0, high=self._max_steps+1, shape=(1,), dtype=np.int32),
                                        "amp_time_left": spaces.Box(low=0, high=14 * self.number_of_robots, shape=(1,), dtype=np.int32),
                                        "notes_in_amp": spaces.Discrete(3),
                                        "notes_scored_while_amp_active": spaces.Box(low=0, high=10, shape=(1,), dtype=np.int32),
                                    })
        for robot in self.robots: 
            self.observation_space.spaces[f"time_spent_cycling_{robot.id}"] = spaces.Box(low=0, high=max_steps, shape=(1,), dtype=np.int32)
            self.observation_space.spaces[f"speaker_cycle_t_{robot.id}"] = spaces.Box(low=robot.SPEAKER_CYCLE_T, high=robot.SPEAKER_CYCLE_T, shape=(1,), dtype=np.float32)
            self.observation_space.spaces[f"amp_cycle_{robot.id}"] = spaces.Box(low=robot.AMP_CYCLE_T, high=robot.AMP_CYCLE_T, shape=(1,), dtype=np.float32)
            self.observation_space.spaces[f"can_score_amp_{robot.id}"] = spaces.Discrete(2)
            self.observation_space.spaces[f"can_score_speaker_{robot.id}"] = spaces.Discrete(2)
        
        print(f"Observation space: {self.observation_space}")

    def action_mask(self):
        robot = self.get_current_robot()
        # mask is a list of 1s and 0s, where 1 means the action is allowed
        mask = [0, 1, 1, 1, 0, 0]
        # action space is, [wait,  move_cycle, score_amp, score_speaker, activate_amp, score_speaker_amped]
        if robot.time_spent_cycling < robot.SPEAKER_CYCLE_T or not robot.CAN_SCORE_SPEAKER:
            mask[3] = 0
        if robot.time_spent_cycling < robot.AMP_CYCLE_T or not robot.CAN_SCORE_AMP:
            mask[2] = 0
        if self.notes_in_amp != 2:
            mask[4] = 0
        if (self.amp_time_left > 0 or self.notes_in_amp == 2) and robot.time_spent_cycling >= robot.SPEAKER_CYCLE_T and robot.CAN_SCORE_SPEAKER:
            mask[5] = 1
            # set all other actions to 0
            mask[0] = 0
            mask[1] = 0
            mask[2] = 0
            mask[3] = 0
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
            "amp_time_left": np.array([self.amp_time_left], dtype=np.int32),
            "notes_in_amp": np.array([self.notes_in_amp], dtype=np.int32),
            "notes_scored_while_amp_active": np.array([self.notes_scored_while_amp_active], dtype=np.int32)
        }
        for robot in self.robots: 
            obs[f"time_spent_cycling_{robot.id}"] = np.array([robot.time_spent_cycling], dtype=np.int32)
            obs[f"speaker_cycle_t_{robot.id}"] = np.array([robot.SPEAKER_CYCLE_T], dtype=np.float32)
            obs[f"amp_cycle_{robot.id}"] = np.array([robot.AMP_CYCLE_T], dtype=np.int32)
            obs[f"can_score_amp_{robot.id}"] = np.array([int(robot.CAN_SCORE_AMP)], dtype=np.int32)
            obs[f"can_score_speaker_{robot.id}"] = np.array([int(robot.CAN_SCORE_SPEAKER)], dtype=np.int32)
        return obs
    
    def get_current_robot(self) -> Robot:
        return self.robots[self._step_count % self.number_of_robots]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._step_count = 0
        self._actions_taken = {"amp_active": []}
        for i in range(self.number_of_robots):
            self._actions_taken[i] = []
        self.notes_scored_while_amp_active = 0
        self._total_episode_reward = 0 
        self.amp_time_left = 0
        self.notes_in_amp = 0
        for robot in self.robots:
            robot.time_spent_cycling = 0

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
        if self.notes_scored_while_amp_active > 0 and self.amp_time_left == 0:
            self.notes_scored_while_amp_active = 0
        action_str = self._action_names[action]
        self._actions_taken[self._step_count % self.number_of_robots].append(action_str)
        if self._step_count % self.number_of_robots == 0:
            self._actions_taken["amp_active"].append(self.amp_time_left)
        current_robot = self.get_current_robot()
        assert id(current_robot) == id(self.robots[self._step_count % self.number_of_robots]), "Robot is being copied, not referenced!"
        if action_str == 'wait':
            #reward = self._step_cost
            reward = 0
        elif action_str == 'cycle':
            current_robot.time_spent_cycling += 1
            reward = 0
        elif action_str == 'score_amp':

            #if self.CAN_SCORE_AMP and self.time_spent_cycling >= self.AMP_CYCLE_T:
            reward = self._amp_reward
            if self.amp_time_left <= 2 * self.number_of_robots:
                self.notes_in_amp += 1
            current_robot.time_spent_cycling = 0

        elif action_str == 'score_speaker':
            assert current_robot.CAN_SCORE_SPEAKER, "Speaker cannot be scored at this time"
            assert current_robot.time_spent_cycling >= current_robot.SPEAKER_CYCLE_T, "Speaker cannot be scored at this time"
            #if self.CAN_SCORE_SPEAKER and self.time_spent_cycling >= self.SPEAKER_CYCLE_T:
            reward = self._speaker_reward
            current_robot.time_spent_cycling = 0

        elif action_str == 'activate_amp':
            #if self.notes_in_amp == 2:
            self.notes_in_amp = 0
            self.amp_time_left = 14 * self.number_of_robots
            current_robot.time_spent_cycling += 1
            #reward = self._amp_reward
            reward = 0
            #print("AMP ACTIVATED!!")
        elif action_str == 'score_speaker_amped':
            assert current_robot.CAN_SCORE_SPEAKER, "Speaker cannot be scored at this time"
            assert current_robot.time_spent_cycling >= current_robot.SPEAKER_CYCLE_T, "Speaker cannot be scored at this time"
            #assert self.amp_time_left > 0, "Amp must be active to score amped speaker"
            assert self.notes_in_amp == 2 or self.amp_time_left > 0, "Amp must have 2 notes to score amped speaker"
            if self.notes_in_amp == 2:
                self.notes_in_amp = 0
                self.amp_time_left = 14 * self.number_of_robots
            self.notes_scored_while_amp_active += 1
            current_robot.time_spent_cycling += 1
            reward = self._amped_speaker_reward * (2 ** (self.notes_scored_while_amp_active - 1))
            current_robot.time_spent_cycling = 0
        else:
            raise ValueError(f"Invalid action: {action}")
        
        self._step_count += 1
        terminated = self._step_count >= self._max_steps
        if reward == 5 or reward == 10 or reward == 15:
            self._total_episode_reward += 5
        else:
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
