import gym
import numpy as np

from typing import Dict, Tuple


class ActionRepetitionEnv(gym.Wrapper):
    def __init__(self,
                 env: gym.Wrapper,
                 action_repeats: int = 4,
                 ):
        """
        Repeats an action for action_repeats frames. A new state is created by picking the maximum per pixel across two
        consecutive states to adjust for flickering sprites.
        Slightly modified from OpenAI baselines AtariWrappers. As detailed in Mnih et al. (2013) -- aka DQN paper.
        :param env: the inner environment
        :param action_repeats: number of times the action is repeated
        """
        super().__init__(env)
        self.action_repeats = action_repeats
        buffer_shape = (2, ) + self.env.observation_space.shape
        self.state_buffer = np.zeros(buffer_shape, dtype=np.uint8)

    def reset(self,
              **kwargs,
              ) -> np.ndarray:
        """
        Resets the environment
        :param kwargs:
        :return: state
        """
        return self.env.reset(**kwargs)

    def step(self,
             action: int,
             ) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Performs the provided action
        :param action: the action taken
        :return: state, reward, done, information dictionary
        """
        total_reward = 0.0
        done = False
        info = {}
        for i in range(self.action_repeats):
            state, reward, done, info = self.env.step(action)
            total_reward += reward
            if i == self.action_repeats - 2:
                self.state_buffer[0] = state
            elif i == self.action_repeats - 1:
                self.state_buffer[1] = state
            if done:
                break
        # Observation on done frame doesn't matter, because we reset immediately
        # Hence we never provide the frame to the agent
        max_frame = self.state_buffer.max(axis=0, initial=0.0)
        return max_frame, total_reward, done, info
