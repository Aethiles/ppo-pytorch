import gym
import numpy as np

from typing import Dict, Tuple


class FireResetEnv(gym.Wrapper):
    def __init__(self,
                 env: gym.Wrapper,
                 ):
        """
        Performs a fire action in environments that will not start until the agent fired.
        Slightly modified from OpenAI baselines AtariWrappers.
        :param env: the inner environment
        """
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self,
              **kwargs,
              ) -> np.ndarray:
        """
        Resets the environment
        :param kwargs:
        :return:
        """
        self.env.reset(**kwargs)
        state, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        state, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return state

    def step(self,
             action: int,
             ) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Performs the provided action
        :param action: the action taken
        :return: state, reward, done, information dictionary
        """
        return self.env.step(action)
