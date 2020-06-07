import gym
import numpy as np

from typing import Dict, Tuple


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self,
                 env: gym.Wrapper,
                 ):
        """
        Signals end of episode on loss of life to improve value estimation.
        Slightly modified from OpenAI baselines AtariWrappers. As detailed in Mnih et al. (2015) -- aka Nature paper.
        :param env: the inner environment
        """
        super().__init__(env)
        self.lives = 0
        self.done = True

    def reset(self,
              **kwargs,
              ) -> np.ndarray:
        """
        Resets the environment
        :param kwargs:
        :return:
        """
        if self.done:
            state = self.env.reset(**kwargs)
        else:
            state, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return state

    def step(self,
             action: int,
             ) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Performs the provided action
        :param action: the action taken
        :return: state, reward, done, information dictionary
        """
        state, reward, done, info = self.env.step(action)
        self.done = done
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            done = True
        self.lives = lives
        return state, reward, done, info
