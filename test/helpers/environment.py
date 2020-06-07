import gym
import numpy as np

from typing import Callable, List
from unittest.mock import MagicMock


class MockEnv(gym.Env):
    def __init__(self,
                 first: Callable,
                 next: Callable,
                 action_meanings: List = None,
                 observation_space: np.ndarray = np.zeros((84, 84, 3)),
                 lives: int = 5,
                 spec_id: str = 'NoFrameskip',
                 ):
        """

        :param first:
        :param next:
        """
        if action_meanings is None:
            action_meanings = ['NOOP', 'FIRE', '', '']
        self.step_ctr = 0
        self.reset_ctr = 0
        self.first = first
        self.next = next
        self.action_meanings = action_meanings
        self.observation_space = observation_space
        self.ale = MagicMock()
        self.ale.lives.return_value = lives
        self.spec = MagicMock()
        self.spec.id = spec_id

    def get_action_meanings(self):
        return self.action_meanings

    def step(self, action):
        self.step_ctr += 1
        return self.next(action)

    def reset(self):
        self.reset_ctr += 1
        return self.first()

    def render(self, mode='human'):
        pass


class MockWrapper(gym.Wrapper):
    def __init__(self,
                 env: MockEnv,
                 ):
        super().__init__(env)
