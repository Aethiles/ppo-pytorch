import gym
import numpy as np

from typing import Dict, Tuple, Union


class StateStackEnv(gym.Wrapper):
    def __init__(self,
                 env: gym.Wrapper,
                 state_shape: Union[Tuple[int, int], Tuple[int, int, int]] = (84, 84),
                 ):
        """
        Returns the history of the 4 most recent states.
        Slightly modified from OpenAI baselines AtariWrappers. As detailed in Mnih et al. (2013) -- aka DQN paper.
        :param env: the inner environment
        :param state_shape: shape of the state needed to construct the state buffer
        """
        super().__init__(env)
        self.state_stack = np.zeros((4,) + state_shape)
        self.observation_space = np.zeros(self.state_stack.shape)

    def reset(self,
              **kwargs,
              ) -> np.ndarray:
        """
        Resets the environment
        :param kwargs:
        :return:
        """
        self.state_stack[:] = 0
        state = self.env.reset()
        self.state_stack[-1] = state
        return self.state_stack

    def step(self,
             action: int,
             ) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Performs the provided action
        :param action:
        :return:
        """
        state, reward, done, info = self.env.step(action)
        self.state_stack = np.roll(self.state_stack, shift=-1, axis=0)
        self.state_stack[-1] = state
        return self.state_stack, reward, done, info
