import gym
import numpy as np

from typing import Dict, Tuple


class NoopResetEnv(gym.Wrapper):
    def __init__(self,
                 env: gym.Wrapper,
                 no_op_max=30,
                 ):
        """
        Samples initial states by performing a random number of no operations on reset.
        Slightly modified from OpenAI baselines AtariWrappers. As detailed in Mnih et al. (2015) -- aka Nature paper.
        :param env: the inner environment
        :param no_op_max: maximum number of no operations
        """
        super().__init__(env)
        self.no_op_max = no_op_max
        self.no_op_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self,
              **kwargs,
              ) -> np.ndarray:
        """
        Resets the environment
        :param kwargs: keyword arguments of the OpenAI core
        :return: state
        """
        self.env.reset(**kwargs)
        no_ops = np.random.randint(1, self.no_op_max + 1)
        state = None
        for _ in range(no_ops):
            state, _, done, _ = self.env.step(self.no_op_action)
            if done:
                state = self.env.reset(**kwargs)
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
