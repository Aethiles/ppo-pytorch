import gym
import numpy as np

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class EpisodeStats:
    reward: float
    length: int


class EvaluationEnv(gym.Wrapper):
    def __init__(self,
                 env: gym.Wrapper,
                 ):
        """
        Records rewards so the performance of the agent can be evaluated. Returns the sum of all rewards observed once
        an episode ended, that is once the environment returned True.
        :param env: the inner environment
        """
        super().__init__(env)
        self.rewards = []

    def step(self,
             action: int,
             ) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Performs the given action returning the resulting state, reward, information if a terminal state was reached and
        further information. If a terminal state was reached, the further information is extended with the reward that
        was gathered during this episode.
        :param action:
        :return: the observed state, the observed reward, True if a terminal state was reached, further information
        """
        state, reward, done, info = self.env.step(action)
        self.rewards.append(reward)
        if done:
            stats = EpisodeStats(sum(self.rewards), len(self.rewards))
            info['episode'] = stats
        return state, reward, done, info

    def reset(self,
              **kwargs,
              ) -> np.ndarray:
        """
        Resets the environment and the reward buffer.
        :param kwargs:
        :return: the observed state
        """
        self.rewards = []
        return self.env.reset(**kwargs)
