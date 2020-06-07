import gym
import numpy as np

from abc import abstractmethod
from enum import auto, Enum


class ClippingMode(Enum):
    BIN = auto()
    CLIP = auto()


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self,
                 env: gym.Wrapper,
                 mode: ClippingMode = ClippingMode.BIN,
                 **kwargs,
                 ):
        """
        Transforms the reward by either binning rewards according to their sign or clipping them to [-5, +5].
        Binning as done by OpenAI in baselines. Clipping as detailed by Ilyas et al. (2018).
        :param env: the inner environment
        :param mode: the mode the reward shall be modified by
        :param kwargs: keyword arguments to be used in clipping mode
        :keyword min: the minimum reward in clipping mode
        :keyword max: the maximum reward in clipping mode
        """
        super().__init__(env)
        if mode is ClippingMode.BIN:
            self.reward = lambda r: np.sign(r)
        elif mode is ClippingMode.CLIP:
            if kwargs is None or kwargs == {}:
                min_reward = -5
                max_reward = +5
            else:
                min_reward = kwargs['min']
                max_reward = kwargs['max']
            self.reward = lambda r: np.clip(r, min_reward, max_reward)

    @abstractmethod
    def reward(self, reward):
        pass
