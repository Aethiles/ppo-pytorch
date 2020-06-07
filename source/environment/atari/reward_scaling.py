import gym
import numpy as np


class RewardScalingEnv(gym.RewardWrapper):
    def __init__(self,
                 env: gym.Wrapper,
                 discount: float,
                 ):
        """
        Keeps track of all observed rewards and scales them by dividing by the current standard deviation of a rolling
        discounted sum of the rewards.
        :param discount:
        """
        super().__init__(env)
        self.discount = discount
        self.rolling_sums = []

    def reward(self,
               reward: float,
               ) -> float:
        """
        Scales the reward by dividing it with the current standard deviation.
        :param reward:
        :return:
        """
        self._update_mean(reward)
        scale = np.float(np.std(self.rolling_sums))
        if scale == 0:
            return reward
        return reward / np.float(np.std(self.rolling_sums))

    def _update_mean(self,
                     reward: float,
                     ):
        """
        Updates the means to include a new reward.
        :param reward:
        :return:
        """
        if len(self.rolling_sums) == 0:
            previous_reward = 0
        else:
            previous_reward = self.rolling_sums[-1]
        self.rolling_sums.append(self.discount * previous_reward + reward)
