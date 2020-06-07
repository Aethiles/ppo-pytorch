import numpy as np
import time
import torch

from abc import ABC, abstractmethod
from collections import deque
from typing import Dict, List


class AbstractLogger(ABC):
    def __init__(self,
                 num_eval_episodes: int = 100,
                 ):
        """
        Abstract base Logger that all other loggers inherit from. Used to supervise and evaluate experiments.
        :param num_eval_episodes: number of episodes used for evaluation of final score
        """
        self.loss_buffer = []
        self.policy_loss_buffer = []
        self.value_loss_buffer = []
        self.entropy_buffer = []
        self.gradient_norm_buffer = []
        self.cumulative_reward = 0
        self.reward_buffer = deque(maxlen=num_eval_episodes)

        self.start_time = 0
        self.episode = 0
        self.time_step = 0

    @abstractmethod
    def finalize(self):
        """
        Abstract method used to finalize logging, e.g. by saving a log object or closing a file writer.
        :return: None
        """
        pass

    @abstractmethod
    def log_final_score(self,
                        score: float,
                        ):
        """
        Abstract method used to log the final score of a policy.
        :param score: the score
        :return: None
        """
        pass

    @abstractmethod
    def log_gradient_norm(self):
        """
        Abstract method used to transfer gradient norms from the corresponding buffer to the log.
        :return: None
        """
        pass

    def store_gradient_norm(self,
                            gradient_norm: float,
                            ):
        """
        Stores the given gradient norm for later logging.
        :param gradient_norm:
        :return: None
        """
        self.gradient_norm_buffer.append(gradient_norm)

    @abstractmethod
    def log_hyperparameters(self,
                            learn_rate: float,
                            clip_param: float,
                            ):
        """
        Abstract method used to store the hyperparameters that change over the course of training.
        :param learn_rate: the learn rate
        :param clip_param: PPO epsilon
        :return: None
        """
        pass

    @abstractmethod
    def log_kl_divergence(self,
                          kl_divergence: float,
                          ):
        """
        Abstract method used to log the KL divergence.
        :param kl_divergence:
        :return: None
        """
        pass

    @abstractmethod
    def log_mean_episode(self,
                         episode_reward: float,
                         episode_length: float,
                         ):
        """
        Abstract method used to log the observed cumulative reward and length of an episode.
        :param episode_reward:
        :param episode_length:
        :return: None
        """
        pass

    def log_terminated_episodes(self,
                                dones: List[bool],
                                infos: List[Dict],
                                ):
        """

        :param dones:
        :param infos:
        :return:
        """
        episodes = [infos[i].get('episode') for i, done in enumerate(dones) if done and 'episode' in infos[i]]
        rewards = []
        for episode in episodes:
            rewards.append(episode.reward)
            self.reward_buffer.append(episode.reward)
        if len(rewards) > 0:
            self.log_mean_episode(np.float(np.mean(rewards)), 0)
        # try:
        #     rewards, lengths = zip(*[(episode.reward, episode.length) for episode in episodes])
        #     self.log_mean_episode(np.float(np.mean(rewards)), np.float(np.mean(lengths)))
        # except ValueError as _:
        #     pass

    @abstractmethod
    def log_ppo_losses(self):
        """
        Abstract method used to transfer losses and entropy from their corresponding buffers to the log.
        :return: None
        """
        pass

    def store_ppo_losses(self,
                         loss: torch.Tensor,
                         policy_loss: torch.Tensor,
                         value_loss: torch.Tensor,
                         entropy: torch.Tensor,
                         ):
        """
        Stores the given losses for later logging.
        :param loss: the full loss
        :param policy_loss: the policy loss
        :param value_loss: the value function loss
        :param entropy: the policy entropy
        :return: None
        """
        self.loss_buffer.append(loss)
        self.policy_loss_buffer.append(policy_loss)
        self.value_loss_buffer.append(value_loss)
        self.entropy_buffer.append(entropy)

    @abstractmethod
    def log_reward(self,
                   reward: float,
                   ):
        """
        Abstract method used to track the cumulative reward.
        :param reward:
        :return:
        """

    def log_step(self,
                 dones: List[bool],
                 infos: List[Dict],
                 rewards: List[float],
                 ):
        """

        :param dones:
        :param infos:
        :param rewards:
        :return:
        """
        self.log_terminated_episodes(dones, infos)
        self.log_reward(np.float(np.mean(rewards)))
        self.time_step += 1

    def start_episode(self):
        """
        Starts a new episode storing the time it's started at.
        :return: None
        """
        self.start_time = time.time()

    def end_episode(self):
        """
        Increments the episode counter. To be extended by Loggers.
        :return: None
        """
        self.episode += 1
        self.log_gradient_norm()
        self.log_ppo_losses()
