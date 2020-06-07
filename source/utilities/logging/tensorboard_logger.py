import numpy as np
import time
import torch

from datetime import datetime
from source.utilities.config.hyperparameters import HyperParameters
from source.utilities.logging.abstract_logger import AbstractLogger
from torch.utils.tensorboard import SummaryWriter


class TensorboardLogger(AbstractLogger):
    def __init__(self,
                 config: HyperParameters,
                 num_eval_episodes: int = 100,
                 ):
        super().__init__()
        log_dir = 'experiments/{}_{}_{}'.format(config.env_name.replace('NoFrameskip', ''),
                                                datetime.utcfromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M'),
                                                config.hostname)
        self.writer = SummaryWriter(log_dir)
        self.log_config(config)
        self.time_step = 0

    def finalize(self):
        """
        Closes the SummaryWriter.
        :return: None
        """
        self.log_final_score(np.float(np.mean(self.reward_buffer)))
        self.writer.close()

    def log_config(self, config):
        """
        Logs the Hyperparameters of the current experiment as text.
        :param config: the config
        :return: None
        """
        self.writer.add_text('Config', config.as_markdown())

    def log_final_score(self,
                        score: float,
                        ):
        """
        Logs the final score of the trained policy.
        :param score: the score
        :return: None
        """
        # TODO revise if this works better as a scalar or as text
        self.writer.add_text('Final_episodes_score', str(score))

    def log_gradient_norm(self):
        """
        Logs the mean of gradient norms observed in this episode's PPO training.
        :return: None
        """
        self.writer.add_scalar('Policy/gradient_norm', np.float(np.mean(self.gradient_norm_buffer)), self.episode)
        self.gradient_norm_buffer.clear()

    def log_hyperparameters(self, learn_rate: float, clip_param: float):
        """
        Logs the learn rate and clip range parameter used in this episode's PPO training.
        :param learn_rate: the learn rate
        :param clip_param: PPO epsilon
        :return: None
        """
        pass
        # self.writer.add_scalar('Hyper/learn_rate', learn_rate, self.episode)
        # self.writer.add_scalar('Hyper/clip_range', clip_param, self.episode)

    def log_kl_divergence(self,
                          kl_divergence: float,
                          ):
        """
        Logs the KL divergence of the policy before and after this episode's gradient updates.
        :param kl_divergence: the KL divergence
        :return: None
        """
        self.writer.add_scalar('Policy/KL_divergence', kl_divergence, self.episode)

    def log_mean_episode(self,
                         episode_reward: float,
                         episode_length: float,
                         ):
        """
        Logs the mean reward and length of episodes that terminated on this time step.
        :param episode_reward:
        :param episode_length:
        :return: None
        """
        # self.writer.add_scalar('Episode/avg_length', episode_length, self.time_step)
        self.writer.add_scalar('Episode/avg_reward', episode_reward, self.time_step)

    def log_ppo_losses(self):
        """
        Logs the means of losses used for this episode's gradient updates.
        :return:
        """
        self.writer.add_scalar('Loss/full',
                               torch.mean(torch.as_tensor(self.loss_buffer)).item(),
                               self.episode)
        self.writer.add_scalar('Loss/policy',
                               torch.mean(torch.as_tensor(self.policy_loss_buffer)).item(),
                               self.episode)
        self.writer.add_scalar('Loss/value',
                               torch.mean(torch.as_tensor(self.value_loss_buffer)).item(),
                               self.episode)
        self.writer.add_scalar('Policy/entropy',
                               torch.mean(torch.as_tensor(self.entropy_buffer)).item(),
                               self.episode)
        self.loss_buffer.clear()
        self.policy_loss_buffer.clear()
        self.value_loss_buffer.clear()
        self.entropy_buffer.clear()

    def log_reward(self,
                   reward: float,
                   ):
        """
        Updates the cumulative reward using the mean reward of the current time step and logs it.
        :param reward: the reward
        :return:
        """
        self.cumulative_reward += reward
        self.writer.add_scalar('Rewards/Cumulative', self.cumulative_reward, self.time_step)
