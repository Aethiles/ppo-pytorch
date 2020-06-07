import os.path
import pickle
import time
import torch

from datetime import datetime
from pathlib import Path
from source.utilities.config.hyperparameters import HyperParameters
from source.utilities.logging.abstract_logger import AbstractLogger


class ListLogger(AbstractLogger):
    def __init__(self,
                 num_training_episodes: int,
                 config: HyperParameters,
                 root_directory: str = '',
                 verbose: bool = False,
                 ):
        """
        Initializes a new Logger.
        :param num_training_episodes: the number of episodes used in training
        :param config: the hyperparameters used in this experiment
        :param root_directory: the root directory the logger shall be saved in
        :param verbose: if True prints the current configuration
        """
        super().__init__()
        self.max_time_steps = num_training_episodes
        self.config = config

        self.episode_rewards = {}
        self.episode_lengths = {}
        self.recent_rewards = {}
        self.recent_lengths = {}

        self.losses = []
        self.policy_losses = []
        self.value_losses = []
        self.entropy_bonuses = []
        self.gradient_norms = []
        self.kl_divergences = []

        self.learn_rates = []
        self.clip_params = []

        self.final_score = None

        self.file_name = '{}.p'.format(datetime.utcfromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M'))
        self.root_directory = root_directory

        self.verbose = verbose
        if verbose:
            print('Running:\n{}'.format(config))

    @classmethod
    def from_pickle(cls,
                    path: str,
                    ) -> 'ListLogger':
        """
        Load a logger from a pickle.
        :param path: path to the pickle
        :return: the logger
        """
        logger = pickle.load(open(path, 'rb'))
        return logger

    def finalize(self):
        """
        Stores a pickle of self using root_directory and name. Creates all missing directories.
        :return: None
        """
        path = os.path.join(self.root_directory, self.config.env_name)
        Path(path).mkdir(parents=True, exist_ok=True)
        full_path = os.path.join(path, self.file_name)
        with open(full_path, 'wb') as f:
            pickle.dump(self, f)

    def log_final_score(self,
                        score: float,
                        ):
        """
        Logs the final score recorded over several episodes.
        :param score: the score
        :return: None
        """
        self.final_score = score
        print('Final performance: {}'.format(score))

    def log_gradient_norm(self):
        """
        Logs the gradient norms of this episode's gradient updates.
        :return: None
        """
        self.gradient_norms.append(torch.as_tensor(self.gradient_norm_buffer).mean().item())
        self.gradient_norm_buffer = []

    def log_hyperparameters(self,
                            learn_rate: float,
                            clip_param: float,
                            ):
        """
        Logs the learn rate and clip range parameter used in this episode's PPO training.
        :param learn_rate: the learn rate
        :param clip_param: PPO epsilon
        :return: None
        """
        self.learn_rates.append(learn_rate)
        self.clip_params.append(clip_param)

    def log_mean_episode(self,
                         episode_reward: float,
                         episode_length: float,
                         ):
        """
        Logs the given episode reward and length.
        :param episode_reward: the episode reward
        :param episode_length: the episode length
        :return:
        """
        self.recent_rewards[self.time_step] = episode_reward
        self.recent_lengths[self.time_step] = episode_length

    def log_ppo_losses(self):
        """
        Logs the means of losses used for this episode's gradient updates.
        :return:
        """
        self.losses.append(torch.mean(torch.as_tensor(self.loss_buffer)).item())
        self.policy_losses.append(torch.mean(torch.as_tensor(self.policy_loss_buffer)).item())
        self.value_losses.append(torch.mean(torch.as_tensor(self.value_loss_buffer)).item())
        self.entropy_bonuses.append(torch.mean(torch.as_tensor(self.entropy_buffer)).item())
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

    def log_kl_divergence(self,
                          kl_divergence: float,
                          ):
        """
        Logs the Kullback-Leibler divergence.
        :param kl_divergence: the kl-divergence
        :return: None
        """
        self.kl_divergences.append(kl_divergence)

    def end_episode(self):
        """
        Moves the recently observed episodes to a long-term buffer and clears the observed losses. If requested,
        information on the agent's and the algorithm's performance are printed before doing so.
        :return: None
        """
        super().end_episode()
        if self.verbose:
            print(self._format())
        self.episode_rewards.update(self.recent_rewards)
        self.episode_lengths.update(self.recent_lengths)
        self.recent_rewards = {}
        self.recent_lengths = {}
        self.episode += 1

    def _format(self) -> str:
        """
        Returns a string containing current information on the agent's performance and performance details of the
        algorithm, such as elapsed and remaining time and time steps performed per second.
        :return: a string
        """
        now = time.time()
        duration = (now - self.start_time) * 1000
        steps_per_sec = self.config.horizon / (duration / 1000)
        remaining_duration = (self.max_time_steps - self.time_step) / steps_per_sec
        printable = '----------------------------------------------\n' \
                    'Time:\n' \
                    'avg episode length: {}\n' \
                    'elapsed time (ms):  {}\n' \
                    'remaining (s):      {}\n' \
                    'steps/s:            {}\n' \
                    'Performance:\n' \
                    'avg reward:         {}\n' \
                    'avg full loss:      {}\n' \
                    'avg policy loss:    {}\n' \
                    'avg value loss:     {}\n' \
                    'avg entropy bonus:  {}\n' \
                    'avg grad norm:      {}\n' \
                    'kl divergence:      {}\n' \
                    'adam learn rate:    {}\n' \
                    'ppo clip param:     {}\n' \
                    '----------------------------------------------' \
            .format(self.recent_lengths,
                    duration,
                    remaining_duration,
                    steps_per_sec,
                    self.recent_rewards,
                    self.losses[-1],
                    self.policy_losses[-1],
                    self.value_losses[-1],
                    self.entropy_bonuses[-1],
                    self.gradient_norms[-1],
                    self.kl_divergences[-1],
                    self.learn_rates[-1],
                    self.clip_params[-1])
        return printable

