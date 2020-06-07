import numpy as np
import torch
import unittest

from source.environment.atari.evaluation import EpisodeStats
from source.utilities.logging.tensorboard_logger import TensorboardLogger
from unittest.mock import MagicMock, patch


class AbstractLoggerTest(unittest.TestCase):
    def setUp(self):
        with patch('source.utilities.logging.tensorboard_logger.SummaryWriter') as mock:
            mock.return_value = mock
            mock.add_text = MagicMock()
            mock.add_scalar = MagicMock()
            self.summary_mock = mock
            self.logger = TensorboardLogger(config=MagicMock())
            self.logger.time_step = np.random.randint(10, 100)
            self.logger.episode = np.random.randint(10, 100)

    def test_finalize(self):
        self.logger.finalize()
        self.summary_mock.close.assert_called_once()

    def test_log_gradient_norm(self):
        self.logger.store_gradient_norm(0)
        self.logger.store_gradient_norm(1)
        self.logger.log_gradient_norm()
        self.summary_mock.add_scalar.assert_called_once()
        self.summary_mock.add_scalar.assert_called_with('Policy/gradient_norm', 0.5, self.logger.episode)
        self.assertEqual(0, len(self.logger.gradient_norm_buffer))

    def test_log_hyperparameters(self):
        self.logger.log_hyperparameters(0.1, 0.1)
        self.assertEqual(2, self.summary_mock.add_scalar.call_count)
        for args, kwargs in self.summary_mock.add_scalar.call_args_list:
            self.assertEqual(0.1, args[1])
            self.assertEqual(self.logger.episode, args[2])

    def test_log_kl_divergence(self):
        self.logger.log_kl_divergence(0.777)
        self.summary_mock.add_scalar.assert_called_with('Policy/KL_divergence', 0.777, self.logger.episode)

    def test_log_reward(self):
        self.logger.log_reward(0.5)
        self.summary_mock.add_scalar.assert_called_with('Rewards/Cumulative', 0.5, self.logger.time_step)
        self.logger.log_reward(1.5)
        self.logger.log_reward(-0.5)
        self.summary_mock.add_scalar.assert_called_with('Rewards/Cumulative', 1.5, self.logger.time_step)

    def test_log_terminated_episodes(self):
        dones = [False, False, True, True]
        infos = [{}, {}, {'episode': EpisodeStats(120, 25)}, {'episode': EpisodeStats(60, 15)}]
        self.logger.log_terminated_episodes(dones, infos)
        # self.assertEqual('Episode/avg_length', self.summary_mock.add_scalar.call_args_list[0][0][0])
        # self.assertEqual(20, self.summary_mock.add_scalar.call_args_list[0][0][1])
        # self.assertEqual(self.logger.time_step, self.summary_mock.add_scalar.call_args_list[0][0][2])
        self.assertEqual('Episode/avg_reward', self.summary_mock.add_scalar.call_args_list[0][0][0])
        self.assertEqual(90, self.summary_mock.add_scalar.call_args_list[0][0][1])
        self.assertEqual(self.logger.time_step, self.summary_mock.add_scalar.call_args_list[0][0][2])

    def test_log_step(self):
        dones = [False, False, True, True]
        infos = [{}, {}, {'episode': EpisodeStats(120, 25)}, {}]
        rewards = [0.5, 0, 1.0, 0.5]
        self.logger.log_step(dones, infos, rewards)
        # self.assertEqual('Episode/avg_length', self.summary_mock.add_scalar.call_args_list[0][0][0])
        # self.assertEqual(25, self.summary_mock.add_scalar.call_args_list[0][0][1])
        # self.assertEqual(self.logger.time_step - 1, self.summary_mock.add_scalar.call_args_list[0][0][2])
        self.assertEqual('Episode/avg_reward', self.summary_mock.add_scalar.call_args_list[0][0][0])
        self.assertEqual(120, self.summary_mock.add_scalar.call_args_list[0][0][1])
        self.assertEqual(self.logger.time_step - 1, self.summary_mock.add_scalar.call_args_list[0][0][2])
        self.assertEqual('Rewards/Cumulative', self.summary_mock.add_scalar.call_args_list[1][0][0])
        self.assertEqual(0.5, self.summary_mock.add_scalar.call_args_list[1][0][1])
        self.assertEqual(self.logger.time_step - 1, self.summary_mock.add_scalar.call_args_list[1][0][2])

    def test_end_episode(self):
        values = np.random.rand(5)
        self.logger.store_gradient_norm(values[0])
        self.logger.store_ppo_losses(torch.tensor(values[1]), torch.tensor(values[2]), torch.tensor(values[3]),
                                     torch.tensor(values[4]))
        self.logger.end_episode()
        self.assertEqual(1, self.summary_mock.add_text.call_count)
        self.assertEqual(5, self.summary_mock.add_scalar.call_count)
        self.assertEqual(0, len(self.logger.loss_buffer))
        self.assertEqual(0, len(self.logger.policy_loss_buffer))
        self.assertEqual(0, len(self.logger.value_loss_buffer))
        self.assertEqual(0, len(self.logger.entropy_buffer))
        for i, (args, kwargs) in enumerate(self.summary_mock.add_scalar.call_args_list):
            self.assertAlmostEqual(values[i], args[1])
            self.assertEqual(self.logger.episode, args[2])


if __name__ == '__main__':
    unittest.main()
