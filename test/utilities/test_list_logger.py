import numpy as np
import os.path
import shutil
import torch
import unittest

from source.environment.atari.evaluation import EpisodeStats
from source.utilities.logging.list_logger import ListLogger
from unittest.mock import MagicMock, patch


class LoggerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = ListLogger(1, MagicMock())

    @patch('time.time', return_value=42313370815)
    def test_setup(self, time_wd):
        self.logger.start_episode()
        self.assertEqual(0, self.logger.time_step)
        self.assertEqual(42313370815, self.logger.start_time)

    def test_log_step_none(self):
        self.logger.start_episode()
        self.logger.log_step([False], list(dict()), [5])
        self.assertEqual(5, self.logger.cumulative_reward)
        self.assertEqual(0, len(self.logger.episode_rewards))
        self.assertEqual(0, len(self.logger.episode_lengths))

    def test_log_step_one_of_two(self):
        stats = [{'episode': None}, {'episode': EpisodeStats(1, 2)}]
        self.logger.start_episode()
        self.logger.log_step([False, True], stats, [0])
        self.logger.store_gradient_norm(MagicMock())
        self.logger.store_ppo_losses(MagicMock(), MagicMock(), MagicMock(), MagicMock())
        self.logger.end_episode()
        self.assertEqual(0, self.logger.cumulative_reward)
        self.assertEqual(1, self.logger.episode_rewards[self.logger.time_step - 1])
        self.assertEqual(2, self.logger.episode_lengths[self.logger.time_step - 1])

    def test_log_step_all(self):
        stats = [{'episode': EpisodeStats(1, 1)}, {'episode': EpisodeStats(2, 5)}, {'episode': EpisodeStats(3, 6)}]
        self.logger.start_episode()
        self.logger.log_step([True] * 3, stats, [1, 2, 0.3])
        self.logger.store_gradient_norm(MagicMock())
        self.logger.store_ppo_losses(MagicMock(), MagicMock(), MagicMock(), MagicMock())
        self.logger.end_episode()
        self.assertAlmostEqual(1.1, self.logger.cumulative_reward)
        self.assertEqual(2, self.logger.episode_rewards[self.logger.time_step - 1])
        self.assertEqual(1, len(self.logger.episode_rewards))
        self.assertEqual(4, self.logger.episode_lengths[self.logger.time_step - 1])
        self.assertEqual(1, len(self.logger.episode_lengths))

    def test_log_step_several_episodes(self):
        stats = [{'episode': EpisodeStats(i, i)} for i in range(5)]
        self.logger.start_episode()
        self.logger.log_step([True, True, False, False, False], stats, [-2, 1, 4])
        self.logger.log_step([False] * 5, stats, [1])
        self.logger.log_step([True] * 5, stats, [1])
        self.assertEqual(2, len(self.logger.recent_rewards))
        self.logger.end_episode()
        self.assertEqual(2, len(self.logger.episode_rewards))
        self.assertEqual(2, len(self.logger.episode_lengths))
        self.assertEqual(0.5, self.logger.episode_rewards[0])
        self.assertEqual(2, self.logger.episode_rewards[2])
        self.assertEqual(0.5, self.logger.episode_lengths[0])
        self.assertEqual(2, self.logger.episode_lengths[2])
        self.assertEqual(3, self.logger.cumulative_reward)

    def test_log_fake_done(self):
        self.logger.start_episode()
        self.logger.log_step([False, False, True], [{}] * 3, 0)
        self.assertEqual(0, len(self.logger.recent_rewards))
        self.assertEqual(0, len(self.logger.recent_lengths))
        self.logger.end_episode()
        self.assertEqual(0, len(self.logger.episode_rewards))
        self.assertEqual(0, len(self.logger.episode_lengths))

    def test_log_gradient_norms(self):
        self.logger.start_episode()
        epochs = np.random.randint(4, 8)
        norms = np.random.rand(epochs)
        for norm in norms:
            self.logger.store_gradient_norm(norm)
        for i in range(epochs):
            self.assertEqual(norms[i], self.logger.gradient_norm_buffer[i])
        self.logger.end_episode()
        self.assertAlmostEqual(np.float(np.mean(norms)), self.logger.gradient_norms[0], delta=1e6)

    def test_log_losses(self):
        self.logger.start_episode()
        epochs = np.random.randint(1, 4)
        losses = torch.rand(epochs)
        for loss in losses:
            self.logger.store_ppo_losses(loss, loss, loss, loss)
        self.logger.end_episode()
        for i in range(epochs):
            self.assertEqual(losses.mean().item(), self.logger.policy_losses[0])
            self.assertEqual(losses.mean().item(), self.logger.value_losses[0])
            self.assertEqual(losses.mean().item(), self.logger.entropy_bonuses[0])

    def test_log_hyperparameters(self):
        self.logger.start_episode()
        self.assertEqual(0, len(self.logger.learn_rates))
        self.assertEqual(0, len(self.logger.clip_params))
        self.logger.log_hyperparameters(0.1, 0.5)
        self.assertEqual(1, len(self.logger.learn_rates))
        self.assertEqual(1, len(self.logger.clip_params))

    def test_format_no_episode_completed(self):
        self.logger.start_episode()
        self.assertRaises(IndexError, self.logger._format)

    @unittest.skip
    def test_format(self):
        self.logger.verbose = True
        stats = [{'episode': EpisodeStats(1, 1)}, {'episode': EpisodeStats(2, 2)}]
        self.logger.start_episode()
        self.logger.log_step([True, False], stats, [0])
        losses = torch.rand(4)
        self.logger.store_ppo_losses(losses[0], losses[1], losses[2], losses[3])
        self.logger.log_kl_divergence(0.75)
        self.logger.store_gradient_norm(1.2)
        self.logger.log_hyperparameters(0.1, 0.1)
        self.logger.end_episode()
        self.logger.start_episode()
        self.logger.store_ppo_losses(losses[0], losses[1], losses[2], losses[3])
        self.logger.log_kl_divergence(0.75)
        self.logger.store_gradient_norm(1.2)
        self.logger.log_step([True, True], stats, [1])
        self.logger.end_episode()
        self.logger.verbose = False

    @patch('source.utilities.logging.list_logger.Path')
    @patch('pickle.dump')
    @patch('builtins.open')
    def test_save(self, open_wd, dump_wd, path_wd):
        self.logger.config.env_name = 'testenv'
        self.logger.file_name = '1970-01-01_00:00.p'
        self.logger.start_episode()
        self.logger.finalize()
        open_wd.assert_called_with('testenv/1970-01-01_00:00.p', 'wb')
        dump_wd.assert_called_with(self.logger, unittest.mock.ANY)

    class SaveConfig:
        env_name = 'testenv'

    def test_save_not_mocked(self):
        self.logger.config = self.SaveConfig()
        self.logger.file_name = '1970-01-01_00:00.p'
        self.logger.root_directory = 'testlog'
        self.logger.start_episode()
        self.logger.finalize()
        self.assertTrue(os.path.exists(os.path.join(self.logger.root_directory,
                                                    self.logger.config.env_name)))
        shutil.rmtree(self.logger.root_directory)
        self.assertFalse(os.path.exists(self.logger.root_directory))


if __name__ == '__main__':
    unittest.main()
