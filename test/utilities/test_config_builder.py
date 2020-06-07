import source.utilities.config.config_builder as config_builder
import unittest

from source.policy.parameters import AtariSize
from source.utilities.config.hyperparameters import HyperParameters
from unittest.mock import patch


class FullTest(unittest.TestCase):
    @patch('socket.gethostname')
    def test_get_hyperparameters(self, socket_wd):
        socket_wd.return_value = 'cosy04'
        hyperparameters = config_builder.get_hyperparameters()
        self.assertEqual(17, hyperparameters.ppo_epochs)
        self.assertEqual(1, hyperparameters.nn_learn_rate)
        self.assertEqual('TestNoFrameskip-v4', hyperparameters.env_name)


class GetGroupTest(unittest.TestCase):
    @patch('socket.gethostname')
    def test_all_groups(self, hostname_wd):
        expected = {0: [1, 2, 3],
                    1: [4, 5, 6],
                    2: [7, 8, 9],
                    3: [10, 11, 12],
                    4: [13, 14, 15]}
        for i in range(15):
            name = 'cosy0' + str(i + 1)
            hostname_wd.return_value = name
            group = config_builder.get_group()
            self.assertTrue((i + 1) in expected[group])

    def test_host_name_not_cosy(self):
        group = config_builder.get_group()
        self.assertEqual(0, group)


class UpdateParametersTest(unittest.TestCase):
    def test_atari_size_exception(self):
        config = {'nn_size': 'LARGE'}
        hyperparameters = HyperParameters('')
        config_builder._update_parameters(hyperparameters, config)
        self.assertEqual(AtariSize.LARGE, hyperparameters.nn_size)


class GetHyperparametersTest(unittest.TestCase):
    config = {'env_name': 'SpaceInvaders-v4',
              'nn_learn_rate': '0.05'}

    @patch('source.utilities.config.config_builder.open_config')
    def test_config_structure(self, config_wd):
        config_wd.return_value = self.config
        config = config_builder.get_hyperparameters()
        self.assertEqual('SpaceInvadersNoFrameskip-v4', config.env_name)
        self.assertEqual(0.05, config.nn_learn_rate)
        self.assertEqual(3, config.ppo_epochs)


class InitTest(unittest.TestCase):
    def test_init_incomplete_name(self):
        result = config_builder._init_hyperparameters('Test-v4')
        self.assertEqual('TestNoFrameskip-v4', result.env_name)

    def test_init_complete_name(self):
        result = config_builder._init_hyperparameters('TestNoFrameskip-v1')
        self.assertEqual('TestNoFrameskip-v1', result.env_name)


if __name__ == '__main__':
    unittest.main()
