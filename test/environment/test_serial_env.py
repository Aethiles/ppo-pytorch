import numpy as np
import source.environment.serial_env as serial_env
import unittest

from test.helpers.environment import MockEnv, MockWrapper
from test.helpers import utils
from unittest.mock import MagicMock, patch


def setup_test(num_envs, state_gen, reward_gen, done_gen, mock_setup=True):
    mock = MockWrapper(MockEnv(lambda: next(state_gen),
                               lambda a: (next(state_gen), next(reward_gen), next(done_gen), {}),
                               ))
    mock.seed = MagicMock()
    if mock_setup:
        setup_mock = MagicMock()
        setup_mock.return_value = mock
        with patch('test.environment.test_serial_env.serial_env.setup_environment',
                   return_value=setup_mock):
            envs = serial_env.SerialEnv('test_env', num_envs, seed=0)
    else:
        with patch('gym.make', return_value=mock):
            envs = serial_env.SerialEnv('NoFrameskip', num_envs, seed=0)
    return envs, mock


class SetupTest(unittest.TestCase):
    def test_one_env(self):
        envs, mock = setup_test(1,
                                utils.stable_state_gen(None),
                                utils.stable_reward_gen(0.0),
                                utils.stable_done_gen(False))
        self.assertEqual(1, len(envs.envs))

    def test_random_envs(self):
        n = np.random.randint(4, 16)
        envs, mock = setup_test(n,
                                utils.stable_state_gen(None),
                                utils.stable_reward_gen(0.0),
                                utils.stable_done_gen(False),
                                False)
        self.assertEqual(n, len(envs.envs))
        self.assertEqual(n, mock.seed.call_count)
        for i in range(n):
            args, kwargs = mock.seed.call_args_list[i]
            self.assertEqual(i, args[0])


class StepTest(unittest.TestCase):
    def test_one_step(self):
        n = np.random.randint(4, 16)
        state = np.ndarray((84, 84, 3), buffer=np.random.rand(3 * 84 * 84), dtype=np.float)
        envs, mock = setup_test(n,
                                utils.stable_state_gen(state),
                                utils.stable_reward_gen(1.),
                                utils.stable_done_gen(False),
                                False)
        states, rewards, dones, infos = envs.step(list(np.arange(n)))
        self.assertEqual(n * 4, mock.step_ctr)
        self.assertEqual(n, len(states))
        self.assertEqual(n, len(rewards))
        self.assertEqual(1, rewards[0])
        self.assertEqual(n, len(dones))

    def test_random_steps(self):
        n = np.random.randint(4, 16)
        state = np.ndarray((84, 84, 3), buffer=np.random.rand(3 * 84 * 84), dtype=np.float)
        envs, mock = setup_test(n,
                                utils.stable_state_gen(state),
                                utils.stable_reward_gen(1.),
                                utils.stable_done_gen(False),
                                False)
        for i in range(np.random.randint(10, 25)):
            states, rewards, dones, infos = envs.step(list(np.random.randint(0, n, size=n)))
            self.assertEqual((i + 1) * n * 4, mock.step_ctr)
            for r in rewards:
                self.assertEqual(1, r)

# TODO fix this test at a later point
"""
    def test_steps_some_envs_done(self):
        def done_gen():
            step = 0
            while True:
                if step % 2 == 0:
                    yield True
                else:
                    for _ in range(4):
                        yield False
                step += 1

        n = np.random.randint(2, 8) * 2
        state = np.ndarray((84, 84, 3), buffer=np.random.rand(3 * 84 * 84), dtype=np.float)
        envs, mock = setup_test(n,
                                utils.stable_state_gen(state),
                                utils.stable_reward_gen(1.),
                                done_gen(),
                                False)
        for env in envs.envs:
            env.env.env.env.env.env.env.no_op_max = 1
        mock.ale.lives.return_value = 0
        for _ in range(25):
            states, rewards, dones, infos = envs.step(list(np.random.randint(0, n, size=n)))
            for i in range(n):
                self.assertEqual(i % 2 == 0, dones[i])
                self.assertEqual((i % 2) * 3 + 1, rewards[i])
        self.assertEqual(n * 25, mock.step_ctr)
"""

class ResetTest(unittest.TestCase):
    def test_reset(self):
        n = np.random.randint(4, 16)
        state = np.ndarray((84, 84, 3), buffer=np.random.rand(3 * 84 * 84), dtype=np.float)
        envs, mock = setup_test(n,
                                utils.stable_state_gen(state),
                                utils.stable_reward_gen(1.),
                                utils.stable_done_gen(False),
                                False)
        envs.reset()
        self.assertEqual(n, mock.reset_ctr)
        self.assertGreaterEqual(n * 30, mock.step_ctr)
        self.assertLessEqual(n, mock.step_ctr)


if __name__ == '__main__':
    unittest.main()
