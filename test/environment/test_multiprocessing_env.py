import numpy as np
import unittest

from source.environment.multiprocessing_env import MultiprocessingEnv, Command
from test.helpers.environment import MockEnv
from unittest.mock import MagicMock, patch


@patch('multiprocessing.Pipe')
@patch('multiprocessing.Process')
def setup_test(name, num_envs, seed, process_wd, pipe_wd):
    actual_pipes = MagicMock()
    env = MultiprocessingEnv(name, num_envs, seed)
    return env, process_wd, pipe_wd


class InitTest(unittest.TestCase):
    def test_init(self):
        for _ in range(10):
            num_envs = np.random.randint(2, 8)
            env, process_mock, pipe_mock = setup_test('test', num_envs, 0)
            self.assertEqual(num_envs, pipe_mock.call_count)
            self.assertEqual(num_envs, process_mock.return_value.start.call_count)
            for i, call in enumerate(process_mock.call_args_list):
                args, kwargs = call
                self.assertEqual(i, kwargs['args'][1])
            self.assertIsNotNone(env.observation_space)
            self.assertIsNotNone(env.action_space)


class StepTest(unittest.TestCase):
    def test_step(self):
        env, _, _ = setup_test('test', 10, 0)
        env._broadcast = MagicMock()
        env._gather = MagicMock()
        env._gather.return_value = [([0], [0], [0], [{}])] * 10
        ret = env.step(np.random.randint(0, 4, 10))
        self.assertEqual(4, len(ret))
        # TODO return
        # TODO called with STEP
        env._broadcast.assert_called()
        env._gather.assert_called()


class ResetTest(unittest.TestCase):
    def test_reset(self):
        env, _, _ = setup_test('test', 10, 0)
        env._broadcast = MagicMock()
        env._gather = MagicMock()
        env._gather.return_value = [[0]] * 10
        ret = env.reset()
        self.assertEqual(10, len(ret))
        # TODO return
        # TODO called with RESET
        env._broadcast.assert_called()
        env._gather.assert_called()


class CloseTest(unittest.TestCase):
    def test_close_mocked(self):
        env, _, _ = setup_test('test', 10, 0)
        env._broadcast = MagicMock()
        env.close()
        env._broadcast.assert_called_with(Command.CLOSE)

    def test_close(self):
        mock = MockEnv(lambda: None,
                       lambda _: (None, None, None, None))
        with patch('gym.make', return_value=mock):
            env = MultiprocessingEnv('test', 20, 0)
        env.close()
        for pipe, process in zip(env.pipes, env.processes):
            self.assertTrue(pipe.closed)
            self.assertFalse(process.is_alive())


if __name__ == '__main__':
    unittest.main()
