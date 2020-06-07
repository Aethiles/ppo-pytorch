import numpy as np
import unittest
import source.environment.atari.setup as setup

from test.helpers.environment import MockEnv, MockWrapper
from unittest.mock import MagicMock, patch


class FullWrapperTest(unittest.TestCase):
    @staticmethod
    def setup_test(reward, done, info, meanings=None, state=np.ndarray((84, 84, 3),
                                                                       buffer=np.random.rand(84 * 84 * 3),
                                                                       dtype=np.float)):
        if meanings is None:
            meanings = ['NOOP', 'FIRE', 'FOO', 'BAR']
        mock = MockEnv(lambda: state,
                       lambda _: (state, reward, done, info),
                       meanings)
        mock.spec = MagicMock()
        mock.spec.id = 'NoFrameskip'
        mock.observation_space = np.zeros((84, 84, 3))
        mock.ale = MagicMock()
        mock.ale.lives.return_value = 5
        with patch('gym.make', return_value=MockWrapper(mock)):
            return setup.setup_environment('')

    def verify_shape_and_counters(self, shape, steps, resets, state, env):
        self.assertEqual(shape, state.shape)
        self.assertGreaterEqual(steps[0], env.unwrapped.step_ctr)
        self.assertLessEqual(steps[1], env.unwrapped.step_ctr)
        self.assertEqual(resets, env.unwrapped.reset_ctr)

    def test_full_reset(self):
        for i in range(100):
            env = self.setup_test(reward=125.0, done=False, info={})
            ret = env.reset()
            self.verify_shape_and_counters((4, 84, 84), (38, 9), 1, ret, env)

    def test_reset_without_done(self):
        env = self.setup_test(reward=125.0, done=False, info={})
        env.reset()
        env.unwrapped.step_ctr = 0
        ret = env.reset()
        self.verify_shape_and_counters((4, 84, 84), (12, 12), 1, ret, env)

    def test_step_positive_reward(self):
        env = self.setup_test(reward=125.0, done=False, info={})
        env.reset()
        env.unwrapped.step_ctr = 0
        ret = env.step(0)
        self.verify_shape_and_counters((4, 84, 84), (4, 4), 1, ret[0], env)
        self.assertEqual(1.0, ret[1])
        self.assertFalse(ret[2])

    def test_negative_reward(self):
        env = self.setup_test(reward=-125.0, done=False, info={})
        env.reset()
        env.unwrapped.step_ctr = 0
        ret = env.step(0)
        self.verify_shape_and_counters((4, 84, 84), (4, 4), 1, ret[0], env)
        self.assertEqual(-1.0, ret[1])
        self.assertFalse(ret[2])

    def test_true_done(self):
        env = self.setup_test(reward=0, done=True, info={})
        env.env.env.env.env.env.env.env.no_op_max = 1
        env.reset()
        # Expected: 6 = 2 resets from NoopResetEnv per reset from FireResetEnv (3 total)
        self.assertEqual(6, env.unwrapped.reset_ctr)
        ret = env.step(0)
        self.assertTrue(ret[2])
        env.reset()
        self.assertEqual(12, env.unwrapped.reset_ctr)


if __name__ == '__main__':
    unittest.main()
