import unittest
import numpy as np

from source.environment.atari.noop_reset import NoopResetEnv
from test.helpers.environment import MockEnv, MockWrapper


class ResetTestCase(unittest.TestCase):
    def test_reset_no_ops_limits(self):
        mock = MockEnv(lambda: None,
                       lambda _: (None, None, False, {}),
                       ['NOOP'])
        iterations = 500
        for _ in range(iterations):
            env = NoopResetEnv(MockWrapper(mock), no_op_max=np.random.randint(1, 31))
            mock.step_ctr = 0
            env.reset()
            self.assertGreaterEqual(mock.step_ctr, 1)
            self.assertLessEqual(mock.step_ctr, env.no_op_max)
        self.assertEqual(mock.reset_ctr, iterations)

    def test_reset_state(self):
        state = np.arange(np.random.randint(4, 16))
        mock = MockEnv(lambda: state,
                       lambda _: (None, None, True, {}),
                       ['NOOP'])
        env = NoopResetEnv(MockWrapper(mock), no_op_max=1)
        ret = env.reset()
        self.assertTrue(np.array_equal(ret, state))
        self.assertEqual(mock.reset_ctr, 2)

    def test_multiple_resets(self):
        state = np.arange(np.random.randint(4, 16))
        mock = MockEnv(lambda: state,
                       lambda _: (None, None, True, {}),
                       ['NOOP'])
        env = NoopResetEnv(MockWrapper(mock))
        for _ in range(10):
            mock.step_ctr = 0
            mock.reset_ctr = 0
            env.reset()
            self.assertEqual(mock.step_ctr + 1, mock.reset_ctr)


class StepTestCase(unittest.TestCase):
    def test_step(self):
        for _ in range(50):
            state = np.random.rand(np.random.randint(4, 16))
            reward = np.random.rand(1)[0]
            done = np.random.randint(0, 2) == 0
            mock = MockEnv(lambda: 5,
                           lambda _: (state, reward, done, {}),
                           ['NOOP'])
            env = NoopResetEnv(MockWrapper(mock))
            ret = env.step(0)
            self.assertTrue(np.array_equal(ret[0], state))
            self.assertEqual(ret[1], reward)
            self.assertEqual(ret[2], done)


if __name__ == '__main__':
    unittest.main()
