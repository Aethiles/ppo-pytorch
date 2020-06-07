import numpy as np
import unittest

from source.environment.atari.episodic_life import EpisodicLifeEnv
from test.helpers.environment import MockEnv, MockWrapper
from unittest.mock import MagicMock


def setup_env():
    state = np.arange(np.random.randint(4, 16))
    reward = np.random.rand(1)[0]
    mock = MockEnv(lambda: state,
                   lambda _: (state, reward, False, {}),
                   [''])
    mock.ale = MagicMock()
    mock.ale.lives.return_value = np.random.randint(3, 8)
    return EpisodicLifeEnv(MockWrapper(mock)), state, reward


class ResetTest(unittest.TestCase):
    def verify(self, ret, env, state, reset_ctr, step_ctr):
        self.assertTrue(np.array_equal(ret, state))
        self.assertEqual(env.lives, env.env.ale.lives.return_value)
        self.assertEqual(env.env.reset_ctr, reset_ctr)
        self.assertEqual(env.env.step_ctr, step_ctr)

    def test_true_reset(self):
        env, state, reward = setup_env()
        ret = env.reset()
        self.verify(ret, env, state, 1, 0)

    def test_simulated_reset(self):
        env, state, reward = setup_env()
        env.done = False
        ret = env.reset()
        self.verify(ret, env, state, 0, 1)


class StepTest(unittest.TestCase):
    def test_live_loss_simulated_done(self):
        env, state, reward = setup_env()
        env.reset()
        env.env.ale.lives.return_value -= 1
        ret = env.step(0)
        self.assertTrue(np.array_equal(ret[0], state))
        self.assertEqual(ret[1], reward)
        self.assertEqual(ret[2], True)
        self.assertEqual(env.lives, env.env.ale.lives.return_value)

    def test_regular_step(self):
        env, state, reward = setup_env()
        env.reset()
        ret = env.step(0)
        self.assertTrue(np.array_equal(ret[0], state))
        self.assertEqual(ret[1], reward)
        self.assertEqual(ret[2], False)


if __name__ == '__main__':
    unittest.main()
