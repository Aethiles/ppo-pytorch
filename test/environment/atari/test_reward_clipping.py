import numpy as np
import unittest

from source.environment.atari.reward_clipping import ClipRewardEnv, ClippingMode
from test.helpers.environment import MockEnv, MockWrapper


def setup_env(reward, mode, **kwargs):
    mock = MockEnv(lambda: None,
                   lambda _: (None, reward, False, {}),
                   [''])
    env = ClipRewardEnv(MockWrapper(mock), mode, **kwargs)
    return env


class ClipRewardTest(unittest.TestCase):
    def test_clipping_reward_in_interval(self):
        for _ in range(100):
            rew = np.random.randint(-5, 5)
            env = setup_env(rew, ClippingMode.CLIP)
            ret = env.step(0)
            self.assertEqual(ret[1], rew)

    def test_clipping_large_positive_reward_to_default(self):
        env = setup_env(6, ClippingMode.CLIP)
        ret = env.step(0)
        self.assertEqual(ret[1], 5)

    def test_clipping_large_negative_reward_to_default(self):
        env = setup_env(-6, ClippingMode.CLIP)
        ret = env.step(0)
        self.assertEqual(ret[1], -5)

    def test_clipping_large_positive_reward(self):
        env = setup_env(6, ClippingMode.CLIP, **{'min': -3, 'max': 3})
        ret = env.step(0)
        self.assertEqual(ret[1], 3)

    def test_clipping_large_negative_reward(self):
        env = setup_env(-6, ClippingMode.CLIP, **{'min': -3, 'max': 3})
        ret = env.step(0)
        self.assertEqual(ret[1], -3)


class BinRewardTest(unittest.TestCase):
    def test_zero_reward(self):
        env = setup_env(0, ClippingMode.BIN)
        ret = env.step(0)
        self.assertEqual(ret[1], 0)

    def test_positive_reward(self):
        for i in range(50):
            env = setup_env(np.random.rand(1)[0] * np.random.randint(1, 10), ClippingMode.BIN)
            ret = env.step(0)
            self.assertEqual(ret[1], 1)

    def test_negative_reward(self):
        for i in range(50):
            env = setup_env(np.random.rand(1)[0] * -np.random.randint(1, 10), ClippingMode.BIN)
            ret = env.step(0)
            self.assertEqual(ret[1], -1)


if __name__ == '__main__':
    unittest.main()
