import numpy as np
import unittest

from source.environment.atari.reward_scaling import RewardScalingEnv
from test.helpers.environment import MockEnv, MockWrapper
from test.helpers.utils import list_reward_gen, stable_reward_gen


def setup_test(reward, discount):
    mock = MockEnv(lambda: None,
                   lambda _: (None, next(reward), False, {}))
    env = RewardScalingEnv(MockWrapper(mock), discount=discount)
    return mock, env


class UpdateTest(unittest.TestCase):
    def test_one_update(self):
        mock, env = setup_test(stable_reward_gen(1), 0)
        env._update_mean(1)
        self.assertEqual(1, env.rolling_sums[-1])
        self.assertEqual(1, len(env.rolling_sums))

    def test_two_updates0(self):
        mock, env = setup_test(stable_reward_gen(1), 0.9)
        env._update_mean(1)
        env._update_mean(1)
        self.assertEqual(1.9, env.rolling_sums[-1])
        self.assertEqual(2, len(env.rolling_sums))

    def test_two_updates1(self):
        mock, env = setup_test(stable_reward_gen(1), 0.9)
        env._update_mean(1)
        env._update_mean(0)
        self.assertEqual(0.9, env.rolling_sums[-1])

    def test_two_updates2(self):
        mock, env = setup_test(stable_reward_gen(1), 0.9)
        env._update_mean(0)
        env._update_mean(1)
        self.assertEqual(1, env.rolling_sums[-1])

    def test_three_updates(self):
        mock, env = setup_test(stable_reward_gen(1), 0.9)
        env._update_mean(1)
        env._update_mean(1)
        env._update_mean(1)
        self.assertEqual(2.71, env.rolling_sums[-1])


class StandardDeviationTest(unittest.TestCase):
    def test_no_scaling(self):
        mock, env = setup_test(stable_reward_gen(1), 0)
        ret = env.step(0)
        self.assertEqual(1, ret[1])

    def test_no_scaling_multiple_steps(self):
        mock, env = setup_test(stable_reward_gen(1), 0)
        for _ in range(np.random.randint(10, 20)):
            ret = env.step(0)
            self.assertEqual(1, ret[1])

    def test_scaling_multiple_steps(self):
        mock, env = setup_test(list_reward_gen([0, 1, 1]), 0.99)
        env.step(0)
        env.step(0)
        ret = env.step(0)
        self.assertAlmostEqual(1 / 0.81241751718, ret[1], delta=1e-8)


if __name__ == '__main__':
    unittest.main()
