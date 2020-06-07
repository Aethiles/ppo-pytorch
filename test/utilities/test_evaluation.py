import unittest

from source.utilities.evaluation import evaluate_policy
from source.environment.atari.setup import setup_environment
from source.policy.parameters import AtariParameters, AtariSize
from source.policy.policy import Policy
from unittest.mock import MagicMock


class EvaluatePolicyTest(unittest.TestCase):
    def test_evaluation_with_lives(self):
        env = setup_environment('BreakoutNoFrameskip-v4')
        params = AtariParameters(AtariSize.SMALL,
                                 input_shape=env.observation_space.shape,
                                 output_size=env.action_space.n,
                                 learn_rate=0,
                                 logger_=MagicMock())
        policy = Policy(params)
        score = evaluate_policy(env, policy, episodes=10)
        self.assertGreater(2.0, score)
        self.assertLessEqual(0, score)

    def test_evaluation_without_lives(self):
        env = setup_environment('PongNoFrameskip-v4')
        params = AtariParameters(AtariSize.SMALL,
                                 input_shape=env.observation_space.shape,
                                 output_size=env.action_space.n,
                                 learn_rate=0,
                                 logger_=MagicMock())
        policy = Policy(params)
        score = evaluate_policy(env, policy, episodes=10)
        self.assertGreaterEqual(-15, score)


if __name__ == '__main__':
    unittest.main()
