import numpy as np
import torch
import unittest

from source.training.ppo import ProximalPolicyOptimization
from source.training.rollout import Rollout
from source.utilities.maths import normalize_tensor
from unittest.mock import MagicMock


class LossFromRolloutWithArtificialValuesTest(unittest.TestCase):
    @staticmethod
    def setup_test(old_values, new_values, old_probs, new_probs, rewards, entropy, discount=1., gae_weight=1., clip=.2):
        policy_mock = MagicMock()
        policy_mock.device = torch.device('cpu')
        ppo = ProximalPolicyOptimization(policy_mock, MagicMock(), MagicMock(), num_envs=1,
                                         discount=discount, gae_weight=gae_weight, clip_range=clip)
        rollout = Rollout(1, 2, np.zeros(1), torch.device('cpu'))
        rollout.values = torch.tensor(old_values[:-1])
        rollout.log_action_probabilities = torch.log(torch.tensor(old_probs))
        rollout.rewards = torch.tensor(rewards)
        rollout.returns = ppo._calculate_returns(rollout.rewards,
                                                 torch.tensor(0),
                                                 torch.tensor([1.] * len(rollout.rewards)))
        rollout.advantages = normalize_tensor(ppo._calculate_advantages(rollout.rewards, torch.tensor(old_values),
                                                                        torch.tensor([1.] * len(rollout.rewards))))
        policy_mock.evaluate.return_value = (torch.log(torch.tensor(new_probs)),
                                             torch.tensor(new_values[:-1]),
                                             torch.tensor(entropy))
        return ppo, rollout

    def test_loss0(self):
        ppo, r = self.setup_test(old_values=[1., 1.], new_values=[1., 1.], old_probs=[0.5], new_probs=[1.],
                                 rewards=[1.], entropy=[1.], clip=0.1)
        loss = ppo.calculate_shared_parameters_loss(MagicMock(), MagicMock(), r.log_action_probabilities, r.values,
                                                    r.advantages, r.returns)
        self.assertTrue(torch.isnan(loss))

    def test_loss1(self):
        ppo, r = self.setup_test(old_values=[[-.2], [.4], [.8]], new_values=[[.1], [.5], [.8]], old_probs=[[.4], [.6]],
                                 new_probs=[[.5], [.9]], rewards=[[1.], [0]], entropy=[.2])
        loss = ppo.calculate_shared_parameters_loss(MagicMock(), MagicMock(), r.log_action_probabilities, r.values,
                                                    r.advantages, r.returns)
        # self.assertAlmostEqual(-0.41656, loss.item(), delta=1e-4)  # maximum in value function loss
        self.assertAlmostEqual(-0.3691, loss.item(), delta=1e-4)  # minimum in value function loss

    def test_loss2(self):
        ppo, r = self.setup_test(old_values=[[-.1], [-.5], [.9]], new_values=[[-.2], [-.3], [.8]],
                                 old_probs=[[.4], [.5]], new_probs=[[.3], [.35]], rewards=[[-1.], [1.]], entropy=[.3],
                                 discount=0.99, gae_weight=0.95)
        loss = ppo.calculate_shared_parameters_loss(MagicMock(), MagicMock(), r.log_action_probabilities, r.values,
                                                    r.advantages, r.returns)
        # self.assertAlmostEqual(-0.4639, loss.item(), delta=1e-4)  # normal return
        self.assertAlmostEqual(-0.45979, loss.item(), delta=1e-4)  # GAE return


# class LossFromRolloutOnMockEnvTest

if __name__ == '__main__':
    unittest.main()
