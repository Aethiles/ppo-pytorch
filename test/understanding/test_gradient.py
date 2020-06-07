import numpy as np
import torch
import unittest

from source.policy.parameters import AtariParameters, AtariSize
from source.policy.policy import Policy
from test.helpers.parameters import TestParameters
from typing import Tuple
from unittest.mock import MagicMock


class GradientUpdateTest(unittest.TestCase):
    @staticmethod
    def setup_test() -> Tuple[torch.Tensor, Policy, TestParameters, int]:
        input_size = np.random.randint(4, 8)
        output_size = np.random.randint(2, 3)
        state = torch.randn((input_size, 1))
        params: TestParameters = TestParameters(input_size=input_size,
                                                output_size=output_size,
                                                device=torch.device('cpu'))
        pol = Policy(params)
        return state, pol, params, output_size

    def test_single_weight_update(self):
        state, pol, params, _ = self.setup_test()
        old_weights = pol.parameters.linear1.weight.data.clone().cpu()
        action, log_prob, _ = pol.get_action_and_value(state)
        loss = torch.cat([np.e**log_prob * -1]).sum().to(pol.device)
        params.gradient_update(loss)
        new_weights = pol.parameters.linear1.weight.data.clone().cpu()
        self.assertFalse(old_weights.equal(new_weights))

    def test_multiple_weight_updates(self):
        state, pol, params, _ = self.setup_test()
        old_weights = pol.parameters.linear1.weight.data.clone().cpu()
        times = np.random.randint(5, 11)
        for _ in range(times):
            action, log_prob, _ = pol.get_action_and_value(state)
            loss = torch.cat([-log_prob * -1]).sum().to(pol.device)
            params.gradient_update(loss)
        self.assertEqual(times, params.forward_ctr)
        self.assertEqual(times, params.update_ctr)
        new_weights = pol.parameters.linear1.weight.data.clone().cpu()
        self.assertFalse(old_weights.equal(new_weights))

    def test_policy_improvement(self):
        state, pol, params, output_size = self.setup_test()
        state_tensor = state.to(pol.device)
        initial_probs = pol.parameters(state_tensor)[0].cpu()
        rewards = []
        log_probs = []
        good_action = np.random.randint(0, output_size)
        for i in range(np.random.randint(5, 11)):
            action, log_prob, _ = pol.get_action_and_value(state)
            reward = -1
            if action == good_action:
                reward = 1
            rewards.append(reward)
            log_probs.append(log_prob)
        losses = [-l * r for r, l in zip(rewards, log_probs)]
        loss = torch.cat(losses).sum().to(pol.device)
        params.gradient_update(loss)
        self.assertEqual(1, params.update_ctr)
        probs = pol.parameters(state_tensor)[0].cpu()
        for action in range(output_size):
            if action == good_action:
                self.assertGreater(probs[0][action].item(), initial_probs[0][action].item())
            else:
                self.assertGreaterEqual(initial_probs[0][action].item(), probs[0][action].item())


class ExplodingValueTest(unittest.TestCase):
    @staticmethod
    def setup(lr, target) -> Tuple[AtariParameters, torch.Tensor, torch.Tensor]:
        state = np.ndarray((4, 84, 84), buffer=np.random.rand(4 * 84 * 84), dtype=np.float)
        params = AtariParameters(AtariSize.SMALL, (4, 84, 84), 4, lr, logger_=MagicMock())
        state = torch.from_numpy(state).unsqueeze(0).float().to(params.device)
        target = torch.tensor([target]).to(params.device)
        return params, state, target

    def test_explosions(self):
        for _ in range(100):
            params, state, target = self.setup(lr=1.0*10**-1, target=10)
            action_probs, value = params(state)
            loss = 0.5 * (value - target) ** 2
            params.gradient_update(loss)
            _, new_value = params(state)
            self.assertLess(1000, torch.abs(new_value / value).item())

    def test_stable_update(self):
        for _ in range(100):
            params, state, target = self.setup(lr=1.0*10**-4, target=10)
            action_probs, value = params(state)
            loss = 0.5 * (value - target) ** 2
            params.gradient_update(loss)
            _, new_value = params(state)
            self.assertGreater(10, torch.dist(new_value, value).item())


if __name__ == '__main__':
    unittest.main()
