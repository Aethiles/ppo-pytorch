import numpy as np
import torch
import unittest

from collections import defaultdict
from source.policy.policy import Policy
from test.helpers.parameters import TestParameters
from unittest.mock import MagicMock


class EvaluationTest(unittest.TestCase):
    @staticmethod
    def setup_test(input_size, output_size, num_states):
        pol = Policy(TestParameters(input_size=input_size,
                                    output_size=output_size))
        states = torch.zeros((num_states, input_size)).to(pol.device)
        actions = torch.zeros(num_states).to(pol.device)
        for i in range(num_states):
            states[i] = torch.from_numpy(np.arange(input_size))
        return pol, states, actions

    def test_value_shape_matching(self):
        pol, states, actions = self.setup_test(np.random.randint(4, 16),
                                               np.random.randint(2, 4),
                                               1)
        _, eval_vals, _ = pol.evaluate(states, actions)
        _, _, act_vals = pol.get_action_and_value(states)
        self.assertTrue(torch.eq(eval_vals, act_vals))

    def test_single_evaluation(self):
        pol, states, actions = self.setup_test(np.random.randint(4, 16),
                                               np.random.randint(2, 4),
                                               1)
        probs, vals, entropy = pol.evaluate(states, actions)
        self.assertEqual(states.shape[0], probs.shape[0])
        self.assertEqual(states.shape[0], vals.shape[0])

    def test_batch_evaluation(self):
        pol, states, actions = self.setup_test(np.random.randint(4, 16),
                                               np.random.randint(2, 4),
                                               np.random.randint(3, 9))
        probs, vals, entropy = pol.evaluate(states, actions)
        self.assertEqual(states.shape[0], probs.shape[0])
        self.assertEqual(states.shape[0], vals.shape[0])

    def test_batch_evaluation_length_mismatch(self):
        pol, states, actions = self.setup_test(np.random.randint(4, 16),
                                               np.random.randint(2, 4),
                                               np.random.randint(3, 9))
        self.assertRaises(Exception, pol.evaluate, (states, actions[:-1]))


class KLDivergenceTest(unittest.TestCase):
    @staticmethod
    def setup_policy(new_probs):
        input_mock = MagicMock()
        input_mock.shape = [0]
        pol = Policy(MagicMock())
        pol.evaluate = lambda a, b: (torch.tensor(new_probs), None, None)
        return pol, input_mock

    def test_same_distributions(self):
        pol, input_mock = self.setup_policy([0., 0., 1., -1.])
        kl_divergence = pol.calculate_kl_divergence(input_mock, input_mock, torch.tensor([0., 0., 1., -1.]))
        self.assertEqual(0, kl_divergence)

    def test_different_distributions0(self):
        pol, input_mock = self.setup_policy([0., 0., 0.])
        kl_divergence = pol.calculate_kl_divergence(input_mock, input_mock, torch.tensor([1., 1., -1.]))
        self.assertAlmostEqual(1/3, kl_divergence, delta=1e-8)

    def test_different_distributions1(self):
        pol, input_mock = self.setup_policy([1., 0., 0.])
        kl_divergence = pol.calculate_kl_divergence(input_mock, input_mock, torch.tensor([1., 1., -1.]))
        self.assertAlmostEqual(0, kl_divergence, delta=1e-8)

    def test_different_distributions2(self):
        pol, input_mock = self.setup_policy([1., 0., 1.])
        kl_divergence = pol.calculate_kl_divergence(input_mock, input_mock, torch.tensor([1., 1., -1.]))
        self.assertAlmostEqual(-1/3, kl_divergence, delta=1e-8)


class ActionAndValueTest(unittest.TestCase):
    input_size = np.random.randint(1, 8)
    output_size = np.random.randint(1, 5)
    # state = np.ndarray((input_size, 1), buffer=np.random.rand(input_size), dtype=np.float64)
    state = torch.randn((input_size, 1))

    def run_action_test(self, pol):
        action, log_prob, _ = pol.get_action_and_value(self.state)
        prob = np.e ** log_prob
        self.assertGreaterEqual(prob, 0)
        self.assertGreaterEqual(1, prob)
        self.assertIn(action.item(), np.arange(self.output_size))

    def test_action_value_matching_evaluation(self):
        pol = Policy(TestParameters(input_size=self.input_size,
                                    output_size=self.output_size))
        action, log_prob, val = pol.get_action_and_value(self.state)
        other_prob, other_val, _ = pol.evaluate(self.state.unsqueeze(0).to(pol.device),
                                                torch.tensor([action])
                                                .to(pol.device))
        self.assertAlmostEqual(log_prob.item(), other_prob.item(), delta=1e-8)
        self.assertAlmostEqual(val.item(), other_val.item(), delta=1e-8)

    def test_cpu_action(self):
        pol = Policy(TestParameters(input_size=self.input_size,
                                    output_size=self.output_size,
                                    device=torch.device('cpu')))
        for _ in range(100):
            self.run_action_test(pol)

    def test_gpu_action(self):
        if not torch.cuda.is_available():
            return
        pol = Policy(TestParameters(input_size=self.input_size,
                                    output_size=self.output_size,
                                    device=torch.device('cuda:0')))
        for _ in range(100):
            self.run_action_test(pol)

    def test_action_sampling(self):
        pol = Policy(TestParameters(input_size=self.input_size,
                                    output_size=self.output_size))
        actions = defaultdict(int)
        probs = {}
        for i in range(10000):
            action, log_prob, _ = pol.get_action_and_value(self.state)
            action = action[0].item()
            actions[action] += 1
            if action not in probs:
                probs[action] = log_prob.item()
        for i in range(self.output_size):
            self.assertAlmostEqual(actions[i] / 10000,
                                   np.e**probs[i],
                                   delta=0.01)


if __name__ == '__main__':
    unittest.main()
