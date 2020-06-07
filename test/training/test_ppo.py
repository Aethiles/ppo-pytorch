import numpy as np
import torch
import unittest

from source.training.ppo import ProximalPolicyOptimization
from source.utilities.maths import normalize_tensor
from unittest.mock import MagicMock, patch


class AdvantageTest(unittest.TestCase):
    @staticmethod
    def gae(rewards, values, masks, workers, gamma=0.99, lambda_=0.95):
        # sum_l^infinity: (γλ)**l * δ^V_{t+l}
        # δ^V_{t} := -V(s_t) + r_t + γV(s_{t+1}) TD residual
        adv = np.zeros((len(rewards), workers))
        for t in range(len(rewards)):
            sum_ = 0
            for k in range(len(rewards[t:])):
                sum_ += (gamma * lambda_) ** k * (rewards[t + k] + gamma * values[t + k + 1] - values[t + k])
            adv[t] = sum_
        return adv

    def setup_test_and_result(self, rewards, values, masks, discount=0.99, gae_weight=0.95, num_workers=1):
        ppo = ProximalPolicyOptimization(MagicMock(), MagicMock(), MagicMock(),
                                         num_envs=num_workers, discount=discount, gae_weight=gae_weight)
        expected = self.gae(np.ndarray((len(rewards), num_workers), buffer=np.array(rewards), dtype=float),
                            np.ndarray((len(values), num_workers), buffer=np.array(values), dtype=float),
                            np.ndarray((len(masks), num_workers), buffer=np.array(masks), dtype=float),
                            workers=num_workers, gamma=discount, lambda_=gae_weight)
        return ppo, torch.tensor(rewards), torch.tensor(values), torch.tensor(masks), expected

    def test_kostrikov0(self):
        ppo, rewards, values, masks, _ = self.setup_test_and_result(
            rewards=[[[0.], [0.]], [[0.], [0.]], [[0.], [0.]], [[0.], [0.]], [[0.], [0.]], [[0.], [0.]], [[0.], [0.]],
                     [[0.], [0.]], [[0.], [0.]], [[0.], [0.]], [[0.], [0.]], [[0.], [0.]]],
            values=[[[0.0328], [0.0313]], [[0.0881], [0.0488]], [[0.0765], [0.0300]], [[0.0462], [0.0566]],
                    [[0.0319], [0.0606]], [[0.0341], [0.0490]], [[0.0547], [0.0566]], [[0.0499], [0.0454]],
                    [[0.0452], [0.0399]], [[0.0461], [0.0315]], [[0.0599], [0.0333]], [[0.0266], [0.0268]],
                    [[0.0322], [0.0154]]],
            masks=[[[1.], [1.]], [[1.], [1.]], [[1.], [1.]], [[1.], [1.]], [[1.], [1.]], [[1.], [1.]], [[1.], [1.]],
                   [[1.], [1.]], [[1.], [1.]], [[1.], [1.]], [[1.], [1.]], [[1.], [1.]]],
            num_workers=2
        )
        result_kostrikov = torch.tensor(
            [[[0.0050], [-0.0053]], [[-0.0526], [-0.0238]], [[-0.0428], [-0.0049]], [[-0.0128], [-0.0330]],
             [[0.0020], [-0.0386]], [[0.0002], [-0.0282]], [[-0.0212], [-0.0375]], [[-0.0169], [-0.0274]],
             [[-0.0124], [-0.0229]], [[-0.0137], [-0.0151]], [[-0.0286], [-0.0176]], [[0.0053], [-0.0116]]]
        ).squeeze()
        result_kostrikov_norm = (result_kostrikov - result_kostrikov.mean()) / (result_kostrikov.std() + 1e-5)

        advantages = ppo._calculate_advantages(rewards.squeeze(), values.squeeze(), masks.squeeze())
        advantages_norm = normalize_tensor(advantages)

        self.assertTrue(torch.allclose(advantages_norm, result_kostrikov_norm, atol=1e-2))

    def test_kostrikov1(self):
        ppo, rewards, values, masks, _ = self.setup_test_and_result(
            rewards=[[[0.], [0.]], [[0.], [0.]], [[0.], [0.]], [[0.], [0.]], [[0.], [0.]], [[0.], [1.]], [[0.], [0.]],
                     [[0.], [0.]], [[0.], [0.]], [[0.], [0.]], [[0.], [0.]], [[0.], [0.]]],
            values=[[[-0.0125], [0.0984]], [[0.0705], [0.0931]], [[0.0966], [0.0898]], [[0.0945], [0.0994]],
                    [[0.0970], [0.0971]], [[0.0962], [0.0898]], [[0.0993], [0.0967]], [[0.0980], [0.0928]],
                    [[0.1028], [0.0886]], [[0.0954], [0.0977]], [[0.0951], [0.1036]], [[0.0993], [0.0979]],
                    [[0.0990], [0.0935]]],
            masks=[[[1.], [1.]], [[1.], [1.]], [[1.], [1.]], [[1.], [1.]], [[1.], [1.]], [[1.], [1.]], [[1.], [1.]],
                   [[1.], [1.]], [[1.], [1.]], [[1.], [1.]], [[1.], [1.]], [[1.], [1.]]],
            num_workers=2
        )

        result_kostrikov = torch.tensor(
            [[[1.0081e-01], [7.2330e-01]], [[1.9666e-02], [7.7566e-01]], [[-5.7922e-03], [8.2927e-01]],
             [[-2.9217e-03], [8.7254e-01]], [[-4.7361e-03], [9.3123e-01]], [[-3.1879e-03], [9.9888e-01]],
             [[-5.5878e-03], [-7.5568e-03]], [[-3.5482e-03], [-2.8363e-03]], [[-7.8167e-03], [2.3921e-03]],
             [[6.2269e-04], [-6.1539e-03]], [[1.9483e-03], [-1.1723e-02]], [[-1.2969e-03], [-5.3525e-03]]]
        ).squeeze()
        result_kostrikov_norm = (result_kostrikov - result_kostrikov.mean()) / (result_kostrikov.std() + 1e-5)

        advantages = ppo._calculate_advantages(rewards.squeeze(), values.squeeze(), masks.squeeze())
        advantages_norm = normalize_tensor(advantages)

        self.assertTrue(torch.allclose(advantages_norm, result_kostrikov_norm, atol=1e-3))

    def test_two_actors_single_advantage0(self):
        ppo, rewards, values, masks, expected = self.setup_test_and_result([[1., 1.]], [[1., 1.], [1., 1.]],
                                                                           [[1., 1.], [1., 1.]], discount=0.99,
                                                                           gae_weight=0.95, num_workers=2)

        advantages = ppo._calculate_advantages(rewards, values, masks)
        self.assertEqual(torch.Size([1, 2]), advantages.shape)
        self.assertAlmostEqual(0.99, advantages[0][0].item(), delta=1e-8)
        self.assertAlmostEqual(0.99, advantages[0][1].item(), delta=1e-8)

    def test_two_actors_single_advantage1(self):
        ppo, rewards, values, masks, expected = self.setup_test_and_result([[0.5, -0.5]], [[-1., 0.], [1., -0.5]],
                                                                           [[1., 1.], [1., 1.]], discount=0.99,
                                                                           gae_weight=0.95, num_workers=2)
        advantages = ppo._calculate_advantages(rewards, values, masks)
        self.assertEqual(torch.Size([1, 2]), advantages.shape)
        self.assertAlmostEqual(2.49, advantages[0][0].item(), delta=1e-8)
        self.assertAlmostEqual(-0.995, advantages[0][1].item(), delta=1e-8)

    def test_two_actors_two_advantages(self):
        ppo, rewards, values, masks, expected = self.setup_test_and_result([[0., 1.], [0., 1.]],
                                                                           [[0., 1.], [0., 1.], [-1., 1.]],
                                                                           [[1., 1.], [1., 1.]],
                                                                           discount=0.99, gae_weight=0.95,
                                                                           num_workers=2)
        self.assertEqual(expected[1][0], -0.99)
        self.assertEqual(expected[0][0], -0.931095)
        self.assertEqual(expected[1][1], 0.99)
        self.assertEqual(expected[0][1], 1.921095)
        advantages = ppo._calculate_advantages(rewards, values, masks)
        self.assertEqual(torch.Size([2, 2]), advantages.shape)
        self.assertAlmostEqual(-0.99, advantages[1][0].item(), delta=1e-6)
        self.assertAlmostEqual(-0.931095, advantages[0][0].item(), delta=1e-6)
        self.assertAlmostEqual(0.99, advantages[1][1].item(), delta=1e-6)
        self.assertAlmostEqual(1.921095, advantages[0][1].item(), delta=1e-6)

    def test_four_actors_ten_advantages(self):
        for _ in range(100):
            ppo, rewards, values, masks, expected = self.setup_test_and_result(np.random.rand(10, 4),
                                                                               np.random.rand(11, 4),
                                                                               [[1., 1., 1., 1.]] * 10,
                                                                               *np.random.rand(2), num_workers=4)
            advantages = ppo._calculate_advantages(rewards, values, masks)
            self.assertEqual(torch.Size([10, 4]), advantages.shape)
            for exp, adv in zip(expected, advantages):
                self.assertTrue(torch.allclose(torch.from_numpy(exp).float(), adv, atol=1e-5))

    def test_single_advantage0(self):
        ppo, rewards, values, masks, expected = self.setup_test_and_result([1.], [1., 1.], [1.])
        self.assertEqual(0.99, expected[0][0])
        advantages = ppo._calculate_advantages(rewards, values, masks)
        self.assertAlmostEqual(expected[0][0], advantages.item(), 7)

    def test_single_advantage1(self):
        ppo, rewards, values, masks, expected = self.setup_test_and_result([0.], [0., 0.], [1.])
        self.assertEqual(0, expected[0][0])
        advantages = ppo._calculate_advantages(rewards, values, masks)
        self.assertAlmostEqual(expected[0][0], advantages.item(), 7)

    def test_single_advantage2(self):
        ppo, rewards, values, masks, expected = self.setup_test_and_result([-1.], [0., 0.], [1.])
        self.assertEqual(-1, expected[0][0])
        advantages = ppo._calculate_advantages(rewards, values, masks)
        self.assertAlmostEqual(expected[0][0], advantages.item(), 7)

    def test_single_advantage3(self):
        ppo, rewards, values, masks, expected = self.setup_test_and_result([0.5], [-1., 1.], [1.])
        self.assertEqual(2.49, expected[0][0])
        advantages = ppo._calculate_advantages(rewards, values, masks)
        self.assertAlmostEqual(expected[0][0], advantages.item(), 7)

    def test_single_advantage4(self):
        ppo, rewards, values, masks, expected = self.setup_test_and_result([-0.5], [0., -0.5], [1.])
        self.assertEqual(-0.995, expected[0][0])
        advantages = ppo._calculate_advantages(rewards, values, masks)
        self.assertAlmostEqual(expected[0][0], advantages.item(), 7)

    def test_single_advantage_modified_discount(self):
        ppo, rewards, values, masks, expected = self.setup_test_and_result([1.], [1., 1.], [1.], discount=0.5)
        self.assertEqual(0.5, expected[0][0])
        advantages = ppo._calculate_advantages(rewards, values, masks)
        self.assertAlmostEqual(expected[0][0], advantages.item(), 7)

    def test_single_advantage_randomized(self):
        for _ in range(100):
            ppo, rewards, values, masks, expected = self.setup_test_and_result(np.random.rand(1), np.random.rand(2),
                                                                               [1.], discount=np.random.rand(1)[0])
            advantages = ppo._calculate_advantages(rewards, values, masks)
            self.assertAlmostEqual(expected[0][0], advantages.item(), 4)

    def test_two_advantages_masked0(self):
        ppo, rewards, values, masks, _ = self.setup_test_and_result([1., 1.], [1., 1., 1.], [0., 0.])
        advantages = ppo._calculate_advantages(rewards, values, masks)
        self.assertAlmostEqual(0., advantages[0][0].item(), 4)
        self.assertAlmostEqual(0., advantages[1][0].item(), 4)

    def test_two_advantages_masked1(self):
        ppo, rewards, values, masks, _ = self.setup_test_and_result([-1., 1.], [1., 1., 1.], [0., 0.])
        advantages = ppo._calculate_advantages(rewards, values, masks)
        self.assertAlmostEqual(-2., advantages[0][0].item(), 4)
        self.assertAlmostEqual(0., advantages[1][0].item(), 4)

    def test_five_advantages_masked_two_actors(self):
        ppo, rewards, values, masks, _ = self.setup_test_and_result([[-1., 1], [1., 1.], [1., -1.], [-1., -1.],
                                                                     [-1., 1.]],
                                                                    [[-1., 1.], [1., 1.], [1., 1.], [-1., -1.],
                                                                     [1., -1.], [-1., -1]],
                                                                    [[1., 1.], [0., 0.], [1., 1.], [1., 0.], [1., 1.]],
                                                                    discount=0.5,
                                                                    num_workers=2)
        advantages = ppo._calculate_advantages(rewards, values, masks)
        expected = torch.tensor([[0.5, 0.5],
                                 [0., 0.],
                                 [-0.8266, -2.5],
                                 [-0.6875, -0.],
                                 [-2.5, 1.5]]).float()
        for i in range(expected.shape[0]):
            for j in range(expected.shape[1]):
                self.assertAlmostEqual(expected[i][j].item(), advantages[i][j].item(), delta=1e-4)

    def test_two_advantages0(self):
        ppo, rewards, values, masks, expected = self.setup_test_and_result([1., 1.], [1., 1., 1.], [1., 1.])
        self.assertEqual(expected[1][0], 0.99)
        self.assertEqual(expected[0][0], 1.921095)
        advantages = ppo._calculate_advantages(rewards, values, masks)
        self.assertAlmostEqual(expected[0][0], advantages[0][0].item(), 4)
        self.assertAlmostEqual(expected[1][0], advantages[1][0].item(), 4)

    def test_two_advantages1(self):
        ppo, rewards, values, masks, expected = self.setup_test_and_result([0., 0.], [0., 0., 0.], [1., 1.])
        self.assertEqual(expected[1][0], 0)
        self.assertEqual(expected[0][0], 0)
        advantages = ppo._calculate_advantages(rewards, values, masks)
        self.assertAlmostEqual(expected[0][0], advantages[0][0].item(), 4)
        self.assertAlmostEqual(expected[1][0], advantages[1][0].item(), 4)

    def test_two_advantages2(self):
        ppo, rewards, values, masks, expected = self.setup_test_and_result([0., 0.], [0., 0., -1.], [1., 1.])
        self.assertEqual(expected[1][0], -0.99)
        self.assertEqual(expected[0][0], -0.931095)
        advantages = ppo._calculate_advantages(rewards, values, masks)
        self.assertAlmostEqual(expected[0][0], advantages[0][0].item(), 4)
        self.assertAlmostEqual(expected[1][0], advantages[1][0].item(), 4)

    def test_two_advantages3(self):
        ppo, rewards, values, masks, expected = self.setup_test_and_result([-0.5, 0.3], [0.15, 0.9, -0.2], [1., 1.],
                                                                           discount=0.7, gae_weight=0.8)
        self.assertAlmostEqual(expected[1][0], -0.74, delta=1e-4)
        self.assertAlmostEqual(expected[0][0], -0.4343, delta=1e-4)
        advantages = ppo._calculate_advantages(rewards, values, masks)
        self.assertAlmostEqual(expected[0][0], advantages[0][0].item(), delta=1e-4)
        self.assertAlmostEqual(expected[1][0], advantages[1][0].item(), delta=1e-4)

    def test_two_advantages_randomized(self):
        for _ in range(100):
            ppo, rewards, values, masks, expected = self.setup_test_and_result(np.random.rand(2), np.random.rand(3),
                                                                               [1., 1.], *np.random.rand(2))
            advantages = ppo._calculate_advantages(rewards, values, masks)
            self.assertAlmostEqual(expected[0][0], advantages[0][0].item(), delta=1e-4)
            self.assertAlmostEqual(expected[1][0], advantages[1][0].item(), delta=1e-4)

    def test_ten_advantages0(self):
        ppo, rewards, values, masks, expected = self.setup_test_and_result([0.] * 10, [0.] * 11, [1.] * 10)
        self.assertTrue(np.array_equal(expected, np.ndarray((10, 1), buffer=np.array([0.] * 10), dtype=float)))
        advantages = ppo._calculate_advantages(rewards, values, masks)
        self.assertTrue(np.array_equal(expected, advantages))

    def test_ten_advantages1(self):
        ppo, rewards, values, masks, expected = self.setup_test_and_result([1.] * 10, [1.] * 11, [1.] * 10)
        self.assertAlmostEqual(7.62904, expected[0][0], delta=1e-4)
        self.assertEqual(0.99, expected[-1][0])
        advantages = ppo._calculate_advantages(rewards, values, masks)
        for adv, exp in zip(advantages, expected):
            self.assertAlmostEqual(exp.flatten(), adv.item(), delta=1e-4)

    def test_ten_advantages_randomized(self):
        for _ in range(100):
            ppo, rewards, values, masks, expected = self.setup_test_and_result(np.random.rand(10), np.random.rand(11),
                                                                               [1.] * 10, *np.random.rand(2))
            advantages = ppo._calculate_advantages(rewards, values, masks)
            for exp, adv in zip(expected, advantages):
                self.assertAlmostEqual(exp.flatten(), adv.item(), delta=1e-4)


class ClippedLossTest(unittest.TestCase):
    @staticmethod
    def setup_test(adv, new_probs, old_probs, clip_range=0.2):
        ppo = ProximalPolicyOptimization(MagicMock(), MagicMock(), MagicMock(), clip_range=clip_range, num_envs=1)
        return ppo, torch.tensor(adv), torch.log(torch.tensor(new_probs)), torch.log(torch.tensor(old_probs))

    def test_single_loss0(self):
        ppo, adv, new_probs, old_probs = self.setup_test([1.], [1.], [1.])
        loss = ppo.calculate_policy_loss(adv, new_probs, old_probs)
        self.assertEqual(1, loss.item())

    def test_single_loss1(self):
        ppo, adv, new_probs, old_probs = self.setup_test([0.], [1.], [1.])
        loss = ppo.calculate_policy_loss(adv, new_probs, old_probs)
        self.assertEqual(0, loss.item())

    def test_single_loss2(self):
        ppo, adv, new_probs, old_probs = self.setup_test([1.], [1.], [0.5], 0.1)
        loss = ppo.calculate_policy_loss(adv, new_probs, old_probs)
        self.assertAlmostEqual(1.1, loss.item(), delta=1e-6)

    def test_single_loss3(self):
        ppo, adv, new_probs, old_probs = self.setup_test([-1.], [1.], [0.5], 0.1)
        loss = ppo.calculate_policy_loss(adv, new_probs, old_probs)
        self.assertEqual(-2, loss.item())

    def test_single_item_loss(self):
        n = np.random.randint(3, 8)
        ppo, adv, new_probs, old_probs = self.setup_test(np.random.rand(n), np.random.rand(n), np.random.rand(n))
        loss = ppo.calculate_policy_loss(adv, new_probs, old_probs)
        self.assertEqual(torch.Size([]), loss.shape)

    def test_two_actors_single_loss0(self):
        ppo, adv, new_probs, old_probs = self.setup_test([[1., 1.]], [[1., 1.]], [[1., 1.]])
        loss = ppo.calculate_policy_loss(adv, new_probs, old_probs)
        self.assertAlmostEqual(1, loss.item(), delta=1e-6)

    def test_two_actors_single_loss1(self):
        ppo, adv, new_probs, old_probs = self.setup_test([[1., .5]], [[1., 1.]], [[.5, .5]])
        loss = ppo.calculate_policy_loss(adv, new_probs, old_probs)
        self.assertAlmostEqual(0.9, loss.item(), delta=1e-6)

    def test_two_losses0(self):
        ppo, adv, new_probs, old_probs = self.setup_test([1., 1.], [1., 1.], [1., 1.])
        loss = ppo.calculate_policy_loss(adv, new_probs, old_probs)
        self.assertAlmostEqual(1, loss.item(), delta=1e-6)

    def test_two_losses1(self):
        ppo, adv, new_probs, old_probs = self.setup_test([0., 0.], [1., 1.], [1., 1.])
        loss = ppo.calculate_policy_loss(adv, new_probs, old_probs)
        self.assertAlmostEqual(0, loss.item(), delta=1e-6)

    def test_two_losses2(self):
        ppo, adv, new_probs, old_probs = self.setup_test([1., 0.], [1., 1.], [1., 1.])
        loss = ppo.calculate_policy_loss(adv, new_probs, old_probs)
        self.assertAlmostEqual(0.5, loss.item(), delta=1e-6)

    def test_two_losses3(self):
        ppo, adv, new_probs, old_probs = self.setup_test([-1., 0.], [1., 1.], [1., 1.])
        loss = ppo.calculate_policy_loss(adv, new_probs, old_probs)
        self.assertAlmostEqual(-0.5, loss.item(), delta=1e-6)

    def test_two_losses4(self):
        ppo, adv, new_probs, old_probs = self.setup_test([1., 0.5], [1., 1.], [0.5, 0.5])
        loss = ppo.calculate_policy_loss(adv, new_probs, old_probs)
        self.assertAlmostEqual(0.9, loss.item(), delta=1e-6)

    def test_two_losses5(self):
        ppo, adv, new_probs, old_probs = self.setup_test([-1., 0.2], [0.7, 1.], [1, 0.5])
        loss = ppo.calculate_policy_loss(adv, new_probs, old_probs)
        self.assertAlmostEqual(-0.28, loss.item(), delta=1e-6)

    def test_ten_losses0(self):
        ppo, adv, new_probs, old_probs = self.setup_test([0.] * 10, [1.] * 10, [1.] * 10)
        loss = ppo.calculate_policy_loss(adv, new_probs, old_probs)
        self.assertEqual(0., loss.item())

    def test_ten_losses1(self):
        ppo, adv, new_probs, old_probs = self.setup_test([1.] * 10, [1.] * 10, [1.] * 10)
        loss = ppo.calculate_policy_loss(adv, new_probs, old_probs)
        self.assertEqual(1., loss.item())

    def test_five_losses(self):
        ppo, adv, new_probs, old_probs = self.setup_test([0., 1, 0.5, -1., 0.2],
                                                         [1., 1., 1, 0.7, 1.],
                                                         [1., 0.5, 0.5, 1, 0.5])
        loss = ppo.calculate_policy_loss(adv, new_probs, old_probs)
        self.assertAlmostEqual(0.248, loss.item())

    def test_five_actors_single_loss(self):
        ppo, adv, new_probs, old_probs = self.setup_test([[0., 1., 0.5, -1., 0.2]],
                                                         [[1., 1., 1., 0.7, 1.]],
                                                         [[1., 0.5, 0.5, 1., 0.5]])
        loss = ppo.calculate_policy_loss(adv, new_probs, old_probs)
        self.assertAlmostEqual(0.248, loss.item(), delta=1e-6)


class ReturnTest(unittest.TestCase):
    @staticmethod
    def setup_test(rewards, masks, discount=0.99, gae_weight=1.0, num_workers=1):
        ppo = ProximalPolicyOptimization(MagicMock(), MagicMock(), MagicMock(),
                                         discount=discount, gae_weight=gae_weight, num_envs=num_workers)
        return ppo, torch.tensor(rewards), torch.tensor(masks)

    def test_kostrikov0(self):
        ppo, rewards, masks = self.setup_test(
            rewards=[[[0.], [0.]]] * 12,
            masks=[[[1.],  [1.]]] * 12,
            num_workers=2
        )

        result_kostrikov = torch.tensor(
            [[[0.0285], [0.0136]], [[0.0288], [0.0138]], [[0.0291], [0.0139]], [[0.0294], [0.0141]],
             [[0.0297], [0.0142]], [[0.0300], [0.0143]], [[0.0303], [0.0145]], [[0.0306], [0.0146]],
             [[0.0309], [0.0148]], [[0.0312], [0.0149]], [[0.0315], [0.0151]], [[0.0319], [0.0152]]]).squeeze()
             # [[0.0322], [0.0154]]])

        returns = ppo._calculate_returns(rewards.squeeze(),
                                         torch.tensor([[[0.0322], [0.0154]]]).squeeze(),
                                         masks.squeeze())
        self.assertTrue(torch.allclose(result_kostrikov, returns, atol=1e-4))

    def test_kostrikov1(self):
        ppo, rewards, masks = self.setup_test(
            rewards=[[[0.], [0.]], [[0.], [1.]], [[0.], [0.]], [[0.], [0.]], [[0.], [0.]], [[0.], [0.]], [[0.], [0.]],
                     [[0.], [0.]], [[0.], [0.]], [[0.], [0.]], [[0.], [0.]], [[0.], [0.]]],
            masks=[[[1.], [1.]], [[1.], [1.]], [[1.], [1.]], [[1.], [1.]], [[1.], [1.]], [[1.], [1.]], [[1.], [1.]],
                   [[1.], [1.]], [[1.], [1.]], [[0.], [1.]], [[1.], [1.]], [[1.], [1.]]],
            num_workers=2
        )

        result_kostrikov = torch.tensor(
            [[[0.0000], [0.9913]], [[0.0000], [1.0013]], [[0.0000], [0.0013]], [[0.0000], [0.0013]],
             [[0.0000], [0.0013]], [[0.0000], [0.0014]], [[0.0000], [0.0014]], [[0.0000], [0.0014]],
             [[0.0000], [0.0014]], [[0.0000], [0.0014]], [[0.0186], [0.0014]], [[0.0188], [0.0014]]]).squeeze()

        returns = ppo._calculate_returns(rewards.squeeze(),
                                         torch.tensor([[[0.0190], [0.0015]]]).squeeze(),
                                         masks.squeeze())
        self.assertTrue(torch.allclose(result_kostrikov, returns, atol=1e-4))

    def test_kostrikov2(self):
        ppo, rewards, masks = self.setup_test(
            rewards=[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1., 0.,
                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      0., 0.],
            masks=[1] * 128,
            num_workers=1)

        result = torch.tensor([[[-0.2020, -0.2060, -0.2102, -0.2121, -0.2142, -0.2162, -0.2185,
                                 -0.2209, -0.2235, -0.2262, -0.2291, -0.2321, -0.2354, -0.2395,
                                 -0.2430, -0.2472, -0.2507, -0.2538, -0.2573, -0.2630, -0.2685,
                                 -0.2740, -0.2805, -0.2868, -0.2949, -0.3026, -0.3105, -0.3181,
                                 -0.3270, -0.3361, -0.3453, -0.3554, -0.3671, -0.3785, -0.3916,
                                 -0.4065, -0.4213, -0.4374, -0.4526, -0.4704, -0.4894, -0.5073,
                                 -0.5283, -0.5496, -0.5719, -0.5970, -0.6231, -0.6508, -0.6809,
                                 -0.7137, -0.7472, -0.7829, -0.8206, -0.8608, -0.9043, -0.9504,
                                 -0.9997, -1.0529, -1.1087, -1.1680, -1.2296, -1.2957, -0.3025,
                                 -0.3108, -0.3196, -0.3286, -0.3387, -0.3490, -0.3599, -0.3713,
                                 -0.3830, -0.3955, -0.4096, -0.4242, -0.4397, -0.4565, -0.4744,
                                 -0.4934, -0.5129, -0.5348, -0.5573, -0.5792, -0.6037, -0.6305,
                                 -0.6593, -0.6907, -0.7247, -0.7599, -0.7974, -0.8367, -0.8784,
                                 -0.9226, -0.9703, -1.0200, -1.0735, -1.1300, -1.1905, -0.1912,
                                 -0.1920, -0.1929, -0.1933, -0.1936, -0.1939, -0.1947, -0.1956,
                                 -0.1968, -0.1980, -0.1993, -0.2003, -0.2013, -0.2020, -0.2026,
                                 -0.2032, -0.2042, -0.2059, -0.2071, -0.2063, -0.2067, -0.2080,
                                 -0.2095, -0.2118, -0.2150, -0.2173, -0.2199, -0.2221, -0.2244,
                                 -0.2267, -0.2298]]])

        returns = ppo._calculate_returns(rewards,
                                         torch.tensor([-0.2321]),
                                         masks)
        self.assertEqual(False, True)

    def test_two_actors_single_return(self):
        ppo, rewards, masks = self.setup_test([[1., 1.]], [[1., 1.]], num_workers=2)
        returns = ppo._calculate_returns(rewards, torch.tensor(0), masks)
        self.assertEqual(torch.Size([1, 2]), returns.shape)
        self.assertEqual(1, returns[0][0].item())
        self.assertEqual(1, returns[0][1].item())

    def test_two_actors_two_returns0(self):
        ppo, rewards, masks = self.setup_test([[1., 1.], [1, 0]], [[1., 1.], [1., 1.]], discount=1, num_workers=2)
        returns = ppo._calculate_returns(rewards, torch.tensor(0), masks)
        self.assertEqual(torch.Size([2, 2]), returns.shape)
        self.assertEqual(2, returns[0][0].item())
        self.assertEqual(1, returns[0][1].item())
        self.assertEqual(1, returns[1][0].item())
        self.assertEqual(0, returns[1][1].item())

    def test_two_actors_two_returns1(self):
        ppo, rewards, masks = self.setup_test([[1., 1.], [1, -1]], [[1., 1.], [1., 1.]], discount=0.5, num_workers=2)
        returns = ppo._calculate_returns(rewards, torch.tensor(0), masks)
        self.assertEqual(torch.Size([2, 2]), returns.shape)
        self.assertEqual(1.5, returns[0][0].item())
        self.assertEqual(0.5, returns[0][1].item())
        self.assertEqual(1, returns[1][0].item())
        self.assertEqual(-1, returns[1][1].item())

    def test_four_actors_four_returns(self):
        ppo, rewards, masks = self.setup_test([[1., 0., -1., 1.],
                                               [1., 0., -1., -1.],
                                               [1., 0., -1., 1.],
                                               [1., 0., -1., -1.]],
                                              [[1.] * 4] * 4,
                                              discount=0.5,
                                              num_workers=4)
        returns = ppo._calculate_returns(rewards, torch.tensor(0), masks)
        self.assertEqual(torch.Size([4, 4]), returns.shape)
        self.assertTrue(torch.allclose(torch.tensor([1.875, 0., -1.875, 0.625]), returns[0]))
        self.assertTrue(torch.allclose(torch.tensor([1.75, 0., -1.75, -0.75]), returns[1]))
        self.assertTrue(torch.allclose(torch.tensor([1.5, 0., -1.5, 0.5]), returns[2]))
        self.assertTrue(torch.allclose(torch.tensor([1., 0., -1., -1.]), returns[3]))

    def test_single_return0(self):
        ppo, rewards, masks = self.setup_test([1.], [1.])
        returns = ppo._calculate_returns(rewards, torch.tensor(0), masks)
        self.assertEqual(1, returns[0].item())

    def test_single_return1(self):
        ppo, rewards, masks = self.setup_test([-1.], [1.])
        returns = ppo._calculate_returns(rewards, torch.tensor(0), masks)
        self.assertEqual(-1, returns[0].item())

    def test_two_returns0(self):
        ppo, rewards, masks = self.setup_test([1., 1.], [1., 1.], 1)
        returns = ppo._calculate_returns(rewards, torch.tensor(0), masks)
        self.assertEqual(1, returns[1].item())
        self.assertEqual(2, returns[0].item())

    def test_two_returns_masked0(self):
        ppo, rewards, masks = self.setup_test([1., 1.], [0., 1.], 1)
        returns = ppo._calculate_returns(rewards, torch.tensor(0), masks)
        self.assertEqual(1, returns[1].item())
        self.assertEqual(1, returns[0].item())

    def test_two_returns_masked1(self):
        ppo, rewards, masks = self.setup_test([1., 1.], [0., 0.], 1)
        returns = ppo._calculate_returns(rewards, torch.tensor(0), masks)
        self.assertEqual(1, returns[1].item())
        self.assertEqual(1, returns[0].item())

    def test_two_returns_masked2(self):
        ppo, rewards, masks = self.setup_test([1., -1.], [0., 1.], 1)
        returns = ppo._calculate_returns(rewards, torch.tensor(0), masks)
        self.assertEqual(-1, returns[1].item())
        self.assertEqual(1, returns[0].item())

    def test_two_returns1(self):
        ppo, rewards, masks = self.setup_test([1., -1.], [1., 1.])
        returns = ppo._calculate_returns(rewards, torch.tensor(0), masks)
        self.assertEqual(-1, returns[1].item())
        self.assertAlmostEqual(0.01, returns[0].item(), delta=1e-8)

    def test_two_returns2(self):
        ppo, rewards, masks = self.setup_test([1., 1.], [1., 1.], 0.5)
        returns = ppo._calculate_returns(rewards, torch.tensor(0), masks)
        self.assertEqual(1, returns[1].item())
        self.assertEqual(1.5, returns[0].item())

    def test_five_returns_masked0(self):
        ppo, rewards, masks = self.setup_test([1., 1., 1., 1., 1.],
                                              [1., 1., 0., 1., 0.],
                                              discount=0.5, gae_weight=0.95)
        returns = ppo._calculate_returns(rewards, torch.tensor(0), masks)
        # self.assertTrue(torch.allclose(torch.tensor([[1.75], [1.5], [1.], [1.5], [1.]]).float(),
        #                                returns))  # normal return
        self.assertTrue(torch.allclose(torch.tensor([[1.7], [1.475], [1.], [1.475], [1.]]).float(),
                                       returns, atol=1e-3))  # GAE return

    def test_five_returns_masked1(self):
        ppo, rewards, masks = self.setup_test([1., 1., 1., 1., 1.],
                                              [0., 1., 1., 1., 0.],
                                              discount=0.5)
        returns = ppo._calculate_returns(rewards, torch.tensor(0), masks)
        self.assertTrue(torch.allclose(torch.tensor([[1.], [1.875], [1.75], [1.5], [1.]]).float(),
                                       returns))

    def test_five_returns_masked_two_actors(self):
        ppo, rewards, masks = self.setup_test([[1., 1.]] * 5,
                                              [[1., 0.], [1., 1.], [0., 1.], [1., 1.], [0., 0.]],
                                              discount=0.5,
                                              num_workers=2)
        returns = ppo._calculate_returns(rewards, torch.tensor(0), masks)
        self.assertTrue(torch.allclose(torch.tensor([[1.75, 1.], [1.5, 1.875], [1., 1.75], [1.5, 1.5], [1., 1.]]).
                                       float(),
                                       returns))

    def test_ten_returns0(self):
        ppo, rewards, masks = self.setup_test([1.] * 10, [1.] * 10, 1)
        returns = ppo._calculate_returns(rewards, torch.tensor(0), masks)
        self.assertEqual(10, returns[0].item())
        self.assertEqual(1, returns[-1].item())

    def test_ten_returns1(self):
        ppo, rewards, masks = self.setup_test([1.] * 10, [1.] * 10)
        returns = ppo._calculate_returns(rewards, torch.tensor(0), masks)
        self.assertAlmostEqual(9.5617924991, returns[0].item(), delta=1e-6)
        self.assertEqual(1, returns[-1].item())

    def test_bootstrapped_return0(self):
        ppo, rewards, masks = self.setup_test([0., 0.], [1., 1.], gae_weight=0.95)
        returns = ppo._calculate_returns(rewards, torch.tensor(1), masks)
        self.assertAlmostEqual(0.88454, returns[0].item(), delta=1e-6)  # GAE return
        self.assertAlmostEqual(0.9405, returns[1].item(), delta=1e-6)  # GAE return

    def test_bootstrapped_return1(self):
        ppo, rewards, masks = self.setup_test([0., 1.], [1., 0.], gae_weight=0.95)
        returns = ppo._calculate_returns(rewards, torch.tensor(1), masks)
        self.assertAlmostEqual(0.9405, returns[0].item(), delta=1e-6)  # GAE return
        # self.assertAlmostEqual(0.99, returns[0].item())  # normal return
        self.assertAlmostEqual(1, returns[1].item())


class SharedLossTest(unittest.TestCase):
    @staticmethod
    def setup_test(clip_loss, value_function_loss, entropy, c1=1., c2=0.01):
        rollout_mock = MagicMock()
        policy_mock = MagicMock()
        policy_mock.evaluate.return_value = ([0, 0], [0, 0], torch.tensor([entropy]))
        policy_mock.device = torch.device('cpu')
        ppo = ProximalPolicyOptimization(policy_mock, MagicMock(), MagicMock(), num_envs=1,
                                         value_function_coeff=c1, entropy_coeff=c2)
        ppo.calculate_policy_loss = lambda advantages, new_log_probabilities, old_log_probabilities: torch.tensor(
            [clip_loss])
        ppo.calculate_value_function_loss = lambda returns, new_values, old_values: torch.tensor([value_function_loss])
        ppo._calculate_advantages = lambda rewards, values: MagicMock()
        expected = clip_loss - c1 * value_function_loss + c2 * entropy
        return ppo, rollout_mock, expected

    def test_shared_loss_default_params0(self):
        ppo, r, expected = self.setup_test(1., 1., 1.)
        loss = ppo.calculate_shared_parameters_loss(r, MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock())
        self.assertAlmostEqual(0.01, expected, delta=1e-6)
        self.assertAlmostEqual(expected, loss.item(), delta=1e-6)

    def test_shared_loss_default_params1(self):
        ppo, r, expected = self.setup_test(0., 0., 0.)
        loss = ppo.calculate_shared_parameters_loss(r, MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock())
        self.assertAlmostEqual(0., expected, delta=1e-6)
        self.assertAlmostEqual(expected, loss.item(), delta=1e-6)

    def test_shared_loss_default_params2(self):
        ppo, r, expected = self.setup_test(-1., 1., -1.)
        loss = ppo.calculate_shared_parameters_loss(r, MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock())
        self.assertAlmostEqual(-2.01, expected, delta=1e-6)
        self.assertAlmostEqual(expected, loss.item(), delta=1e-6)

    def test_shared_loss_default_params3(self):
        ppo, r, expected = self.setup_test(1., -1., -1.)
        loss = ppo.calculate_shared_parameters_loss(r, MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock())
        self.assertAlmostEqual(1.99, expected, delta=1e-6)
        self.assertAlmostEqual(expected, loss.item(), delta=1e-6)

    def test_shared_loss0(self):
        ppo, r, expected = self.setup_test(0.5, -0.8, 0.2, 0.5, 0.1)
        loss = ppo.calculate_shared_parameters_loss(r, MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock())
        self.assertAlmostEqual(0.92, expected, delta=1e-6)
        self.assertAlmostEqual(expected, loss.item(), delta=1e-6)

    def test_shared_loss1(self):
        ppo, r, expected = self.setup_test(-0.3, 0.2, 0.4, 0.9, 0.1)
        loss = ppo.calculate_shared_parameters_loss(r, MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock())
        self.assertAlmostEqual(-0.44, expected, delta=1e-6)
        self.assertAlmostEqual(expected, loss.item(), delta=1e-6)

    def test_shared_loss2(self):
        ppo, r, expected = self.setup_test(0.3, -0.7, 0.1, 0.4, 0.9)
        loss = ppo.calculate_shared_parameters_loss(r, MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock())
        self.assertAlmostEqual(0.67, expected, delta=1e-6)
        self.assertAlmostEqual(expected, loss.item(), delta=1e-6)

    def test_shared_loss_randomized(self):
        for _ in range(100):
            ppo, r, expected = self.setup_test(*np.random.uniform(-1, 1, [5]))
            loss = ppo.calculate_shared_parameters_loss(r, MagicMock(), MagicMock(), MagicMock(), MagicMock(),
                                                        MagicMock())
            self.assertAlmostEqual(expected, loss.item(), delta=1e-6)


class ValueFunctionLossTest(unittest.TestCase):
    @staticmethod
    def setup_test(returns, values, old_values, discount=0.99):
        ppo = ProximalPolicyOptimization(MagicMock(), MagicMock(), MagicMock(), discount=discount, num_envs=1)
        return ppo, torch.tensor(returns), torch.tensor(values), torch.tensor(old_values)

    def test_shape_mismatch0(self):
        ppo, returns, values, old_values = self.setup_test([0], [0, 0], [0, 0])
        self.assertRaises(Exception, ppo.calculate_value_function_loss, returns, values, old_values)

    def test_shape_mismatch1(self):
        ppo, returns, values, old_values = self.setup_test([0, 0], [0], [0, 0])
        self.assertRaises(Exception, ppo.calculate_value_function_loss, returns, values, old_values)

    def test_shape_mismatch2(self):
        ppo, returns, values, old_values = self.setup_test([0, 0], [0, 0], [0])
        self.assertRaises(Exception, ppo.calculate_value_function_loss, returns, values, old_values)

    def test_single_value0(self):
        ppo, returns, values, old_values = self.setup_test([1.], [1.], [1.])
        loss = ppo.calculate_value_function_loss(returns, values, old_values)
        self.assertEqual(0, loss)

    def test_single_value1(self):
        ppo, returns, values, old_values = self.setup_test([1.], [0.], [0.])
        loss = ppo.calculate_value_function_loss(returns, values, old_values)
        self.assertEqual(0.5, loss)

    def test_single_value2(self):
        ppo, returns, values, old_values = self.setup_test([1.], [0.5], [0.25])
        loss = ppo.calculate_value_function_loss(returns, values, old_values)
        self.assertAlmostEqual(0.2112, loss.item(), delta=1e-4)  # maximum
        # self.assertEqual(0.125, loss)  # minimum

    def test_single_value3(self):
        ppo, returns, values, old_values = self.setup_test([1.], [0.25], [0.5])
        loss = ppo.calculate_value_function_loss(returns, values, old_values)
        self.assertAlmostEqual(0.2812, loss.item(), delta=1e-4)  # maximum
        # self.assertEqual(0.18, loss)  # minimum

    def test_two_actors_single_value0(self):
        ppo, returns, values, old_values = self.setup_test([[1., 1.]], [[1., 1.]], [[1., 1.]], 1)
        loss = ppo.calculate_value_function_loss(returns, values, old_values)
        self.assertEqual(0, loss)

    def test_two_actors_single_value1(self):
        ppo, returns, values, old_values = self.setup_test([[1.5, 1.]], [[1., 1.]], [[1., 1.]], 0.5)
        loss = ppo.calculate_value_function_loss(returns, values, old_values)
        self.assertEqual(0.0625, loss)

    def test_two_values0(self):
        ppo, returns, values, old_values = self.setup_test([1., 1.], [1., 1.], [1., 1.], 1)
        loss = ppo.calculate_value_function_loss(returns, values, old_values)
        self.assertEqual(0, loss)

    def test_two_values1(self):
        ppo, returns, values, old_values = self.setup_test([1.5, 1.], [1., 1.], [1., 1.], 0.5)
        loss = ppo.calculate_value_function_loss(returns, values, old_values)
        self.assertEqual(0.0625, loss)

    def test_two_values2(self):
        ppo, returns, values, old_values = self.setup_test([-1., 0.], [-0.6, 0.4], [-0.55, 0.5], 0.5)
        loss = ppo.calculate_value_function_loss(returns, values, old_values)
        self.assertEqual(0.08, loss)

    def test_two_values3(self):
        ppo, returns, values, old_values = self.setup_test([-0.5, 1.], [-0.6, 0.4], [-0.3, 0.7], 0.5)
        loss = ppo.calculate_value_function_loss(returns, values, old_values)
        self.assertAlmostEqual(0.0925, loss.item(), delta=1e-8)  # maximum
        # self.assertAlmostEqual(0.0425, loss.item(), delta=1e-8)  # minimum

    def test_kostrikov0(self):
        ppo, returns, values, old_values = self.setup_test(returns=[0.0000, 0.0000, 0.2050, 0.0000, 0.9135, 0.0000,
                                                                    0.0000, 0.2537, 0.0000, 0.9089, 0.0000, 0.7857,
                                                                    0.0000, 0.0000, 0.0000, 0.0000, 0.9180, 1.5308,
                                                                    0.0000, 0.0000, 1.5872, 0.0000, 0.0000, 0.8864,
                                                                    0.0000, 0.1735, 1.9403, 0.0000, 0.2563, 1.6424,
                                                                    0.0000, 0.8515],
                                                           old_values=[0.1050, 0.2280, 0.2278, 0.2020, 0.1165, 0.2387,
                                                                       0.2143, 0.2652, 0.1182, 0.2749, 0.0973, 0.2877,
                                                                       0.0882, 0.1786, 0.1512, 0.2034, 0.2666, 0.2587,
                                                                       0.1090, 0.2481, 0.2108, 0.0556, 0.2165, 0.2632,
                                                                       0.2512, 0.1290, 0.1841, 0.1563, 0.2538, 0.2220,
                                                                       0.2450, 0.2472],
                                                           values=[0.1050, 0.2280, 0.2278, 0.2020, 0.1165, 0.2387,
                                                                   0.2143, 0.2652, 0.1182, 0.2749, 0.0973, 0.2877,
                                                                   0.0882, 0.1786, 0.1512, 0.2034, 0.2666, 0.2587,
                                                                   0.1090, 0.2481, 0.2108, 0.0556, 0.2165, 0.2632,
                                                                   0.2512, 0.1290, 0.1841, 0.1563, 0.2538, 0.2220,
                                                                   0.2450, 0.2472])
        expected = 0.1826
        loss = ppo.calculate_value_function_loss(returns, values, old_values)
        self.assertAlmostEqual(expected, loss.item(), delta=1e-4)

    def test_kostrikov1(self):
        ppo, returns, values, old_values = self.setup_test(returns=[0.0000, 0.0000, 0.0000, 0.9273, 1.4125, 0.0000,
                                                                    0.0000, 1.1452, 0.0000, 0.0000, 0.0000, 0.0000,
                                                                    0.0000, 1.5400, 1.1513, 0.0000, 0.0000, 0.8262,
                                                                    0.0000, 0.0000, 0.0000, 0.8601, 0.1788, 0.0000,
                                                                    0.7106, 0.0000, 0.0000, 0.0000, 0.0000, 0.7106,
                                                                    0.1770, 0.0000],
                                                           values=[0.2806, 0.3907, 0.1304, 0.4263, 0.4235, 0.2170,
                                                                   0.1121, 0.4427, 0.3336, 0.3439, 0.3602, 0.3865,
                                                                   0.4297, 0.3572, 0.4994, 0.3989, 0.3984, 0.4337,
                                                                   0.3433, 0.2399, 0.3624, 0.4165, 0.4152, 0.3328,
                                                                   0.4734, 0.3667, 0.2749, 0.4001, 0.2416, 0.4364,
                                                                   0.3972, 0.2861],
                                                           old_values=[0.1153, 0.2425, -0.0272, 0.2323, 0.2487, 0.1180,
                                                                       -0.0342, 0.2444, 0.1803, 0.1928, 0.2017, 0.2234,
                                                                       0.2547, 0.1472, 0.2918, 0.2166, 0.2276, 0.2562,
                                                                       0.1786, 0.1280, 0.2021, 0.2457, 0.2303, 0.1780,
                                                                       0.2818, 0.2199, 0.1113, 0.2343, 0.1309, 0.2611,
                                                                       0.2111, 0.1265])
        ppo.clip_range = 0.2
        expected = 0.1002  # if using maximum
        # expected = 0.0997  # if using minimum
        loss = ppo.calculate_value_function_loss(returns, values, old_values)
        self.assertAlmostEqual(expected, loss.item(), delta=1e-4)


class PPOTest(unittest.TestCase):
    @staticmethod
    def setup_test(old_values, new_values, old_probs, new_probs, rewards, entropy,
                   num_workers=1, discount=1., gae_weight=1., clip_range=0.2):
        policy_mock = MagicMock()
        policy_mock.device = torch.device('cpu')
        rollout_mock = MagicMock()
        # rollout_mock.actions = torch.zeros(1)
        ppo = ProximalPolicyOptimization(policy_mock, MagicMock(), MagicMock(), num_envs=num_workers,
                                         discount=discount, gae_weight=gae_weight, clip_range=clip_range)
        policy_mock.evaluate.return_value = (torch.log(torch.tensor(new_probs)),
                                             torch.tensor(new_values[:-1]),
                                             torch.tensor(entropy))
        rollout_mock.log_action_probabilities = torch.log(torch.tensor(old_probs))
        advantages = ppo._calculate_advantages(torch.tensor(rewards),
                                               torch.tensor(old_values),
                                               torch.tensor([[1]] * len(rewards)))
        rollout_mock.returns = ppo._calculate_returns(advantages,
                                                      torch.tensor(old_values[:-1]))
        rollout_mock.values = torch.tensor(old_values[:-1])
        rollout_mock.advantages = normalize_tensor(advantages)
        return ppo, rollout_mock

    @staticmethod
    def setup_kostrikov(new_probs, new_values, entropy, value_func_coeff):
        policy_mock = MagicMock()
        policy_mock.evaluate.return_value = (new_probs, new_values, entropy)
        policy_mock.device = torch.device('cpu')
        ppo = ProximalPolicyOptimization(policy_mock, MagicMock(), MagicMock(), 1,
                                         value_function_coeff=value_func_coeff)
        return ppo

    def test_shared_loss0(self):
        ppo, r = self.setup_test(old_values=[1., 1.], new_values=[1., 1.], old_probs=[1.], new_probs=[1.], rewards=[1.],
                                 entropy=[1.])
        loss = ppo.calculate_shared_parameters_loss(MagicMock(), MagicMock(), r.log_action_probabilities, r.values,
                                                    r.advantages, r.returns)
        self.assertTrue(torch.isnan(loss))

    def test_shared_loss1(self):
        ppo, r = self.setup_test(old_values=[[-.2], [.4], [.8]], new_values=[[.1], [.5], [.8]], old_probs=[[.4], [.6]],
                                 new_probs=[[.5], [.9]], rewards=[[1.], [0]], entropy=[.2], discount=0.99,
                                 gae_weight=0.95)
        loss = ppo.calculate_shared_parameters_loss(MagicMock(), MagicMock(), r.log_action_probabilities, r.values,
                                                    r.advantages, r.returns)
        self.assertAlmostEqual(-0.9039, loss.item(), delta=1e-4)  # minimum in value function loss

    def test_shared_loss2(self):
        ppo, r = self.setup_test(old_values=[[-.1], [-.5], [.9]], new_values=[[-.2], [-.3], [.8]],
                                 old_probs=[[.4], [.5]], new_probs=[[.3], [.35]], rewards=[[-1.], [1.]], entropy=[.3],
                                 discount=0.99, gae_weight=0.95)
        loss = ppo.calculate_shared_parameters_loss(MagicMock(), MagicMock(), r.log_action_probabilities, r.values,
                                                    r.advantages, r.returns)
        self.assertAlmostEqual(-1.459878, loss.item(), delta=1e-4)

    def test_kostrikov0(self):
        old_values = torch.tensor([0.0173, 0.0246, 0.0317, 0.0241, 0.0292, 0.0443, 0.0248, 0.0341, 0.0446,
                                   0.0497, 0.0183, 0.0498, 0.0348, 0.0438, 0.0320, 0.0500, 0.0357, 0.0407,
                                   0.0402, 0.0451, 0.0240, 0.0368, 0.0118, 0.0254, 0.0220, 0.0236, 0.0258,
                                   0.0476, 0.0246, 0.0256, 0.0155, 0.1421])
        old_probs = torch.tensor([-1.3850, -1.3844, -1.3886, -1.3853, -1.3847, -1.3844, -1.3848, -1.3850,
                                  -1.3850, -1.3887, -1.3845, -1.3844, -1.3845, -1.3844, -1.3851, -1.3845,
                                  -1.3887, -1.3850, -1.3886, -1.3887, -1.3852, -1.3845, -1.3869, -1.3871,
                                  -1.3845, -1.3850, -1.3851, -1.3869, -1.3846, -1.3851, -1.3848, -1.3856])
        advantages = torch.tensor([-0.5719, -0.5087, -0.6056, -0.5879, -0.5560, -0.6353, -0.5894, 1.1848,
                                   1.1421, -0.6481, 1.5568, -0.6485, -0.6130, 1.6519, 1.7027, -0.5692,
                                   -0.6150, -0.5630, -0.6257, -0.6373, -0.5876, -0.6176, -0.4791, -0.5908,
                                   -0.5829, 1.6315, -0.5917, 1.3630, 1.5205, -0.5914, -0.5229, -0.8658])
        returns = torch.tensor([0.0000, 0.0341, 0.0000, 0.0000, 0.0186, 0.0000, 0.0000, 0.7623, 0.7547,
                                0.0000, 0.9044, 0.0000, 0.0000, 0.9703, 0.9801, 0.0337, 0.0000, 0.0271,
                                0.0000, 0.0000, 0.0000, 0.0000, 0.0338, 0.0000, 0.0000, 0.9415, 0.0000,
                                0.8515, 0.8953, 0.0000, 0.0190, 0.0000])
        ppo = self.setup_kostrikov(torch.tensor([-1.3850, -1.3844, -1.3886, -1.3853, -1.3847, -1.3844, -1.3848, -1.3850,
                                                 -1.3850, -1.3887, -1.3845, -1.3844, -1.3845, -1.3844, -1.3851, -1.3845,
                                                 -1.3887, -1.3850, -1.3886, -1.3887, -1.3852, -1.3845, -1.3869, -1.3871,
                                                 -1.3845, -1.3850, -1.3851, -1.3869, -1.3846, -1.3851, -1.3848,
                                                 -1.3856]),
                                   torch.tensor([0.0173, 0.0246, 0.0317, 0.0241, 0.0292, 0.0443, 0.0248, 0.0341, 0.0446,
                                                 0.0497, 0.0183, 0.0498, 0.0348, 0.0438, 0.0320, 0.0500, 0.0357, 0.0407,
                                                 0.0402, 0.0451, 0.0240, 0.0368, 0.0118, 0.0254, 0.0220, 0.0236, 0.0258,
                                                 0.0476, 0.0246, 0.0256, 0.0155, 0.1421]),
                                   torch.tensor(1.3863),
                                   0.5)
        expected = -0.1148
        loss = ppo.calculate_shared_parameters_loss(MagicMock(), MagicMock(), old_probs, old_values, advantages,
                                                    returns)
        self.assertAlmostEqual(expected, loss.item(), delta=1e-4)

    def test_kostrikov1(self):
        old_values = torch.tensor([0.0255, 0.0207, 0.0116, 0.0230, 0.0543, 0.0478, 0.0346, -0.0003,
                                   0.0319, 0.0360, 0.0184, 0.0434, 0.0535, 0.0340, -0.0136, 0.0473,
                                   0.0251, 0.0291, 0.0666, 0.0478, 0.0461, 0.0257, 0.0207, 0.0634,
                                   0.0358, 0.0288, 0.0077, 0.0236, 0.0289, 0.0538, 0.0326, 0.0200])
        old_probs = torch.tensor([-1.3845, -1.3887, -1.3847, -1.3870, -1.3849, -1.3887, -1.3887, -1.3884,
                                  -1.3850, -1.3886, -1.3870, -1.3870, -1.3852, -1.3864, -1.3865, -1.3870,
                                  -1.3870, -1.3886, -1.3870, -1.3844, -1.3870, -1.3869, -1.3844, -1.3888,
                                  -1.3845, -1.3851, -1.3850, -1.3849, -1.3852, -1.3845, -1.3844, -1.3850])
        advantages = torch.tensor([1.5865, -0.5799, -0.5584, 1.1752, -0.6589, 2.7520, -0.6126, -0.5303,
                                   -0.6062, -0.6158, -0.5744, 0.9912, -0.6570, -0.6112, 1.2792, 1.2654,
                                   -0.5902, -0.5995, 1.5527, -0.6437, 1.3868, -0.5915, -0.5797, -0.6805,
                                   3.0277, -0.5990, 1.5817, -0.5866, -0.5990, 2.0917, 2.4021, -0.5782])
        returns = torch.tensor([0.9243, 0.0000, 0.0000, 0.7472, 0.0000, 1.4412, 0.0000, 0.0000, 0.0000,
                                0.0000, 0.0000, 0.6894, 0.0000, 0.0000, 0.7547, 0.8097, 0.0000, 0.0000,
                                0.9510, 0.0000, 0.8601, 0.0000, 0.0000, 0.0000, 1.5463, 0.0000, 0.9044,
                                0.0000, 0.0000, 1.1670, 1.2775, 0.0000])
        ppo = self.setup_kostrikov(torch.tensor([-1.3949, -1.4653, -1.3952, -1.3497, -1.3401, -1.4647, -1.4652, -1.4600,
                                                 -1.3402, -1.4660, -1.3502, -1.3497, -1.3393, -1.3875, -1.3926, -1.3496,
                                                 -1.3496, -1.4661, -1.3492, -1.3950, -1.3497, -1.3501, -1.3952, -1.4641,
                                                 -1.3950, -1.3401, -1.3401, -1.3399, -1.3400, -1.3475, -1.3947,
                                                 -1.3397]),
                                   torch.tensor([0.1183, 0.1184, 0.1109, 0.1174, 0.1266, 0.1210, 0.1226, 0.1392, 0.1214,
                                                 0.1253, 0.1282, 0.1167, 0.1227, 0.0275, 0.0650, 0.1307, 0.1164, 0.1297,
                                                 0.1283, 0.1093, 0.1355, 0.1204, 0.1167, 0.1310, 0.1224, 0.1191, 0.1167,
                                                 0.1176, 0.1162, 0.1164, 0.1202, 0.1132]),
                                   torch.tensor(1.3852), 0.5)
        expected = 0.21733
        loss = ppo.calculate_shared_parameters_loss(MagicMock(), MagicMock(), old_probs, old_values, advantages,
                                                    returns)
        self.assertAlmostEqual(expected, loss.item(), delta=1e-4)


class LearnRateAndClipChangeTest(unittest.TestCase):
    @staticmethod
    def setup_test(max_time_step):
        params = MagicMock()
        ppo = ProximalPolicyOptimization(MagicMock(), params, MagicMock(),
                                         num_envs=1, max_updates=max_time_step, epochs=0)
        ppo._calculate_advantages = MagicMock()
        rollout = MagicMock()
        rollout.get_flattened_states_actions_probs.return_value = (MagicMock(), MagicMock(), MagicMock())
        return params, ppo, rollout

    @patch('torch.zeros')
    def test_no_lr_nor_cip_change(self, zeros_wd):
        params, ppo, rollout = self.setup_test(None)
        ppo.train(rollout)
        params.change_learn_rate_by.assert_not_called()

    @patch('torch.zeros')
    def test_lr_and_clip_change(self, zeros_wd):
        params, ppo, rollout = self.setup_test(100)
        ppo.train(rollout)
        self.assertAlmostEqual(0.99 * ppo.initial_clip_range, ppo.clip_range, delta=1e-8)
        self.assertAlmostEqual(0.99, params.change_learn_rate_by.call_args[0][0], delta=1e-8)

    @patch('torch.zeros')
    def test_lr_and_clip_annealing(self, zeros_wd):
        params, ppo, rollout = self.setup_test(10)
        for i in range(10):
            self.assertAlmostEqual((10 - i) / 10 * ppo.initial_clip_range, ppo.clip_range, delta=1e-8)
            ppo.train(rollout)
            self.assertAlmostEqual((9 - i) / 10, params.change_learn_rate_by.call_args[0][0], delta=1e-8)
        self.assertEqual(0, ppo.clip_range)

    @patch('torch.zeros')
    def test_lr_and_clip_changes(self, zeros_wd):
        params, ppo, rollout = self.setup_test(100)
        steps = 0
        for i in range(10):
            ppo.train(rollout)
            self.assertAlmostEqual((1 - (i + 1) / 100) * ppo.initial_clip_range, ppo.clip_range, delta=1e-8)
            self.assertAlmostEqual(1 - (i + 1) / 100, params.change_learn_rate_by.call_args[0][0], delta=1e-8)


class BatchTest(unittest.TestCase):
    @staticmethod
    def setup_test(steps, workers=1, epochs=3):
        ppo = ProximalPolicyOptimization(MagicMock(), MagicMock(), MagicMock(), workers, epochs=epochs)
        ppo.calculate_shared_parameters_loss = MagicMock()
        rollout = MagicMock()
        rollout.time_step = steps
        rollout.get_flattened_states_actions_probs.return_value = (MagicMock(), MagicMock(), MagicMock())
        return ppo, rollout

    @patch('torch.zeros')
    def test_full_sample_usage_one_worker(self, zeros_wd):
        ppo, rollout = self.setup_test(100, epochs=4)
        ppo.train(rollout)
        index_set = set()
        for call in rollout.__getitem__.call_args_list:
            args, kwargs = call
            index_set = index_set.union(set(args[0]))
        self.assertEqual(100, len(index_set))

    @patch('torch.zeros')
    def test_full_sample_usage_three_workers(self, zeros_wd):
        ppo, rollout = self.setup_test(100, workers=3, epochs=4)
        ppo.train(rollout)
        index_set = set()
        for call in rollout.__getitem__.call_args_list:
            args, kwargs = call
            index_set = index_set.union(set(args[0]))
        self.assertEqual(ppo.num_envs * 100, len(index_set))

    # TODO fix this test
    """
    @patch('torch.zeros')
    def test_early_stop(self, zeros_wd):
        ppo, rollout = self.setup_test(5, 1)
        ppo.train(rollout)
        self.assertEqual(1, rollout.__getitem__.call_count)"""

    @patch('torch.zeros')
    def test_no_enough_steps(self, zeros_wd):
        ppo, rollout = self.setup_test(1, 1)
        self.assertRaises(Exception, ppo.train, rollout)


if __name__ == '__main__':
    unittest.main()
