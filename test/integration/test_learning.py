import numpy as np
import test.helpers.utils as utils
import time
import unittest

from source.training.ppo import ProximalPolicyOptimization
from unittest.mock import MagicMock


class SimplePPOTest(unittest.TestCase):
    def run_and_verify(self, r, pol, epochs, num_envs=1):
        ro1 = r.run()
        old_probs, old_val = pol.parameters(ro1.states[-2])
        ppo = ProximalPolicyOptimization(pol, pol.parameters, MagicMock(), num_envs=num_envs)
        for _ in range(epochs):
            rollout = r.run()
            ppo.train(rollout)
        new_probs, new_val = pol.parameters(ro1.states[-2])
        self.assertLess(old_probs[0][0].item(), new_probs[0][0].item())

    def test_without_gradient_clipping(self):
        """
        Still fails occasionally
        :return:
        """
        state = np.ndarray((84, 84, 3), buffer=np.random.rand(84 * 84 * 3) * 255, dtype=np.float)
        reward = utils.RewardGenerator(0)
        r, env, pol = utils.setup_test(utils.stable_state_gen(state), reward.reward_func, utils.stable_done_gen(False),
                                       horizon=10)
        self.run_and_verify(r, pol, 25)

    def test_with_gradient_clipping(self):
        state = np.ndarray((84, 84, 3), buffer=np.random.rand(84 * 84 * 3) * 255, dtype=np.float)
        reward = utils.RewardGenerator(0)
        r, env, pol = utils.setup_test(utils.stable_state_gen(state), reward.reward_func, utils.stable_done_gen(False),
                                       horizon=10, max_grad_norm=0.5)
        self.run_and_verify(r, pol, 25)

    def test_four_environments(self):
        state = np.ndarray((84, 84, 3), buffer=np.random.rand(84 * 84 * 3) * 255, dtype=np.float)
        reward = utils.RewardGenerator(0)
        r, env, pol = utils.setup_test(utils.stable_state_gen(state), reward.reward_func, utils.stable_done_gen(False),
                                       horizon=10, max_grad_norm=0.5, num_envs=4)
        self.run_and_verify(r, pol, 25, num_envs=4)

    @unittest.skip
    def test_performance(self):
        reward = utils.RewardGenerator(0)
        r, env, pol = utils.setup_test(utils.random_state_gen((84, 84, 3)),
                                       reward.reward_func,
                                       utils.stable_done_gen(False),
                                       horizon=128,
                                       max_grad_norm=0.5)
        times = []
        ppo = ProximalPolicyOptimization(pol, pol.parameters, MagicMock(), 1)
        for _ in range(20):
            start = time.time()
            for i in range(10):
                rollout = r.run()
                ppo.train(rollout)
            times.append(time.time() - start)
            print(_)
        print('Mean:\t{}\nVar:\t{}\n'.format(np.mean(times), np.var(times)))


if __name__ == '__main__':
    unittest.main()
