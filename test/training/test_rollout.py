import numpy as np
import torch
import unittest

from source.training.rollout import Rollout


class GetFlattenedTest(unittest.TestCase):
    @staticmethod
    def setup_test(horizon=np.random.randint(5, 25), num_envs=np.random.randint(5, 25),
                   shape=np.random.randint(2, 10, 2)):
        r = Rollout(horizon=horizon, initial_state=np.random.rand(num_envs, *shape), num_environments=num_envs,
                    device=torch.device('cpu'))
        for _ in range(horizon):
            r.save_time_step(np.random.rand(num_envs, *shape),
                             list(np.random.rand(num_envs)),
                             torch.from_numpy(np.random.randint(0, 4, num_envs)),
                             torch.randn(num_envs),
                             torch.randn(num_envs),
                             [False] * num_envs)
        r.advantages = torch.rand(horizon, num_envs)
        r.returns = torch.rand(horizon, num_envs)
        return r, num_envs

    def test_get_flattened_states_actions_probs(self):
        for _ in range(10):
            r, num_envs = self.setup_test()
            states, actions, log_action_probs = r.get_flattened_states_actions_probs()
            self.assertEqual((r.horizon + 1) * num_envs, states.shape[0])
            self.assertEqual(r.horizon * num_envs, actions.shape[0])
            self.assertEqual(r.horizon * num_envs, log_action_probs.shape[0])


class SaveTest(unittest.TestCase):
    def test_init(self):
        r = Rollout(horizon=1, initial_state=np.arange(1), num_environments=1, device=torch.device('cpu'))
        self.assertTrue(torch.eq(torch.from_numpy(np.arange(1)), r.states[0]))

    def test_init_two_workers(self):
        state1 = np.random.rand(4, 8, 8)
        state2 = np.random.rand(4, 8, 8)
        r = Rollout(horizon=1, initial_state=np.stack((state1, state2)), num_environments=2,
                    device=torch.device('cpu'))
        self.assertTrue(torch.all(torch.eq(torch.as_tensor(state1).float(), r.states[0][0])))
        self.assertTrue(torch.all(torch.eq(torch.as_tensor(state2).float(), r.states[0][1])))

    def test_random_init(self):
        size = np.random.randint(4, 16)
        val = np.random.randint(-25, 25)
        r = Rollout(horizon=1, initial_state=np.expand_dims(np.arange(size) - val, 0), num_environments=1,
                    device=torch.device('cpu'))
        self.assertTrue(torch.all(torch.eq(torch.from_numpy(np.arange(size) - val), r.states[0])))

    def test_single_step(self):
        size = np.random.randint(4, 16)
        r = Rollout(horizon=1, initial_state=np.expand_dims(np.arange(size), 0), num_environments=1,
                    device=torch.device('cpu'))
        r.save_time_step(np.arange(size) - size, [0.2], torch.tensor([2]), torch.tensor([0.5]), torch.tensor([1.2]),
                         [False])
        self.assertTrue(torch.all(torch.eq(torch.stack((torch.from_numpy(np.arange(size)),
                                                        torch.from_numpy(np.arange(size) - size))).unsqueeze(1),
                                           r.states)))
        self.assertTrue(torch.eq(torch.tensor(0.2), r.rewards))

    def test_single_step_two_workers(self):
        state1 = np.random.rand(4, 8, 8)
        state2 = np.random.rand(4, 8, 8)
        r = Rollout(horizon=1, initial_state=np.random.rand(2, 4, 8, 8), num_environments=2,
                    device=torch.device('cpu'))
        r.save_time_step(np.stack((state1, state2)), [0.2, -0.2], torch.tensor([2, 0]), torch.tensor([0.5, 1.]),
                         torch.tensor([1.2, 0.33]), [False, True])
        self.assertTrue(torch.all(torch.eq(torch.as_tensor(state1).float(), r.states[1][0])))
        self.assertTrue(torch.all(torch.eq(torch.as_tensor(state2).float(), r.states[1][1])))

    def test_mask_generation(self):
        size = np.random.randint(4, 16)
        r = Rollout(horizon=10, initial_state=np.random.rand(2, size), num_environments=2,
                    device=torch.device('cpu'))
        for i in range(10):
            r.save_time_step(np.random.rand(2, size), [0., 0.], torch.tensor([0, 0]), torch.rand(2), torch.rand(2),
                             [False, True])
        self.assertFalse(torch.all(r.states[10] == 0))
        self.assertTrue(torch.all(r.masks[1:, 1] == 0))
        self.assertTrue(torch.all(r.masks[1:, 0] == 1))

    def test_exception_on_step(self):
        r = Rollout(horizon=1, initial_state=np.expand_dims(np.arange(5), 0), num_environments=1,
                    device=torch.device('cpu'))
        r.save_time_step(np.expand_dims(np.arange(5), 0), np.random.rand(1)[0], torch.rand(1),
                         torch.rand(1), torch.rand(1), [False])
        self.assertRaises(IndexError, r.save_time_step, np.arange(5), np.random.rand(1)[0], np.random.rand(1)[0],
                          torch.from_numpy(np.random.rand(1)), np.random.rand(1)[0], [False])


class FinalizeTest(unittest.TestCase):
    @staticmethod
    def setup_test(horizon, steps):
        size = np.random.randint(4, 16)
        r = Rollout(horizon=horizon, initial_state=np.expand_dims(np.arange(size), 0), num_environments=1,
                    device=torch.device('cpu'))
        for i in range(steps):
            r.save_time_step(np.expand_dims(np.arange(size) + i, 0), [i/10], torch.tensor([i]), torch.tensor([i]),
                             torch.tensor([i/10]), [False])
        return r

    def verify(self, steps, r):
        self.assertEqual(steps + 1, r.states.shape[0])
        self.assertEqual(steps, r.actions.shape[0])
        self.assertEqual(steps, r.log_action_probabilities.shape[0])
        self.assertEqual(steps, r.rewards.shape[0])
        self.assertEqual(steps + 1, r.values.shape[0])

    def test_finalization(self):
        horizon = np.random.randint(4, 8)
        r = self.setup_test(horizon, horizon)
        r.finalize(torch.tensor([-1.]))
        self.verify(horizon, r)


if __name__ == '__main__':
    unittest.main()
