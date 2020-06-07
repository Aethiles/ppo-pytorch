import copy
import gym
import numpy as np
import unittest
import test.helpers.utils as utils
import torch

from source.environment.multiprocessing_env import MultiprocessingEnv
from source.training.rollout import Rollout


class SerialRolloutIntegrationTest(unittest.TestCase):
    def test_rollout_without_done(self):
        reward_gen = utils.RewardGenerator(0)
        r, env, _ = utils.setup_test(utils.stable_state_gen(state=np.ndarray((84, 84, 3),
                                                                             buffer=np.random.rand(84 * 84 * 3) * 255,
                                                                             dtype=float)),
                                     reward_gen.reward_func,
                                     utils.stable_done_gen(False),
                                     horizon=1000)
        rollout = r.run()
        self.assertEqual(1, env.reset_ctr)
        self.assertLessEqual(1000 * 4, env.step_ctr)
        self.assertTrue(torch.all(rollout.rewards[rollout.actions == 0] >= 1))
        self.assertTrue(torch.all(torch.eq(rollout.states[3], rollout.states[-1])))
        self.assertTrue(torch.all(rollout.masks == True))

    def test_rollout_with_done(self):
        reward_gen = utils.RewardGenerator(0)
        r, env, _ = utils.setup_test(utils.stable_state_gen(state=np.ndarray((84, 84, 3),
                                                                             buffer=np.random.rand(84 * 84 * 3) * 255,
                                                                             dtype=float)),
                                     reward_gen.reward_func,
                                     utils.done_at_n_gen(209),
                                     horizon=100,
                                     noop=1)
        rollout = r.run()
        self.assertEqual(100, rollout.actions.shape[0])
        self.assertEqual(101, rollout.values.shape[0])
        self.assertEqual(rollout.values[-2].item(), rollout.values[-1].item())
        self.assertTrue(torch.all(rollout.states[51][0][2] == 0))
        self.assertTrue(torch.all(rollout.states[51][0][1] == 0))
        self.assertTrue(torch.all(rollout.states[51][0][0] == 0))
        self.assertFalse(rollout.masks[50].item())


class ParallelRolloutIntegrationTest(unittest.TestCase):
    def test_rollout(self):
        reward_gen = utils.RewardGenerator(0)
        r, _, _ = utils.setup_test(utils.stable_state_gen(state=np.ndarray((84, 84, 3),
                                                                           buffer=np.random.rand(84 * 84 * 3) * 255,
                                                                           dtype=float)),
                                   reward_gen.reward_func,
                                   utils.stable_done_gen(False),
                                   horizon=128,
                                   num_envs=8,
                                   env_class=MultiprocessingEnv)
        rollout = r.run()
        r.environment.close()
        self.assertTrue(torch.all(rollout.rewards[rollout.actions == 0] >= 1))
        self.assertEqual(torch.Size([129, 8, 4, 84, 84]), rollout.states.shape)
        self.assertTrue(torch.all(rollout.states[0, :, :3] == 0))
        for pipe, process in zip(r.environment.pipes, r.environment.processes):
            self.assertTrue(pipe.closed)
            self.assertFalse(process.is_alive())


class RolloutRewardTest(unittest.TestCase):
    def test_rollout_no_postprocessing(self):
        rollout = Rollout(horizon=200, num_environments=1, device=torch.device('cpu'), initial_state=np.array(-1))
        env = gym.make('SpaceInvaders-v4')
        env.reset()
        rewards = []
        for i in range(200):
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            rewards.append(copy.copy(reward))
            rollout.save_time_step(np.array(i), [reward], torch.tensor([action]), torch.tensor(i), torch.tensor(i),
                                   [done])
        for i, reward in enumerate(rewards):
            self.assertEqual(reward, rollout.rewards[i])
        self.assertTrue(torch.any(rollout.rewards != 0))

    def test_rollout(self):
        env = MultiprocessingEnv('SpaceInvadersNoFrameskip-v4', num_envs=8)
        state = env.reset()
        rollout = Rollout(horizon=1000, num_environments=8, device=torch.device('cpu'), initial_state=state)
        obs = {'r': [], 's': [], 'a': []}
        for i in range(1000):
            action = np.random.randint(0, 6)
            state, reward, done, info = env.step([action] * 8)
            obs['r'].append(copy.copy(reward))
            obs['s'].append(copy.copy(state))
            obs['a'].append(copy.copy([action] * 8))
            rollout.save_time_step(state, reward, torch.tensor([action] * 8), torch.tensor([i] * 8),
                                   torch.tensor([i] * 8), done)
        for i in range(1000):
            self.assertTrue(torch.all(torch.eq(torch.tensor(obs['r'][i]), rollout.rewards[i])))
            self.assertTrue(torch.all(torch.eq(torch.tensor(obs['s'][i]).float(), rollout.states[i+1])))
            self.assertTrue(torch.all(torch.eq(torch.tensor(obs['a'][i]), rollout.actions[i])))

        self.assertTrue(torch.any(rollout.rewards != 0))


if __name__ == '__main__':
    unittest.main()
