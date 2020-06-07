import numpy as np
import scipy.signal
import time
import torch
import unittest

from source.environment.atari.setup import setup_environment
from typing import Tuple


@unittest.skip
class AdvantagePerformanceTest(unittest.TestCase):
    @staticmethod
    def setup_test(elements,
                   device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'),
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
        rewards = torch.rand(elements).to(device)
        values = torch.rand(elements + 1).to(device)
        return rewards, values

    def test_single_loop(self):
        """
        Mean: 0.0349413
        Var:  8.7323241e-6
        """
        rewards, values = self.setup_test(1000)
        times = []
        for _ in range(100):
            start = time.time()
            horizon = rewards.shape[0]
            advantages = torch.zeros(horizon).to(rewards.device)
            old_advantage = 0
            deltas = rewards + 0.99 * values[1:] - values[:-1]
            for t in reversed(range(horizon)):
                # TODO use done states
                advantages[t] = deltas[t] + 0.99 * 0.95 * old_advantage
                old_advantage = advantages[t]
            times.append(time.time() - start)
        print('Mean:\t{}\nVar:\t{}\n'.format(np.mean(times), np.var(times)))

    def test_nested_loop(self):
        """
        Mean: 25.212536668777467
        Var:  0.004107787365401237
        """
        rewards, values = self.setup_test(1000)
        times = []
        for _ in range(10):
            start = time.time()
            adv = torch.zeros(rewards.shape[0]).to(rewards.device)
            for t in range(len(rewards)):
                sum_ = 0
                for k in range(len(rewards[t:])):
                    sum_ += (0.99 * 0.95) ** k * (rewards[t+k] + 0.99 * values[t+k+1] - values[t+k])
                adv[t] = sum_
            times.append(time.time() - start)
        print('Mean:\t{}\nVar:\t{}\n'.format(np.mean(times), np.var(times)))

    def test_scipy(self):
        """
        Just numpy and scipy:
        Mean: 0.000015611
        Var:  4.45646e-11
        Converting to tensor and putting on gpu, too:
        Mean: 0.0147864
        Var:  0.0214754
        :return:
        """
        rewards = np.random.rand(1000)
        values = np.random.rand(1001)
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        times = []
        for _ in range(100):
            start = time.time()
            deltas = rewards + 0.99 * values[1:] - values[:-1]
            adv = scipy.signal.lfilter([1], [1, float(-0.95 * 0.99)], deltas[::-1], axis=0)
            adv = torch.flip(torch.from_numpy(adv).to(device), dims=(0, ))
            times.append(time.time() - start)
        print('Mean:\t{}\nVar:\t{}\n'.format(np.mean(times), np.var(times)))


@unittest.skip
class RolloutToTensorTest(unittest.TestCase):
    def test_ndarray_one_conversion_at_end(self):
        """
        1000 elements:
        Mean: 0.28480599
        Var:  0.02050979
        100 elements:
        Mean: 0.0856
        Var:  0.0207
        """
        times = []
        for _ in range(100):
            start = time.time()
            device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
            states = np.zeros((1000, 4, 84, 84))
            for i in range(1000):
                state = np.ndarray((4, 84, 84), buffer=np.random.rand(4 * 84 * 84), dtype=np.float)
                states[i] = state
            torch.from_numpy(states).to(device)
            times.append(time.time() - start)
        print('Mean:\t{}\nVar:\t{}\n'.format(np.mean(times), np.var(times)))

    def test_ndarray_immediate_conversion(self):
        """
        1000 elements:
        Mean: 0.27870254
        Var:  0.02631753
        100 elements:
        Mean: 0.0622
        Var:  0.0239
        """
        times = []
        for _ in range(100):
            start = time.time()
            device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
            states = torch.zeros((1000, 4, 84, 84)).to(torch.device(device))
            for i in range(1000):
                state = np.ndarray((4, 84, 84), buffer=np.random.rand(4 * 84 * 84), dtype=np.float)
                states[i] = torch.from_numpy(state)
            times.append(time.time() - start)
        print('Mean:\t{}\nVar:\t{}\n'.format(np.mean(times), np.var(times)))

    def test_array_one_conversion_at_end(self):
        """
        1000 elements:
        Mean: 0.0166
        Var:  0.0248
        100 elements:
        Mean: 0.0149
        Var:  0.0216
        """
        times = []
        for _ in range(100):
            start = time.time()
            device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
            rewards = np.zeros(1000)
            for i in range(1000):
                reward = np.float(np.random.rand(1))
                rewards[i] = reward
            torch.from_numpy(rewards).to(device)
            times.append(time.time() - start)
        print('Mean:\t{}\nVar:\t{}\n'.format(np.mean(times), np.var(times)))

    def test_array_immediate_conversion(self):
        """
        1000 elements:
        Mean: 0.0410
        Var:  0.0237
        100 elements:
        Mean: 0.0187
        Var:  0.0249
        """
        times = []
        for _ in range(100):
            start = time.time()
            device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
            rewards = torch.zeros(1000).to(torch.device(device))
            for i in range(1000):
                reward = np.float(np.random.rand(1))
                rewards[i] = torch.tensor(reward)
            times.append(time.time() - start)
        print('Mean:\t{}\nVar:\t{}\n'.format(np.mean(times), np.var(times)))


@unittest.skip
class TensorToNumpyTest(unittest.TestCase):
    def test_array_one_conversion_at_end(self):
        """
        1000 elements:
        Mean: 0.0336
        Var:  0.0215
        100 elements:
        Mean: 0.0179
        Var:  0.0255
        """
        times = []
        for _ in range(100):
            device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
            start = time.time()
            values = torch.zeros(1000).to(device)
            for i in range(100):
                value = torch.rand(1)
                values[i] = value
            values = values.cpu().numpy()
            times.append(time.time() - start)
        print('Mean:\t{}\nVar:\t{}\n'.format(np.mean(times), np.var(times)))

    def test_array_immediate_conversion(self):
        """
        1000 elements:
        Mean: 0.0505
        Var:  0.0254
        100 elements:
        Mean: 0.0198
        Var:  0.0274
        """
        times = []
        for _ in range(100):
            device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
            start = time.time()
            values = np.zeros(1000)
            for i in range(100):
                value = torch.rand(1).to(device)
                values[i] = value.item()
            times.append(time.time() - start)
        print('Mean:\t{}\nVar:\t{}\n'.format(np.mean(times), np.var(times)))


@unittest.skip
class ReturnTest(unittest.TestCase):
    def test_loop(self):
        """
        CPU: 0.0129
        GPU: 0.0355
        """
        device = torch.device('cpu')
        rewards = torch.rand(1000).to(device)
        times = []
        for _ in range(100):
            start = time.time()
            horizon = rewards.shape[0]
            returns = torch.zeros(horizon).to(rewards.device)
            old_return = 0
            for t in reversed(range(horizon)):
                returns[t] = rewards[t] + 0.99 * old_return
                old_return = returns[t]
            times.append(time.time() - start)
        print('Mean:\t{}\nVar:\t{}\n'.format(np.mean(times), np.var(times)))

    def test_cumsum(self):
        """
        CPU: 0.000075
        GPU: 0.000094
        """
        device = torch.device('cuda:0')
        rewards = torch.rand(1000).to(device)
        times = []
        for _ in range(100):
            start = time.time()
            factors = torch.tensor([0.99] * 1000).to(device)
            factors = torch.cumprod(factors, 0)
            # factors = factors.to(device)
            returns = torch.cumsum(rewards, dim=0) * factors
            times.append(time.time() - start)
        print('Mean:\t{}\nVar:\t{}\n'.format(np.mean(times), np.var(times)))


@unittest.skip
class TensorTransposing(unittest.TestCase):
    @staticmethod
    def setup_test(workers, horizon):
        return torch.rand((horizon, workers))

    def test_transposing_in_place_then_sum(self):
        times = []
        for _ in range(100):
            data = self.setup_test(8, 128)
            start = time.time()
            data.transpose_(0, 1)
            res = torch.sum(data, dim=1)
            times.append(time.time() - start)
        print('Mean:\t{}\nVar:\t{}\n'.format(np.mean(times), np.var(times)))

    def test_transposing_then_sum(self):
        times = []
        for _ in range(100):
            data = self.setup_test(8, 128)
            start = time.time()
            data = data.transpose(0, 1)
            res = torch.sum(data, dim=1)
            times.append(time.time() - start)
        print('Mean:\t{}\nVar:\t{}\n'.format(np.mean(times), np.var(times)))

    def test_summing(self):
        times = []
        for _ in range(100):
            data = self.setup_test(8, 128)
            start = time.time()
            res = torch.sum(data, dim=0)
            times.append(time.time() - start)
        print('Mean:\t{}\nVar:\t{}\n'.format(np.mean(times), np.var(times)))


@unittest.skip
class EnvironmentSteppingTest(unittest.TestCase):
    def test_1000_steps(self):
        env = setup_environment('BreakoutNoFrameskip-v4')
        env.reset()
        for _ in range(100000):
            state, reward, done, info = env.step(env.action_space.sample())
            if done:
                env.reset()


if __name__ == '__main__':
    unittest.main()
