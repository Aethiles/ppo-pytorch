import numpy as np
import torch
import unittest

from source.utilities.logging.list_logger import ListLogger
from source.policy.parameters import AtariSize, AtariParameters
from unittest.mock import MagicMock


class AtariTestCase(unittest.TestCase):
    state = np.ndarray((4, 84, 84), buffer=np.random.rand(4 * 84 * 84), dtype=np.float64)
    output_size = np.random.randint(2, 7)

    def get_estimations(self, net_size, device=None):
        if device is None:
            params = AtariParameters(net_size, self.state.shape, self.output_size, 2.5*1e-4, logger_=None)
        else:
            params = AtariParameters(net_size, self.state.shape, self.output_size, 2.5*1e-4, device=device,
                                     logger_=None)
        state = torch.from_numpy(self.state).float().unsqueeze(0).to(params.device)
        return params(state)

    def verify_estimation_shape(self, estimations):
        # action probabilities and action values
        self.assertEqual(len(estimations), 2)
        # batch size is 0, hence get the first element
        self.assertEqual(len(estimations[0][0]), self.output_size)
        self.assertEqual(len(estimations[1][0]), 1)

    def test_small_forward_cpu(self):
        estimations = self.get_estimations(AtariSize.SMALL, torch.device('cpu'))
        self.verify_estimation_shape(estimations)

    def test_large_forward_cpu(self):
        estimations = self.get_estimations(AtariSize.LARGE, torch.device('cpu'))
        self.verify_estimation_shape(estimations)

    @unittest.skipIf(not torch.cuda.is_available(), 'Cuda is not available so GPU cannot be tested')
    def test_small_forward_gpu(self):
        estimations = self.get_estimations(AtariSize.SMALL, torch.device('cuda'))
        self.verify_estimation_shape(estimations)

    @unittest.skipIf(not torch.cuda.is_available(), 'Cuda is not available so GPU cannot be tested')
    def test_large_forward_gpu(self):
        estimations = self.get_estimations(AtariSize.LARGE, torch.device('cuda'))
        self.verify_estimation_shape(estimations)

    def test_device_default(self):
        params = AtariParameters(AtariSize.SMALL, self.state.shape, 1, 2.5*1e-4, logger_=None)
        if torch.cuda.is_available():
            self.assertEqual('cuda', params.device.type)
        else:
            self.assertEqual('cpu', params.device.type)

    def test_device_cpu(self):
        params = AtariParameters(AtariSize.SMALL, self.state.shape, 1, 2.5*1e-4, device=torch.device('cpu'),
                                 logger_=None)
        self.assertEqual('cpu', params.device.type)

    @unittest.skipIf(not torch.cuda.is_available(), 'Cuda is not available so GPU cannot be tested')
    def test_device_gpu(self):
        params = AtariParameters(AtariSize.SMALL, self.state.shape, 1, 2.5*1e-4, device=torch.device('cuda'),
                                 logger_=None)
        self.assertEqual('cuda', params.device.type)

    def test_batch_forward(self):
        params = AtariParameters(AtariSize.SMALL, self.state.shape, 4, 2.5*1e-4, logger_=None)
        states = torch.cat([torch.from_numpy(self.state).float().unsqueeze(0).to(params.device) for _ in range(10)])
        estimations = params(states)
        self.assertEqual(len(estimations[0]), 10)
        self.assertEqual(len(estimations[1]), 10)

    def test_grad_norm_logging(self):
        logger = ListLogger('', 1, None)
        params = AtariParameters(AtariSize.SMALL, self.state.shape, 1, 2.5*1e-4, logger_=logger)
        estimation = params(torch.as_tensor(self.state).float().unsqueeze(0).to(params.device))[0]
        params.gradient_update(estimation ** 2)
        self.assertNotEqual(0, len(logger.gradient_norm_buffer))

    def test_change_learn_rate_by(self):
        params = AtariParameters(AtariSize.SMALL, self.state.shape, 1, 0.1, logger_=MagicMock())
        params.change_learn_rate_by(0.5)
        self.assertEqual(0.1 * 0.5, params.optimizer.param_groups[0]['lr'])


if __name__ == '__main__':
    unittest.main()
