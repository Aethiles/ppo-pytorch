import numpy as np
import torch
import torch.nn as nn
import unittest

from source.utilities.neural_network import get_output_shape, orthogonal


class TestConvNet(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))
        return self.conv(x)


class GetOutputShapeTest(unittest.TestCase):
    k = np.random.randint(4, 16)
    s = np.random.randint(2, 8)
    i = np.random.randint(16, 32) * s + k
    kernel_size = (k, k)
    stride = (s, s)
    input_shape = (i, i)
    net = TestConvNet(kernel_size, stride)

    def test_output(self):
        input_ = np.ndarray(shape=self.input_shape, buffer=np.random.rand(self.input_shape[0]**2), dtype=np.float64)
        input_ = torch.from_numpy(input_).float().unsqueeze(0)
        output = self.net(input_)
        self.assertEqual(output.shape[-2:], get_output_shape(self.input_shape, self.kernel_size, self.stride))


class OrthogonalInitializationTest(unittest.TestCase):
    def test_init_relu(self):
        for _ in range(10):
            net: TestConvNet = TestConvNet(4, 2)
            old_weights = net.conv.weight.data.clone()
            orthogonal(net.conv, 'relu')
            new_weights = net.conv.weight.data.clone()
            self.assertFalse(torch.equal(old_weights, new_weights))

    def test_init_0_01(self):
        for _ in range(10):
            net: TestConvNet = TestConvNet(4, 2)
            old_weights = net.conv.weight.data.clone()
            orthogonal(net.conv, 0.01)
            new_weights = net.conv.weight.data.clone()
            self.assertFalse(torch.equal(old_weights, new_weights))

    def test_init_none(self):
        for _ in range(10):
            net: TestConvNet = TestConvNet(4, 2)
            old_weights = net.conv.weight.data.clone()
            orthogonal(net.conv)
            new_weights = net.conv.weight.data.clone()
            self.assertFalse(torch.equal(old_weights, new_weights))


if __name__ == '__main__':
    unittest.main()
