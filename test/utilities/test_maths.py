import torch
import unittest

from source.utilities.maths import normalize_tensor


class NormalizationTest(unittest.TestCase):
    def test_nan(self):
        t = torch.tensor([1.])
        r = normalize_tensor(t)
        self.assertTrue(torch.isnan(r))

    def test_two_values(self):
        t = torch.tensor([0., 1.])
        mean = 0.5
        std = 0.7071
        n = (t - mean) / (std + 1e-5)
        r = normalize_tensor(t)
        for i in range(2):
            self.assertAlmostEqual(n[i].item(), r[i].item(), delta=1e-5)


if __name__ == '__main__':
    unittest.main()
