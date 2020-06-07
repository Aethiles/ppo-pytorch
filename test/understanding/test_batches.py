import numpy as np
import torch.utils.data.sampler as samplers
import unittest


class BatchTest(unittest.TestCase):
    def test_different_batches(self):
        mini_batch_size = 3
        sampler = samplers.BatchSampler(sampler=samplers.SubsetRandomSampler(range(12)),
                                        batch_size=mini_batch_size,
                                        drop_last=True)
        runs = []
        for i in range(100):
            indices = []
            for batch in sampler:
                indices.append(batch)
            runs.append(indices)
        for i, j in zip(range(100), range(100)):
            if i != j:
                self.assertFalse(np.array_equal(runs[i], runs[j]))


if __name__ == '__main__':
    unittest.main()
