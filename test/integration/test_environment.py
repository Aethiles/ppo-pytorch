import numpy as np
import pickle
import unittest

from source.environment.atari.image_transformation import ImageTransformationEnv
from unittest.mock import MagicMock


class TransformationTest(unittest.TestCase):
    def test_transformation(self):
        env = ImageTransformationEnv(MagicMock())
        with open('integration/warps.p', 'rb') as f:
            data = pickle.load(f)
        for k in data:
            state = env.observation(data[k][0])
            self.assertTrue(np.allclose(state, data[k][1].squeeze() / 255.0))


if __name__ == '__main__':
    unittest.main()
