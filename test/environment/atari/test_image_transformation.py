import numpy as np
import unittest

from source.environment.atari.image_transformation import ImageTransformationEnv, ShapeError
from test.helpers.environment import MockEnv, MockWrapper


def setup_env(shape, state=None):
    mock = MockEnv(lambda: state,
                   lambda _: (state, None, None, None),
                   [''])
    env = ImageTransformationEnv(MockWrapper(mock), shape)
    return env


class GrayscaleTest(unittest.TestCase):
    def test_wrong_datatype(self):
        image = np.ndarray((3, 16, 9), buffer=np.zeros(3 * 16 * 9), dtype=np.float64)
        env = setup_env(image.shape)
        self.assertRaises(TypeError, env.convert_to_grayscale, image)

    def test_wrong_order(self):
        image = np.ndarray((3, 16, 9), buffer=np.zeros(3 * 16 * 9), dtype=np.uint8)
        env = setup_env(image.shape)
        self.assertRaises(ShapeError, env.convert_to_grayscale, image)

    def test_two_channels(self):
        image = np.ndarray((16, 9, 2), buffer=np.zeros(2 * 16 * 9), dtype=np.uint8)
        env = setup_env(image.shape)
        self.assertRaises(ShapeError, env.convert_to_grayscale, image)

    def test_no_channels(self):
        image = np.ndarray((16, 9), buffer=np.zeros(16 * 9), dtype=np.uint8)
        env = setup_env(image.shape)
        self.assertRaises(ShapeError, env.convert_to_grayscale, image)

    def test_conversion(self):
        image = np.ndarray((16, 9, 3), buffer=np.zeros(16 * 9 * 3), dtype=np.uint8)
        env = setup_env(image.shape)
        result = env.convert_to_grayscale(image)
        self.assertEqual((16, 9), result.shape)


class ResizeTest(unittest.TestCase):
    def test_resizing(self):
        image = np.ndarray((210, 160), buffer=np.random.rand(210 * 160), dtype=float)
        env = setup_env((84, 84))
        image = env.resize(image)
        self.assertEqual(image.shape, (84, 84))


class ObservationTest(unittest.TestCase):
    def test_grayscale_and_resize(self):
        image = np.ndarray((210, 160, 3), buffer=np.random.randint(0, 256, size=210 * 160 * 3), dtype=np.uint8)
        env = setup_env((84, 84), state=image)
        ret = env.reset()
        self.assertEqual((84, 84), ret.shape)

    def test_resize_only(self):
        image = np.ndarray((210, 160, 3), buffer=np.random.randint(0, 256, size=210 * 160 * 3), dtype=np.uint8)
        env = setup_env((84, 84), state=image)
        env.grayscale = False
        ret = env.reset()
        self.assertEqual((84, 84, 3), ret.shape)

    def test_maximum_value(self):
        image = np.ndarray((210, 160, 3), buffer=np.random.randint(0, 256, size=210 * 160 * 3), dtype=np.uint8)
        env = setup_env((84, 84), state=image)
        ret = env.reset()
        self.assertGreaterEqual(1, np.max(ret))
        self.assertLessEqual(0, np.max(ret))


if __name__ == '__main__':
    unittest.main()
