import unittest

from source.environment.atari.fire_reset import FireResetEnv
from test.helpers.environment import MockEnv, MockWrapper


class ResetTest(unittest.TestCase):
    def test_fire(self):
        mock = MockEnv(lambda: None,
                       lambda _: (None, None, False, {}),
                       ['NOOP', 'FIRE', '3'])
        env = FireResetEnv(MockWrapper(mock))
        env.reset()
        self.assertEqual(mock.step_ctr, 2)
        self.assertEqual(mock.reset_ctr, 1)

    def test_done(self):
        mock = MockEnv(lambda: None,
                       lambda _: (None, None, True, {}),
                       ['NOOP', 'FIRE', '3'])
        env = FireResetEnv(MockWrapper(mock))
        env.reset()
        self.assertEqual(mock.step_ctr, 2)
        self.assertEqual(mock.reset_ctr, 3)


if __name__ == '__main__':
    unittest.main()
