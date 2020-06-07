import numpy as np
import unittest

from source.environment.atari.action_repetition import ActionRepetitionEnv
from test.helpers.environment import MockEnv, MockWrapper


class StepTest(unittest.TestCase):
    def test_k_skipped_frames(self):
        def state_generator(iterations=5):
            for i in range(iterations):
                if i == 3:
                    yield np.arange(8) + 2
                else:
                    yield np.arange(8)

        gen = state_generator()
        mock = MockEnv(lambda: next(gen),
                       lambda _: (next(gen), 1, False, {}),
                       [''])
        mock.observation_space = np.zeros(8)
        env = ActionRepetitionEnv(MockWrapper(mock))
        env.reset()
        ret = env.step(0)
        self.assertTrue(np.array_equal(ret[0], np.arange(8) + 2))
        self.assertEqual(mock.step_ctr, 4)
        self.assertEqual(ret[1], 4)
        self.assertFalse(ret[2])

    def test_max_frame(self):
        def state_generator(iterations=5):
            for i in range(iterations):
                if i == 3:
                    s = np.arange(8)
                    s[0] = 8
                    yield s
                else:
                    yield np.arange(8)

        gen = state_generator()
        mock = MockEnv(lambda: next(gen),
                       lambda _: (next(gen), 0.2, False, {}),
                       [''])
        mock.observation_space = np.zeros(8)
        env = ActionRepetitionEnv(MockWrapper(mock))
        env.reset()
        ret = env.step(0)
        state = np.arange(8)
        state[0] = 8
        self.assertTrue(np.array_equal(ret[0], state))
        self.assertEqual(ret[1], 0.2 * 4)
        self.assertEqual(mock.step_ctr, 4)

    def test_abort_on_done(self):
        abort_on = np.random.randint(1, 4)

        def done_generator(iterations=5):
            for i in range(iterations):
                if i == abort_on:
                    yield True
                else:
                    yield False

        gen = done_generator()
        mock = MockEnv(lambda: np.random.rand(8),
                       lambda _: (np.random.rand(8), 0, next(gen), {}),
                       [''])
        mock.observation_space = np.zeros(8)
        env = ActionRepetitionEnv(MockWrapper(mock))
        env.reset()
        ret = env.step(0)
        self.assertEqual(mock.step_ctr, abort_on + 1)
        self.assertTrue(ret[2])


if __name__ == '__main__':
    unittest.main()
