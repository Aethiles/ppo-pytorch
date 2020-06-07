import numpy as np
import unittest

from source.environment.atari.state_stack import StateStackEnv
from test.helpers.environment import MockEnv, MockWrapper
from test.helpers.utils import done_at_n_gen


class ResetTest(unittest.TestCase):
    def test_reset(self):
        shape = np.random.randint(4, 16)
        state = np.ndarray((shape, shape), buffer=np.random.rand(shape ** 2), dtype=np.float)
        mock = MockEnv(lambda: state,
                       lambda _: (state, 0, False, {}),
                       [''])
        env = StateStackEnv(MockWrapper(mock), state.shape)
        ret = env.reset()
        self.assertEqual(len(ret), 4)
        self.assertTrue(np.array_equal(ret[3], state))
        self.assertFalse(np.all(ret == 0))
        self.assertTrue(np.all(ret[:2] == 0))

    def test_reset_after_multiple_steps(self):
        shape = np.random.randint(4, 16)
        states = [np.zeros((shape, shape)) + i for i in range(5)]

        def state_gen():
            for state in states:
                yield state

        gen = state_gen()
        done_gen = done_at_n_gen(2)
        mock = MockEnv(lambda: next(gen),
                       lambda _: (next(gen), 0, next(done_gen), {}))
        env = StateStackEnv(MockWrapper(mock), states[0].shape)
        env.reset()
        env.step(0)
        ret = env.step(0)
        self.assertTrue(np.all(ret[0][-1] == 2))
        self.assertTrue(np.all(ret[0][-2] == 1))
        ret = env.step(0)
        self.assertTrue(ret[2])
        ret = env.reset()
        self.assertTrue(np.all(ret[-1] == 4))
        self.assertTrue(np.all(ret[-2] == 0))
        self.assertTrue(np.all(ret[-3] == 0))


class StepTest(unittest.TestCase):
    @staticmethod
    def setup_states(shape):
        return [np.ndarray((shape, shape), buffer=np.random.rand(shape ** 2), dtype=np.float) for _ in range(5)]

    def test_steps(self):
        shape = np.random.randint(4, 16)
        states = self.setup_states(shape)

        def state_generator():
            for i in range(5):
                yield states[i]

        gen = state_generator()
        mock = MockEnv(lambda: next(gen),
                       lambda _: (next(gen), 0, False, {}),
                       [''])
        env = StateStackEnv(MockWrapper(mock), (shape, shape))
        ret = env.reset()
        self.assertTrue(np.array_equal(ret[3], states[0]))
        ret = env.step(0)
        self.assertTrue(np.array_equal(ret[0][2], states[0]))
        self.assertTrue(np.array_equal(ret[0][3], states[1]))
        ret = env.step(0)
        self.assertTrue(np.array_equal(ret[0][1], states[0]))
        self.assertTrue(np.array_equal(ret[0][2], states[1]))
        self.assertTrue(np.array_equal(ret[0][3], states[2]))
        ret = env.step(0)
        self.assertTrue(np.array_equal(ret[0][0], states[0]))
        self.assertTrue(np.array_equal(ret[0][1], states[1]))
        self.assertTrue(np.array_equal(ret[0][2], states[2]))
        self.assertTrue(np.array_equal(ret[0][3], states[3]))
        ret = env.step(0)
        self.assertTrue(np.array_equal(ret[0][0], states[1]))
        self.assertTrue(np.array_equal(ret[0][1], states[2]))
        self.assertTrue(np.array_equal(ret[0][2], states[3]))
        self.assertTrue(np.array_equal(ret[0][3], states[4]))


if __name__ == '__main__':
    unittest.main()
