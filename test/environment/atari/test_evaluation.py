import unittest

from source.environment.atari.evaluation import EvaluationEnv
from test.helpers.environment import MockWrapper, MockEnv
from test.helpers.utils import stable_done_gen, done_at_n_gen


class EvaluationTest(unittest.TestCase):
    def setup_test(self, done_gen):
        mock = MockEnv(lambda: None,
                       lambda _: (None, 1, next(done_gen), {}))
        return EvaluationEnv(MockWrapper(mock)), mock

    def test_not_done(self):
        env, mock = self.setup_test(stable_done_gen(False))
        for i in range(25):
            _, _, done, info = env.step(0)
            self.assertFalse(done)
            self.assertDictEqual({}, info)

    def test_one_done(self):
        env, mock = self.setup_test(done_at_n_gen(6))
        done = False
        info = {}
        for i in range(7):
            _, _, done, info = env.step(0)
        self.assertTrue(done)
        self.assertTrue('episode' in info)
        self.assertEqual(7, info.get('episode').reward)


if __name__ == '__main__':
    unittest.main()
