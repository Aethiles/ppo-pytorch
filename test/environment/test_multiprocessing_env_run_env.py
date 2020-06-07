import multiprocessing as mp
import numpy as np
import source.environment.multiprocessing_env as subprocess
import unittest

from unittest.mock import MagicMock, patch


def setup_test(num_processes, state, reward=0.0, done=False, info=None):
    """
    Sets up num_processes processes and pipes
    :param num_processes:
    :param state:
    :param reward:
    :param done:
    :param info:
    :return:
    """
    pipes = []
    processes = []
    mock = MagicMock()
    mock.reset = MagicMock()
    mock.reset.return_value = state
    mock.step = MagicMock()
    mock.step.return_value = (state, reward, done, info)
    for i in range(num_processes):
        pipe = mp.Pipe()
        pipes.append(pipe[0])
        with patch('test.environment.test_multiprocessing_env_run_env.subprocess.setup_environment',
                   return_value=mock):
            process = mp.Process(target=subprocess.run_env, args=('', 0, pipe[1]))
            processes.append(process)
            process.start()
    return processes, pipes, mock


def clean_up(processes, pipes):
    for process, pipe in zip(processes, pipes):
        pipe.send((subprocess.Command.CLOSE, None))
        process.join()
        pipe.close()


class StepTest(unittest.TestCase):
    def test_steps_multiple_workers(self):
        state = np.random.rand(4, 8, 8)
        processes, pipes, mock = setup_test(8, state)
        for pipe in pipes:
            pipe.send((subprocess.Command.STEP, 0))
            ret = pipe.recv()
            self.assertTrue(np.array_equal(ret[0], state))
            self.assertEqual(0.0, ret[1])
            self.assertEqual(False, ret[2])
            self.assertEqual(None, ret[3])
        clean_up(processes, pipes)

    def test_steps(self):
        state = np.random.rand(4, 8, 8)
        processes, pipes, mock = setup_test(1, state)
        for _ in range(20):
            pipes[0].send((subprocess.Command.STEP, 0))
            ret = pipes[0].recv()
            self.assertTrue(np.array_equal(ret[0], state))
            self.assertEqual(0.0, ret[1])
            self.assertEqual(False, ret[2])
            self.assertEqual(None, ret[3])
        clean_up(processes, pipes)


class ResetTest(unittest.TestCase):
    def test_reset(self):
        state = np.random.rand(4, 8, 8)
        processes, pipes, mock = setup_test(1, state)
        pipes[0].send((subprocess.Command.RESET, None))
        ret = pipes[0].recv()
        self.assertTrue(np.array_equal(state, ret))
        clean_up(processes, pipes)

    def test_resets(self):
        state = np.random.rand(4, 8, 8)
        processes, pipes, mock = setup_test(8, state)
        for pipe in pipes:
            pipe.send((subprocess.Command.RESET, None))
            ret = pipe.recv()
            self.assertTrue(np.array_equal(state, ret))
        clean_up(processes, pipes)


class CloseTest(unittest.TestCase):
    @staticmethod
    def test_closes():
        processes, pipes, mock = setup_test(5, None)
        for pipe in pipes:
            pipe.send((subprocess.Command.CLOSE, None))
            pipe.close()
        for process in processes:
            process.join()


if __name__ == '__main__':
    unittest.main()
