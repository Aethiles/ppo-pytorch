import numpy as np

from source.environment.serial_env import SerialEnv
from source.policy.policy import Policy
from source.policy.parameters import AtariParameters
from source.training.runner import Runner
from source.utilities.config.hyperparameters import HyperParameters
from test.helpers.environment import MockEnv, MockWrapper
from typing import Tuple
from unittest.mock import MagicMock, patch


def setup_test(state_gen, reward_gen, done_gen, horizon, num_envs=1, noop=30, max_grad_norm=None,
               env_class=SerialEnv) -> Tuple[Runner, MockEnv, Policy]:
    h = HyperParameters('')
    pol = Policy(AtariParameters(h.nn_size,
                                 h.nn_input_shape,
                                 4,
                                 learn_rate=h.nn_learn_rate,
                                 max_gradient_norm=max_grad_norm,
                                 logger_=MagicMock()))
    mock = MockEnv(lambda: next(state_gen),
                   lambda a: (next(state_gen), reward_gen(a), next(done_gen), {}))
    with patch('gym.make', return_value=MockWrapper(mock)):
        # env = ParallelEnvironments('', num_envs)
        env = env_class('', num_envs)
        if env_class is SerialEnv:
            for e in env.envs:
                e.env.env.env.env.env.env.env.no_op_max = noop
        runner = Runner(horizon=horizon,
                        policy=pol,
                        environment=env,
                        logger_=MagicMock(),
                        )
    return runner, mock, pol


def stable_state_gen(state):
    """
    Always yields state
    :param state: the state
    :return: the state
    """
    while True:
        yield state


def random_state_gen(shape):
    """
    Always yields a new random state
    :param shape: the shape of the state
    :return: the state
    """
    while True:
        yield np.ndarray(shape, buffer=np.ndarray(np.prod(shape)))


def stable_reward_gen(reward):
    """
    Always yields reward
    :param reward:
    :return:
    """
    while True:
        yield reward


def list_reward_gen(rewards):
    """
    Yields the rewards in rewards one by one.
    :param rewards:
    :return:
    """
    for reward in rewards:
        yield reward


def stable_done_gen(done):
    """
    Always yields done
    :param done: True if done, False else
    :return: done
    """
    while True:
        yield done


def done_at_n_gen(n):
    """
    Yields True exactly at nth step, False else
    :param n:
    :return:
    """
    i = 0
    while True:
        if i == n:
            yield True
        else:
            yield False
        i += 1


class RewardGenerator:
    def __init__(self,
                 good_action,
                 good_reward=1,
                 bad_reward=-1):
        self.good_action = good_action
        self.good_reward = good_reward
        self.bad_reward = bad_reward

    def reward_func(self, action):
        return self.good_reward if action == self.good_action else self.bad_reward
