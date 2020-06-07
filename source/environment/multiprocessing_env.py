import multiprocessing as mp
import numpy as np
import time

from enum import Enum, auto
from source.environment.atari.setup import setup_environment
from source.environment.stack import StackEnv
from typing import Dict, List, Tuple


class Command(Enum):
    STEP = auto()
    RESET = auto()
    CLOSE = auto()
    GET_CONFIG = auto()


def run_env(env_name: str,
            seed: int,
            pipe: mp.Pipe,
            ):
    """

    :param env_name:
    :param seed:
    :param pipe:
    :return: None
    """
    env = setup_environment(env_name, seed)

    while True:
        command, args = pipe.recv()

        if command == Command.STEP:
            state, reward, done, info = env.step(args)
            if done:
                state = env.reset()
            pipe.send((state, reward, done, info))
        elif command == Command.RESET:
            state = env.reset()
            pipe.send(state)
        elif command == Command.CLOSE:
            env.close()
            break
        elif command == Command.GET_CONFIG:
            pipe.send((env.action_space, env.observation_space))

    pipe.close()


class MultiprocessingEnv(StackEnv):
    def __init__(self,
                 name: str,
                 num_envs: int = 8,
                 seed: int = int(time.time()),
                 ):
        """

        :param name:
        :param num_envs:
        :param seed:
        """
        super().__init__(name, num_envs, seed)
        self.processes = []
        self.pipes = []

        for i in range(num_envs):
            pipe = mp.Pipe()
            self.pipes.append(pipe[0])
            process = mp.Process(target=run_env, args=(name, seed + i, pipe[1]))
            process.daemon = True
            self.processes.append(process)
            process.start()

        self.pipes[0].send((Command.GET_CONFIG, None))
        config = self.pipes[0].recv()
        self.action_space = config[0]
        self.observation_space = config[1]

    def _broadcast(self,
                   command: Command,
                   actions: List[int] = None,
                   ):
        """

        :param command:
        :param actions:
        :return: None
        """
        for i, pipe in enumerate(self.pipes):
            if actions is not None:
                pipe.send((command, actions[i]))
            else:
                pipe.send((command, None))

    def _gather(self) -> List[any]:
        """

        :return:
        """
        return [pipe.recv() for pipe in self.pipes]

    def step(self,
             actions: List[int],
             ) -> Tuple[np.ndarray, List[float], List[bool], List[Dict]]:
        """
        Steps all inner environments and returns a list of the observed states, rewards, dones and infos.
        :param actions: the actions
        :return:
        """
        self._broadcast(Command.STEP, actions)
        observations = self._gather()
        states, rewards, dones, infos = list(map(list, zip(*observations)))
        return np.array(states), rewards, dones, infos

    def reset(self) -> np.ndarray:
        """

        :return:
        """
        self._broadcast(Command.RESET)
        return np.array(self._gather())

    def close(self):
        """

        :return:
        """
        self._broadcast(Command.CLOSE)
        for pipe, process in zip(self.pipes, self.processes):
            pipe.close()
            process.join()
