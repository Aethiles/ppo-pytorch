import numpy as np
import time

from source.environment.atari.setup import setup_environment
from typing import Dict, List, Tuple
from source.environment.stack import StackEnv


class SerialEnv(StackEnv):
    def __init__(self,
                 name: str,
                 num_envs: int = 8,
                 seed: int = int(time.time()),
                 ):
        """
        Wraps multiple environments with different seeds.
        :param name: the name of the OpenAI gym environment
        :param num_envs: the number of environments to run
        :param seed: the seed to use
        """
        super().__init__(name, num_envs, seed)
        self.envs = [setup_environment(name, seed + i) for i in range(num_envs)]
        self.action_space = self.envs[0].action_space
        self.observation_space = self.envs[0].observation_space

    def step(self,
             actions: List[int],
             ) -> Tuple[np.ndarray, List[float], List[bool], List[Dict]]:
        """
        Steps all inner environments and returns a list of the observed states, rewards, dones and infos.
        :param actions: the actions
        :return: states, rewards, dones, infos
        """
        states = np.zeros((self.num_envs,) + self.observation_space.shape)
        rewards = []
        dones = []
        infos = []
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            state, reward, done, info = env.step(action)
            if done:
                state = env.reset()
            states[i] = state
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        return states, rewards, dones, infos

    def reset(self,
              **kwargs,
              ) -> np.ndarray:
        """
        Resets the inner environments.
        :param kwargs:
        :return: a list of observed states
        """
        states = np.zeros((self.num_envs,) + self.observation_space.shape)
        for i, env in enumerate(self.envs):
            states[i] = env.reset(**kwargs)
        return states

    def close(self):
        """
        Closes all environments.
        :return:
        """
        for env in self.envs:
            env.close()
