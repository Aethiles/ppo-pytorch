import numpy as np
import time
import torch

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple


class StackEnv(ABC):
    def __init__(self,
                 name: str,
                 num_envs: int = 8,
                 seed: int = int(time.time())):
        self.name = name
        self.envs = None
        self.action_space = None
        self.observation_space = None

        self.seed = seed
        self.num_envs = num_envs

    @abstractmethod
    def step(self,
             actions: torch.Tensor,
             ) -> Tuple[np.ndarray, List[float], List[bool], List[Dict]]:
        """
        Performs the given actions in their respective environments.
        :param actions: the actions
        :return: observations following the actions
        """
        pass

    @abstractmethod
    def reset(self) -> np.ndarray:
        """
        Resets all environments
        :return: observed states
        """
        pass

    @abstractmethod
    def close(self):
        """
        Closes all environments
        :return: None
        """
        pass
