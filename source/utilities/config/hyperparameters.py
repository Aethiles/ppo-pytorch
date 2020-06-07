import socket

from dataclasses import dataclass
from source.policy.parameters import AtariSize
from typing import Tuple


@dataclass
class HyperParameters:
    env_name: str
    hostname: str = socket.gethostname()

    # Neural network parameters
    nn_size: AtariSize = AtariSize.LARGE
    nn_learn_rate: float = 2.5 * 1e-4
    nn_adam_eps: float = 1e-5
    nn_input_shape: Tuple[int, int, int] = (4, 84, 84)
    nn_max_gradient_norm: float = 0.5

    # Episode etc
    max_time_steps: int = 10_000_000
    horizon: int = 128

    # GAE
    gae_discount: float = 0.99
    gae_weight: float = 0.95

    # PPO
    ppo_num_envs: int = 8
    ppo_epochs: int = 4
    ppo_num_mini_batches: int = 4
    ppo_clipping_parameter: float = 0.1
    ppo_value_function_coeff: float = 0.5
    ppo_entropy_coeff: float = 0.01
    ppo_use_linear_decay: bool = True

    # Seeds
    torch_seed: int = -1
    atari_seed: int = -1

    def __str__(self):
        return 'Env name     : {}  \n' \
               'Model size   : {}  \n' \
               'Learn rate   : {}  \n' \
               'Time steps   : {}  \n' \
               'Horizon      : {}  \n' \
               'Num envs     : {}  \n' \
               'Max grad norm: {}  \n' \
               'GAE γ        : {}  \n' \
               'GAE λ        : {}  \n' \
               'PPO epochs   : {}  \n' \
               'PPO batches  : {}  \n' \
               'PPO ɛ        : {}  \n' \
               'PPO val coeff: {}  \n' \
               'PPO ent coeff: {}  \n' \
               'PPO annealing: {}  \n' \
               'Torch seed   : {}  \n' \
               'Atari seed   : {}' \
            .format(self.env_name.replace('NoFrameskip', ''),
                    self.nn_size.name,
                    self.nn_learn_rate,
                    self.max_time_steps,
                    self.horizon,
                    self.ppo_num_envs,
                    self.nn_max_gradient_norm,
                    self.gae_discount,
                    self.gae_weight,
                    self.ppo_epochs,
                    self.ppo_num_mini_batches,
                    self.ppo_clipping_parameter,
                    self.ppo_value_function_coeff,
                    self.ppo_entropy_coeff,
                    self.ppo_use_linear_decay,
                    self.torch_seed,
                    self.atari_seed)

    def as_markdown(self):
        string = '| Parameter | Value |  \n| ------ | ------ |  \n'
        for key, value in self.__dict__.items():
            string += '| {} | {} |  \n'.format(key, value)
        return string
