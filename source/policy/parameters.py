import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from enum import auto, Enum
from source.utilities.logging.abstract_logger import AbstractLogger
from source.utilities.neural_network import get_output_shape, orthogonal
from typing import Tuple


class AtariSize(Enum):
    SMALL = auto()
    LARGE = auto()


class AtariParameters(nn.Module):
    """
    This class contains commonly used setups for learning Atari games. Valid configurations are:
    - AtariSize.SMALL as described in 'Playing Atari with Deep Reinforcement Learning' by Mnih et al., 2013
    - AtariSize.LARGE as described in 'Human-level control through deep reinforcement learning' by Mnih et al., 2015
    """
    def __init__(self,
                 size: AtariSize,
                 input_shape: Tuple[int, int, int],
                 output_size: int,
                 learn_rate: float,
                 logger_: AbstractLogger,
                 adam_epsilon: float = 1e-5,
                 max_gradient_norm: float = None,
                 device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                 ):
        """
        :param size: determines the number of convolutional layers used, the number of output channels and the size of
            the hidden linear layer
        :param input_shape: the shape of an observation (channels, height, width)
        :param output_size: the number of valid actions
        :param learn_rate: the initial learn rate
        :param max_gradient_norm: the value the l2 norm is clipped to
        :param device: the torch.device to use in execution and learning
        """
        super(AtariParameters, self).__init__()
        self.size = size
        if size is AtariSize.SMALL:
            conv_output = get_output_shape(get_output_shape(input_shape[1:], (8, 8), (4, 4)), (4, 4), (2, 2))
            self.linear_shape = 32 * int(np.prod(conv_output))
            linear_output = 256

            self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=(8, 8), stride=(4, 4))
            self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(4, 4), stride=(2, 2))
            self.linear = nn.Linear(self.linear_shape, linear_output)
        elif size is AtariSize.LARGE:
            conv_output = get_output_shape(get_output_shape(get_output_shape(input_shape[1:], (8, 8), (4, 4)),
                                                            (4, 4), (2, 2)), (3, 3), (1, 1))
            self.linear_shape = 64 * int(np.prod(conv_output))
            linear_output = 512

            self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=(8, 8), stride=(4, 4))
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=(2, 2))
            self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1))
            self.linear = nn.Linear(self.linear_shape, linear_output)
        else:
            raise Exception('Invalid size')

        self.value_output = nn.Linear(linear_output, 1)
        self.action_output = nn.Linear(linear_output, output_size)

        self.device = device
        self.init_weights()
        self.to(device)
        self.initial_learn_rate = learn_rate
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learn_rate, eps=adam_epsilon)
        self.max_gradient_norm = max_gradient_norm
        self.logger = logger_

    def init_weights(self):
        orthogonal(self.conv1, gain='relu')
        orthogonal(self.conv2, gain='relu')
        if self.size is AtariSize.LARGE:
            orthogonal(self.conv3, gain='relu')
        orthogonal(self.linear, gain='relu')
        orthogonal(self.value_output, gain=1)
        orthogonal(self.action_output, gain=0.01)

    def forward(self,
                state: torch.Tensor,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass returning logits of actions as well as the value of the given state
        :param state: the state that shall be evaluated
        :return: action logits, state value
        """
        # state = state.reshape((1, state.shape[0], state.shape[1], state.shape[2]))
        state = F.relu(self.conv1(state))
        state = F.relu(self.conv2(state))
        if self.size is AtariSize.LARGE:
            state = F.relu(self.conv3(state))
        # Baselines flattens all dimensions but the first (to preserve image batch possibly?)
        state = state.view(-1, self.linear_shape)
        state = F.relu(self.linear(state))
        value = self.value_output(state)
        action_logits = self.action_output(state)
        return action_logits, value

    def gradient_update(self,
                        loss: torch.Tensor,
                        ):
        """
        Performs a gradient update with the given loss.
        :param loss: the loss
        :return: None
        """
        self.optimizer.zero_grad()
        loss.backward()
        self._log_gradient_norm()
        if self.max_gradient_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_gradient_norm)
        self.optimizer.step()

    def _log_gradient_norm(self):
        total_norm = 0
        for param in self.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1 / 2)
        self.logger.store_gradient_norm(total_norm)

    def change_learn_rate_by(self,
                             factor: float,
                             ):
        """
        Changes the Adam optimizer's learn rate by the given factor
        :param factor: the factor
        :return: None
        """
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.initial_learn_rate * factor
