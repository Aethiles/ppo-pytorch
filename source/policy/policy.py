import torch
import torch.nn as nn

from source.utilities.errors import ShapeError
from torch.distributions import Categorical
from typing import Tuple


class Policy:
    """
    This class represents the agent's policy. As the optimal policy must be approximated, this class contains the
    necessary parametrization and a method to perform gradient updates. Furthermore, the action can be prompted to
    select an action.
    """
    def __init__(self,
                 parameters: nn.Module,
                 ):
        """
        New Policy instance
        :param parameters: the parametrization of the Policy
        """
        self.device = parameters.device
        self.parameters = parameters

    def get_action_and_value(self,
                             state: torch.Tensor,
                             ) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Selects an action by sampling the current probability distribution of actions given a state.
        :param state: the state of the environment
        :return: the action and the natural logarithm of its probability as well as the current state's value
        """
        action_logits, value = self.parameters(state)
        distribution = Categorical(logits=action_logits)
        action = distribution.sample()
        action_log_probability = distribution.log_prob(action)
        return action, action_log_probability, value.squeeze()

    def evaluate(self,
                 states: torch.Tensor,
                 actions: torch.Tensor,
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluates the given actions by determining their probability in their respective states. Evaluates the given
        states by feeding them to the parameters. Also returns the entropy of the action distributions generated for
        each state.
        :param states: observed states
        :param actions: taken actions
        :return: this policy's probabilities, values and the entropy of the action distribution
        """
        if states.shape[0] != actions.shape[0]:
            raise ShapeError('Expected same number of states ({}) as number of actions ({})'
                             .format(states.shape[0], actions.shape[0]))
        probabilities, values = self.parameters(states)
        distribution = Categorical(logits=probabilities)
        log_action_probabilities = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return log_action_probabilities, values.squeeze(-1), entropy

    def calculate_kl_divergence(self,
                                states: torch.Tensor,
                                actions: torch.Tensor,
                                other_log_probabilities: torch.Tensor,
                                ) -> float:
        """

        :param states:
        :param actions:
        :param other_log_probabilities:
        :return:
        """
        log_probabilities, _, _ = self.evaluate(states, actions)
        return (other_log_probabilities - log_probabilities).mean().item()
