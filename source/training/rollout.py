import numpy as np
import torch

from typing import List, Tuple, Union


class Rollout:
    def __init__(self,
                 num_environments: int,
                 horizon: int,
                 initial_state: np.ndarray,
                 device: torch.device,
                 ):
        """
        This class is used to store an agent's trajectory through its environment as well as the probabilities of all
        actions taken by the agent and the values of all states encountered.
        :param num_environments: number of environments run in parallel
        :param horizon: length of the trajectory
        :param initial_state: the state the trajectory begins in
        """
        self.horizon = horizon
        self.num_envs = num_environments
        self.time_step = 0
        self.states: torch.Tensor = torch.zeros((horizon + 1, num_environments) + initial_state.shape[1:],
                                                device=device)
        self.states[0] = torch.tensor(initial_state, device=device)
        self.actions: torch.Tensor = torch.zeros((horizon, num_environments), device=device)
        self.log_action_probabilities: torch.Tensor = torch.zeros((horizon, num_environments), device=device)
        self.rewards: torch.Tensor = torch.zeros((horizon, num_environments), device=device)
        self.values: torch.Tensor = torch.zeros((horizon + 1, num_environments), device=device)
        self.advantages: torch.Tensor = torch.empty(0, device=device)
        self.returns: torch.Tensor = torch.empty(0, device=device)
        self.masks: torch.Tensor = torch.zeros((horizon, num_environments), device=device)

    def save_time_step(self,
                       state: np.ndarray,
                       reward: List[float],
                       action: torch.Tensor,
                       log_action_probability: torch.Tensor,
                       value: torch.Tensor,
                       done: List[bool],
                       ):
        """
        Saves the given action and its logarithmic probability as well as the observed reward and state after taking
        this action. Stores the value of the state the action was taken in, too.
        :param state: the state observed after taking the action
        :param reward: the reward observed after taking the action
        :param action: the action taken
        :param log_action_probability: the logarithmic probability of the action
        :param value: the value of the previous state
        :param done: True if the environment reached a terminal state, else False
        :return: None
        """
        # TODO check these clones
        self.states[self.time_step + 1] = torch.tensor(state)
        self.rewards[self.time_step] = torch.tensor(reward)
        self.actions[self.time_step] = action.clone()
        self.log_action_probabilities[self.time_step] = log_action_probability.clone()
        self.values[self.time_step] = value.clone()
        self.masks[self.time_step] = torch.tensor(np.invert(done))
        self.time_step += 1

    def finalize(self,
                 value: torch.Tensor,
                 ):
        """
        Finalizes by saving the value of the final state, scaling the rewards and transferring all data to the device
        that PyTorch modules are run on. If the episode ended prematurely, the surplus elements will be sliced off.
        :param value: the value of the final state
        :return: None
        """
        self.values[self.time_step] = value.clone()

    def get_flattened_states_actions_probs(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns all states and actions after flattening them.
        :return:
        """
        return self.states.view((self.states.shape[0] * self.states.shape[1], ) + self.states.shape[2:]), \
            self.actions.flatten(), \
            self.log_action_probabilities.flatten()

    def __getitem__(self,
                    item: Union[int, List[int], np.ndarray],
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Flattens the rollouts and discards the last value and state.
        :param item: the item or items to get
        :return: a tuple consisting of states, actions, log_action_probabilities, values, advantages, returns
        """
        return self.states[:-1].view(-1, *self.states.size()[2:])[item], \
            self.actions.flatten()[item], \
            self.log_action_probabilities.flatten()[item], \
            self.values[:-1].flatten()[item], \
            self.advantages.flatten()[item], \
            self.returns.flatten()[item]
