import torch
import torch.utils.data.sampler as samplers

from source.training.rollout import Rollout
from source.policy.parameters import AtariParameters
from source.policy.policy import Policy
from source.utilities.errors import ShapeError
from source.utilities.logging.abstract_logger import AbstractLogger
from source.utilities.maths import normalize_tensor


class ProximalPolicyOptimization:
    def __init__(self,
                 policy: Policy,
                 parameters: AtariParameters,
                 logger_: AbstractLogger,
                 num_envs: int,
                 num_mini_batches: int = 4,
                 epochs: int = 4,
                 clip_range: float = 0.1,
                 value_function_coeff: float = 1,
                 entropy_coeff: float = 0.01,
                 discount: float = 0.99,
                 gae_weight: float = 0.95,
                 max_updates: int = None,
                 use_linear_decay: bool = True,
                 ):
        """
        This class offers a customisable implementation of Schulman et al. Proximal Policy Optimization. It trains the
        given policy using the shared parameters configuration loss.
        :param policy: the target policy being trained
        :param parameters: the parametrization of the policy
        :param logger_: the logger used to track losses and hyperparameter annealing
        :param num_envs: the number of environments used in rollout generation
        :param num_mini_batches: the number of mini batches to perform per epoch
        :param epochs: number of times to perform gradient updates
        :param clip_range: epsilon from PPO CLIP, limits the probability ratio and value function loss
        :param value_function_coeff: coefficient controlling the influence of the value function loss
        :param entropy_coeff: coefficient controlling the influence of the entropy bonus
        :param discount: the discount factor applied to future values/advantages
        :param gae_weight: delta from GAE, used to weigh and normalize advantages
        :param max_updates: number of times the PPO algorithm is performed
        :param use_linear_decay: True if learn rate and clipping parameter shall decay linearly hitting 0 at the end of
                                 the training
        """
        self.policy = policy
        self.parameters = parameters
        self.num_envs = num_envs
        self.num_mini_batches = num_mini_batches
        self.epochs = epochs
        self.initial_clip_range = clip_range
        self.clip_range = clip_range
        self.value_function_coeff = value_function_coeff
        self.entropy_coeff = entropy_coeff
        self.discount = discount
        self.gae_weight = gae_weight
        self.updates = 0
        self.use_linear_decay = use_linear_decay
        self.max_updates = max_updates
        self.logger = logger_

    def train(self,
              rollout: Rollout,
              ):
        """
        Calculates a loss from the given rollout and trains a policy with it for self.epochs.
        :param rollout: the rollout the policy is to be trained on
        :return: None
        """
        advantages = self._calculate_advantages(rollout.rewards,
                                                rollout.values,
                                                rollout.masks)
        rollout.returns = self._calculate_returns(advantages, rollout.values[:-1])
        rollout.advantages = normalize_tensor(advantages)

        mini_batch_size = int(rollout.time_step / self.num_mini_batches) * self.num_envs
        sampler = samplers.BatchSampler(sampler=samplers.SubsetRandomSampler(range(rollout.time_step * self.num_envs)),
                                        batch_size=mini_batch_size,
                                        drop_last=True)

        for _ in range(self.epochs):
            for batch in sampler:
                loss = self.calculate_shared_parameters_loss(*rollout[batch])
                # Do gradient update with -loss as PyTorch performs gradient descent
                # whereas we want to perform a gradient ascent
                self.parameters.gradient_update(-loss)
        states, actions, old_log_action_probabilities = rollout.get_flattened_states_actions_probs()
        kl_divergence = self.policy.calculate_kl_divergence(states[:-self.num_envs],
                                                            actions,
                                                            old_log_action_probabilities)
        self.logger.log_kl_divergence(kl_divergence)
        self.updates += 1
        if self.use_linear_decay:
            self._anneal_hyper_parameters()

    def _anneal_hyper_parameters(self):
        """
        Progresses the annealing of PPO's clip parameter and the Adam optimizer's learn rate.
        :return:
        """
        self.logger.log_hyperparameters(self.parameters.optimizer.param_groups[0]['lr'],
                                        self.clip_range)
        multiplier = 1 - self.updates / self.max_updates
        self.clip_range = self.initial_clip_range * multiplier
        self.parameters.change_learn_rate_by(multiplier)

    def calculate_shared_parameters_loss(self,
                                         states: torch.Tensor,
                                         actions: torch.Tensor,
                                         old_log_probabilities: torch.Tensor,
                                         old_values: torch.Tensor,
                                         advantages: torch.Tensor,
                                         returns: torch.Tensor,
                                         ) -> torch.Tensor:
        """
        Calculates a loss for a shared parameters configuration.
        :param states: states to evaluate with the new policy
        :param actions: actions taken by the old policy in the given states
        :param old_log_probabilities: probabilities of the actions taken by the old policy
        :param old_values: values estimated for the given states by the old policy
        :param advantages: advantages of the given actions
        :param returns: returns of the given states
        :return: the loss
        """
        new_log_probabilities, new_values, entropy = self.policy.evaluate(states=states,
                                                                          actions=actions)
        entropy_bonus = torch.mean(entropy)
        policy_loss = self.calculate_policy_loss(advantages=advantages,
                                                 new_log_probabilities=new_log_probabilities,
                                                 old_log_probabilities=old_log_probabilities)
        value_function_loss = self.calculate_value_function_loss(returns=returns,
                                                                 new_values=new_values,
                                                                 old_values=old_values)
        loss = policy_loss - self.value_function_coeff * value_function_loss + self.entropy_coeff * entropy_bonus
        self.logger.store_ppo_losses(loss, policy_loss, value_function_loss, entropy_bonus)
        return loss

    def calculate_policy_loss(self,
                              advantages: torch.Tensor,
                              new_log_probabilities: torch.Tensor,
                              old_log_probabilities: torch.Tensor
                              ) -> torch.Tensor:
        """
        Calculates a clipped loss for actions.
        :param advantages: the advantages of all actions taken during an episode
        :param new_log_probabilities: current logarithmic probabilities of all actions taken during an episode
        :param old_log_probabilities: previous logarithmic probabilities of all actions taken during an episode
        :return: the action loss
        """
        if advantages.shape != new_log_probabilities.shape or advantages.shape != old_log_probabilities.shape:
            raise ShapeError('Shapes of advantages ({}), new_log_probabilities ({}) and old_log_probabilities ({}) must'
                             'be identical.'
                             .format(advantages.shape, new_log_probabilities.shape, old_log_probabilities.shape))
        probability_ratio = torch.exp(new_log_probabilities - old_log_probabilities)
        policy_loss = advantages * probability_ratio
        clipped_policy_loss = advantages * torch.clamp(probability_ratio,
                                                       min=1 - self.clip_range,
                                                       max=1 + self.clip_range)
        loss = torch.mean(torch.min(policy_loss, clipped_policy_loss))
        return loss

    def calculate_value_function_loss(self,
                                      returns: torch.Tensor,
                                      new_values: torch.Tensor,
                                      old_values: torch.Tensor,
                                      ) -> torch.Tensor:
        """
        Calculates the value function loss by computing the mean squared error. Loss may be clipped.
        :param returns: returns of each time step
        :param new_values: the current values of observed states in the batch
        :param old_values: the old values of observed states in the batch
        :return: the value function loss
        """
        if returns.shape != new_values.shape or returns.shape != old_values.shape:
            raise ShapeError('Shapes of rewards ({}), new_values ({}) and old_values ({}) must be identical.'
                             .format(returns.shape, new_values.shape, old_values.shape))
        losses = (new_values - returns) ** 2
        clipped_values = old_values + (new_values - old_values).clamp(min=-self.clip_range, max=self.clip_range)
        clipped_losses = (clipped_values - returns) ** 2
        squared_error = 0.5 * torch.mean(torch.max(losses, clipped_losses))
        return squared_error

    def _calculate_advantages(self,
                              rewards: torch.Tensor,
                              values: torch.Tensor,
                              masks: torch.Tensor,
                              ) -> torch.Tensor:
        """
        Calculates Generalized Advantage Estimations (GAE) as derived by Schulman et al. (2016)
        :param rewards: rewards observed during play
        :param values: values of states
        :param masks: masks used to reset advantage calculation after a terminal state
        :return: the advantages
        """
        if rewards.shape != masks.shape:
            raise ShapeError('Shapes of rewards ({}) and masks ({}) must be identical.'
                             .format(rewards.shape, masks.shape))
        if rewards.shape[0] != (values.shape[0] - 1) or rewards.shape[1:] != values.shape[1:]:
            raise ShapeError('There must be one more value than rewards and masks and remaining shape must be'
                             ' identical, but shapes are {} (values) and {} (rewards, masks).'
                             .format(values.shape, rewards.shape))
        horizon = rewards.shape[0]
        advantages = torch.zeros((horizon, self.num_envs), device=rewards.device)
        deltas = rewards + masks * self.discount * values[1:] - values[:-1]
        old_advantage = 0
        for t in reversed(range(horizon)):
            advantages[t] = deltas[t] + masks[t] * self.discount * self.gae_weight * old_advantage
            old_advantage = advantages[t]
        return advantages

    @staticmethod
    def _calculate_returns(advantages: torch.Tensor,
                           values: torch.Tensor,
                           ) -> torch.Tensor:
        """
        Calculates returns from the given advantages and values.
        :param advantages:
        :param values:
        :return: the returns
        """
        if advantages.shape != values.shape:
            raise ShapeError('Shapes of advantages ({}) and values ({}) must be identical.'
                             .format(advantages.shape, values.shape))
        return advantages + values
