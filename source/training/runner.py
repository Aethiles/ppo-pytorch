import torch

from source.environment.stack import StackEnv
from source.policy.policy import Policy
from source.training.rollout import Rollout
from source.utilities.logging.abstract_logger import AbstractLogger


class Runner:
    def __init__(self,
                 horizon: int,
                 policy: Policy,
                 environment: StackEnv,
                 logger_: AbstractLogger,
                 ):
        """
        Runs the environments for horizon time steps generating a rollout class containing the observed data.
        :param horizon: number of time steps the training shall take
        :param policy: the policy used for generating the training
        :param environment: the gym environment the policy is trained on
        :param logger_: logger used to log terminated episode rewards and cumulative rewards
        """
        self.horizon = horizon
        self.policy = policy
        self.environment = environment
        self.state = self.environment.reset()

        self.logger = logger_

    def run(self) -> Rollout:
        """
        Runs the policy and environment for horizon time steps or until the environment is done. Records and returns a
        Rollout of the episode.
        :return: Rollout of the episode
        """
        rollout = Rollout(self.environment.num_envs, self.horizon, self.state, self.policy.device)
        for time_step in range(self.horizon):
            with torch.no_grad():
                action, log_prob, value = self.policy.get_action_and_value(rollout.states[time_step])
            self.state, reward, dones, infos = self.environment.step(action.cpu().numpy())
            rollout.save_time_step(self.state, reward, action, log_prob, value, dones)
            self.logger.log_step(dones, infos, reward)

        with torch.no_grad():
            _, value = self.policy.parameters(rollout.states[-1])
        rollout.finalize(value.squeeze())
        return rollout
