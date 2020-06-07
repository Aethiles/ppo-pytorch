import math
import source.utilities.config.config_builder as config_builder
import sys
import torch

from source.environment.multiprocessing_env import MultiprocessingEnv
from source.policy.parameters import AtariParameters
from source.policy.policy import Policy
from source.training.runner import Runner
from source.training.ppo import ProximalPolicyOptimization
from source.utilities.logging.tensorboard_logger import TensorboardLogger
from source.utilities.print import print_message


def main(print_to_file: bool = False):
    config = config_builder.get_hyperparameters()
    episodes = math.floor(config.max_time_steps / (config.horizon * config.ppo_num_envs))
    env = MultiprocessingEnv(name=config.env_name,
                             num_envs=config.ppo_num_envs)
    config.torch_seed = torch.initial_seed()
    config.atari_seed = env.seed
    logger = TensorboardLogger(config)
    theta = AtariParameters(size=config.nn_size,
                            input_shape=config.nn_input_shape,
                            output_size=env.action_space.n,
                            learn_rate=config.nn_learn_rate,
                            max_gradient_norm=config.nn_max_gradient_norm,
                            adam_epsilon=config.nn_adam_eps,
                            logger_=logger)
    pi = Policy(parameters=theta)
    r = Runner(horizon=config.horizon,
               policy=pi,
               environment=env,
               logger_=logger)
    ppo = ProximalPolicyOptimization(policy=pi,
                                     parameters=pi.parameters,
                                     num_envs=config.ppo_num_envs,
                                     num_mini_batches=config.ppo_num_mini_batches,
                                     epochs=config.ppo_epochs,
                                     clip_range=config.ppo_clipping_parameter,
                                     value_function_coeff=config.ppo_value_function_coeff,
                                     entropy_coeff=config.ppo_entropy_coeff,
                                     discount=config.gae_discount,
                                     gae_weight=config.gae_weight,
                                     max_updates=episodes,
                                     use_linear_decay=config.ppo_use_linear_decay,
                                     logger_=logger,
                                     )
    print_message('Training begins:\n{} \nfor {} episodes'.format(config, episodes),
                  print_to_file,
                  config,
                  'w')
    for episode in range(episodes):
        logger.start_episode()
        rollout = r.run()
        ppo.train(rollout)
        logger.end_episode()
    env.close()
    logger.finalize()
    print_message('Training and evaluation completed.',
                  print_to_file,
                  config)


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == '--file':
        main(True)
    else:
        main()
