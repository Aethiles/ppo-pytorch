import argparse
import math
import sys
import time
import torch

from source.environment.atari.setup import setup_environment
from source.policy.parameters import AtariParameters
from source.policy.policy import Policy
from source.training.runner import Runner
from source.training.ppo import ProximalPolicyOptimization
from source.utilities.config.hyperparameters import HyperParameters
from source.utilities.logging.tensorboard_logger import TensorboardLogger
from source.utilities.print import print_message


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, default='Pong', choices=['BeamRider', 'Breakout', 'Pong'], 
                        help='Game to use in this demo')
    parser.add_argument('--level', type=str, default='trained', choices=['new', 'learning', 'trained'],
                        help='Skill level of the agent')
    parser.add_argument('--seed', type=int, default=int(time.time()), help='Seed determines behavior of the agent')
    parser.add_argument('--record-video', action='store_true', default=False, help='Set to true if video output shall be recorded')
    parser.add_argument('--sleep', type=float, default=1/30, help='Pause before another frame is rendered. Defaults to 30 fps.')
    args = parser.parse_args()
    args.game += 'NoFrameskip-v4'
    return args


def main():
    args = parse_args()
    print(f'Showing a {args.level} agent on {args.game} with seed {args.seed}.')
    if 'Pong' in args.game:
        print(f'The agent controls the green paddle.')
    config = HyperParameters('')
    torch.manual_seed(args.seed)
    env = setup_environment(args.game, args.seed, no_op_max=1, record_video=args.record_video)
    theta = AtariParameters(size=config.nn_size,
                            input_shape=config.nn_input_shape,
                            output_size=env.action_space.n,
                            learn_rate=config.nn_learn_rate,
                            max_gradient_norm=config.nn_max_gradient_norm,
                            adam_epsilon=config.nn_adam_eps,
                            logger_=None)
    path = f'./models/{args.game.replace("NoFrameskip-v4", "")}_{args.level}.pt'
    theta.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
           
    pi = Policy(parameters=theta)
    state = env.reset()
    env.render()

    done = False
    while not done:
        state = torch.from_numpy(state).unsqueeze(0).float()
        action, _,  _ = pi.get_action_and_value(state)
        state, reward, done, info = env.step(action)
        if 'ale.lives' in info and info['ale.lives'] != 0:
            done = False
        env.render()
        time.sleep(args.sleep)
    print(f'The agent scored {info["episode"].reward} points over {info["episode"].length} time steps.')
    env.close()


if __name__ == '__main__':
        main()

