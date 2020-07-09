import gym
import time

from gym.wrappers import Monitor
from source.environment.atari import action_repetition, evaluation, state_stack, reward_clipping, noop_reset, \
    episodic_life, fire_reset, image_transformation


def setup_environment(name: str,
                      seed: int = int(time.time()),
                      no_op_max: int = 30,
                      record_video: bool = False,
                      ) -> gym.Env:
    """
    Wrapper to setup environment. Seed defaults to the current unix timestamp in seconds.
    :param name: name of the gym environment
    :param seed: seed used by the gym environment, defaults to int(time.time())
    :param no_op_max: Maximum number of no operations to run when resetting the environment. Must be at least 1.
    :return: the gym environment
    """
    env = gym.make(name)
    env.seed(seed)
    if record_video:
        env = Monitor(env, './video')
    assert 'NoFrameskip' in env.spec.id
    env = noop_reset.NoopResetEnv(env, no_op_max=no_op_max)
    env = action_repetition.ActionRepetitionEnv(env)
    env = evaluation.EvaluationEnv(env)
    env = episodic_life.EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = fire_reset.FireResetEnv(env)
    env = image_transformation.ImageTransformationEnv(env)
    env = reward_clipping.ClipRewardEnv(env)
    env = state_stack.StateStackEnv(env)
    return env
