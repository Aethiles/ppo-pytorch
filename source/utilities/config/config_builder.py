import configparser
import socket

from source.policy.parameters import AtariSize
from source.utilities.config.hyperparameters import HyperParameters
from typing import Dict


def get_hyperparameters() -> HyperParameters:
    """
    Returns HyperParameters that were updated using parameters from the config value.
    :return: the new hyper parameters
    """
    config = open_config()
    hyperparameters = _init_hyperparameters(config.pop('env_name'))
    _update_parameters(hyperparameters, config)
    return hyperparameters


def _init_hyperparameters(env_name: str,
                          ) -> HyperParameters:
    """
    Initializes a new HyperParameters instance. The env_name will be adjusted to include NoFrameskip if it does not do
    so already.
    :param env_name: the game to be run
    :return: the hyper parameters
    """
    if 'NoFrameskip' not in env_name:
        env_name = env_name.replace('-v', 'NoFrameskip-v')
    return HyperParameters(env_name)


def _update_parameters(hyperparameters: HyperParameters, config: Dict):
    """
    Updates the given HyperParameters updating default values with the values found in the given config.
    :param config: the config
    :param hyperparameters: the hyper parameters
    :return:
    """
    if 'nn_size' in config:
        hyperparameters.nn_size = AtariSize[config.pop('nn_size')]
    for key in config.keys():
        type_ = hyperparameters.__annotations__[key]
        value = type_(config[key])
        hyperparameters.__dict__[key] = value


def get_group() -> int:
    """
    Turns a COSYxx host name into a group. Groups consist of 3 COSY machines:
    COSY01-COSY03 are part of group 0, COSY04-COSY06 are part of group 1 etc.
    :return: the group number
    """
    hostname = socket.gethostname()
    if 'cosy' in hostname:
        hostname = hostname.replace('cosy', '')
        group = int((int(hostname) - 1) / 3)
    else:
        group = 0
    return group


def open_config() -> Dict:
    """
    Opens the config file and creates a config dictionary or section proxy from the config and the machine's group id.
    :return:
    """
    group = get_group()
    c = configparser.ConfigParser()
    c.read('config')
    config = c['Shared']
    config.update(c[str(group)])
    return config
