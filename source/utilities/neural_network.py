import torch
import torch.nn.init as init

from typing import Tuple, Union


def get_output_shape(input_size: Tuple[int, int],
                     kernel_size: Tuple[int, int],
                     stride: Tuple[int, int],
                     ) -> Tuple[int, int]:
    """
    :param input_size:
    :param kernel_size:
    :param stride:
    :return:
    """
    # TODO check that input really is int
    x = int((input_size[0] - kernel_size[0]) / stride[0] + 1)
    y = int((input_size[1] - kernel_size[1]) / stride[1] + 1)
    return x, y


def orthogonal(layer: torch.nn.Module,
               gain: Union[float, str] = None,
               ):
    """
    Performs an in-place orthogonal initialization on the given layer scaled with the given gain. The bias is set to 0.
    :param layer: the layer to initialize
    :param gain: the gain to use for scaling
    :return: None
    """
    if type(gain) is str:
        gain = init.calculate_gain(gain)
    elif gain is None:
        gain = 1
    init.orthogonal_(layer.weight.data, gain)
    init.constant_(layer.bias.data, val=0)
