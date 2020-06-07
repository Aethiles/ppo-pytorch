import torch


def normalize_tensor(tensor: torch.Tensor,
                     epsilon: float = 1e-5,
                     ) -> torch.Tensor:
    """

    :param tensor:
    :param epsilon:
    :return:
    """
    return (tensor - tensor.mean()) / (tensor.std() + epsilon)
