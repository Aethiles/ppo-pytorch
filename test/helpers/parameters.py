import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple


class TestParameters(nn.Module):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                 ):
        super(TestParameters, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear1 = nn.Linear(self.input_size, self.output_size)
        self.linear2 = nn.Linear(self.input_size, 1)

        self.forward_ctr = 0
        self.update_ctr = 0
        self.device = device
        self.to(device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=2.5 * 1e-4)

    def forward(self,
                x: torch.Tensor,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.view(-1, self.input_size)
        y = F.relu(self.linear2(x))
        x = torch.sigmoid(self.linear1(x))
        self.forward_ctr += 1
        return x, y

    def gradient_update(self,
                        loss: torch.Tensor,
                        clip: bool = False,
                        ):
        self.optimizer.zero_grad()
        loss.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        self.optimizer.step()
        self.update_ctr += 1
