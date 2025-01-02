import os
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import CLIPVisionConfig


class QuickGELUActivation(nn.Module):
    """
    Applies GELU approximation that is fast but somewhat inaccurate. See: https://github.com/hendrycks/GELUs
    """

    def forward(self, input: Tensor) -> Tensor:
        return input * torch.sigmoid(1.702 * input)


class CLIPMLP(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.config = config
        assert (
            config.hidden_act == "quick_gelu"
        ), "Only quick_gelu activation is supported"

        self.activation_fn = QuickGELUActivation()

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # model weights
        self.fc1 = torch.rand(self.intermediate_size, self.hidden_size)
        self.fc2 = torch.rand(self.hidden_size, self.intermediate_size)

    def weight_init(self):
        nn.init.xavier_normal_(self.fc1)
        nn.init.xavier_normal_(self.fc2)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = F.linear(
            self.activation_fn(F.linear(hidden_states, self.fc1)),
            self.fc2,
        )
        return hidden_states
