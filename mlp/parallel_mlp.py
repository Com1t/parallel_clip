import torch
from torch import nn, Tensor
from torch.nn import functional as F
import torch.distributed as dist
from transformers import CLIPVisionConfig

from model_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)


class QuickGELUActivation(nn.Module):
    """
    Applies GELU approximation that is fast but somewhat inaccurate. See: https://github.com/hendrycks/GELUs
    """

    def forward(self, input: Tensor) -> Tensor:
        return input * torch.sigmoid(1.702 * input)


class ParallelCLIPMLP(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super(ParallelCLIPMLP, self).__init__()
        self.config = config
        assert (
            config.hidden_act == "quick_gelu"
        ), "Only quick_gelu activation is supported"

        self.activation_fn = QuickGELUActivation()

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # ColumnParallelLinear splits the output dimension across GPUs
        self.fc1 = ColumnParallelLinear(
            in_features=self.hidden_size,
            out_features=self.intermediate_size,
            gather_output=False,  # Ensures the final output is gathered across all GPUs
        )

        # RowParallelLinear splits the input dimension across GPUs
        self.fc2 = RowParallelLinear(
            in_features=self.intermediate_size,
            out_features=self.hidden_size,
            input_is_parallel=True,
        )

    def init_layer_weight(self, target_layer, raw_weight):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        weight_partition_dim = [0, 0]
        if isinstance(target_layer, ColumnParallelLinear):
            weight_partition_dim[0] = raw_weight.shape[0] // world_size
            weight_partition_dim[1] = raw_weight.shape[1]
            split_dim = 0
        elif isinstance(target_layer, RowParallelLinear):
            weight_partition_dim[0] = raw_weight.shape[0]
            weight_partition_dim[1] = raw_weight.shape[1] // world_size
            split_dim = 1
        else:
            raise TypeError("ColumnParallelLinear or RowParallelLinear are allowed")

        output_tensor = torch.zeros(
            weight_partition_dim[0], weight_partition_dim[1]
        ).to(target_layer.weight.device)

        if rank == 0:
            weight_list = torch.split(
                raw_weight, weight_partition_dim[split_dim], dim=split_dim
            )
            scatter_list = [t.contiguous() for t in weight_list]
        else:
            scatter_list = None
        dist.scatter(output_tensor, scatter_list, src=0)
        target_layer.weight = nn.Parameter(output_tensor)

    def weight_init(self, fc1, fc2):
        self.init_layer_weight(self.fc1, fc1)
        self.init_layer_weight(self.fc2, fc2)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc2(self.activation_fn(self.fc1(hidden_states)))
        return hidden_states
