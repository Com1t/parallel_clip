import torch
from torch import nn
import torch.distributed as dist
from typing import Optional, Tuple
from transformers import CLIPVisionConfig


from model_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)


class ParallelCLIPAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        # ColumnParallelLinear splits the output dimension across GPUs
        self.q_proj = ColumnParallelLinear(
            in_features=self.embed_dim,
            out_features=self.num_heads * self.head_dim,
            gather_output=False,  # Ensures the final output is splited across all GPUs
        )
        self.k_proj = ColumnParallelLinear(
            in_features=self.embed_dim,
            out_features=self.num_heads * self.head_dim,
            gather_output=False,  # Ensures the final output is splited across all GPUs
        )
        self.v_proj = ColumnParallelLinear(
            in_features=self.embed_dim,
            out_features=self.num_heads * self.head_dim,
            gather_output=False,  # Ensures the final output is splited across all GPUs
        )

        self.o_proj = RowParallelLinear(
            in_features=self.num_heads * self.head_dim,
            out_features=self.embed_dim,
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

    def weight_init(self, q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight):
        self.init_layer_weight(self.q_proj, q_proj_weight)
        self.init_layer_weight(self.k_proj, k_proj_weight)
        self.init_layer_weight(self.v_proj, v_proj_weight)
        self.init_layer_weight(self.o_proj, o_proj_weight)

    # Adapted from ParallelCLIPAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert (
            output_attentions is False
        ), "output_attentions is not supported in ParallelCLIPAttention"

        # CLIP text model uses both `causal_attention_mask` and `attention_mask`
        if attention_mask is not None and causal_attention_mask is not None:
            attn_mask = attention_mask + causal_attention_mask
        elif causal_attention_mask is not None:
            attn_mask = causal_attention_mask
        else:
            attn_mask = attention_mask

        bsz, tgt_len, embed_dim = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, tgt_len, -1, self.head_dim).transpose(
            1, 2
        )
        key_states = key_states.view(bsz, tgt_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, tgt_len, -1, self.head_dim).transpose(
            1, 2
        )

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attn_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # CLIP text model uses both `causal_attention_mask` and `attention_mask` sequentially.
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            scale=self.scale,
        )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output, None
