import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Tuple
from transformers import CLIPVisionConfig


class CLIPAttention(nn.Module):
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

        # model weights
        self.k_proj = torch.rand(self.embed_dim, self.embed_dim)
        self.v_proj = torch.rand(self.embed_dim, self.embed_dim)
        self.q_proj = torch.rand(self.embed_dim, self.embed_dim)
        self.o_proj = torch.rand(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def weight_init(self):
        nn.init.xavier_normal_(self.q_proj)
        nn.init.xavier_normal_(self.k_proj)
        nn.init.xavier_normal_(self.v_proj)
        nn.init.xavier_normal_(self.o_proj)

    # Adapted from CLIPAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert (
            output_attentions is False
        ), "output_attentions is not supported in CLIPAttention"

        # CLIP text model uses both `causal_attention_mask` and `attention_mask`
        if attention_mask is not None and causal_attention_mask is not None:
            attn_mask = attention_mask + causal_attention_mask
        elif causal_attention_mask is not None:
            attn_mask = causal_attention_mask
        else:
            attn_mask = attention_mask

        bsz, tgt_len, embed_dim = hidden_states.size()

        query_states = F.linear(hidden_states, self.q_proj)
        key_states = F.linear(hidden_states, self.k_proj)
        value_states = F.linear(hidden_states, self.v_proj)

        query_states = query_states.view(
            bsz, -1, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        value_states = value_states.view(
            bsz, -1, self.num_heads, self.head_dim
        ).transpose(1, 2)

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
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = F.linear(attn_output, self.o_proj)

        return attn_output, None
