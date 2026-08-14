import copy

import torch
from torch import nn

from components.layers.attention import MultiHeadAttentionBlock


def clones(module: nn.Module, n: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class PositionwiseFeedForward(nn.Module):
    def __init__(
        self, embed_dim: int, d_ff: int, bias: bool = False, dropout: float = 0.1
    ):
        super().__init__()
        self.c_fc = nn.Linear(embed_dim, d_ff, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(d_ff, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Following similar approach to gpt2 implementation with gelu
        out = self.c_fc(x)
        out = self.gelu(out)
        out = self.c_proj(out)
        out = self.dropout(out)
        return out


def causal_mask(tokens: torch.Tensor, device) -> torch.Tensor:
    return torch.ones(size=tokens.shape, dtype=torch.bool, device=device).tril(
        diagonal=0
    )


def causal_mask_without_pads(
    tokens: torch.Tensor, pad_token_index: int
) -> torch.Tensor:
    size = tokens.size(0)
    mask = torch.ones(size, size, dtype=torch.bool, device=tokens.device).tril(
        diagonal=0
    )
    mask = mask & (tokens != pad_token_index).unsqueeze(0)
    return mask


def construct_modality_influence_mask(
    tokens: torch.Tensor, pad_token_index: int, end_of_modality_token: int
) -> torch.Tensor:
    size = tokens.size(0)
    mask = torch.ones(size, dtype=torch.bool, device=tokens.device)

    # we go through the tokens until we find the end_of_modality_token, setting all previous to False
    for i in range(size):
        if tokens[i] == end_of_modality_token:
            mask[i] = False
            break

        mask[i] = False

    # where its pad we also set to False
    mask = mask & (tokens != pad_token_index)

    return mask


class DecoderBlock(nn.Module):
    # Decoder is made of self-attn, src-attn, and feed forward (defined below)

    def __init__(
        self,
        embed_dim: int,
        self_attn: MultiHeadAttentionBlock,
        cross_attention: MultiHeadAttentionBlock,
        feed_forward: PositionwiseFeedForward,
    ):
        super().__init__()

        self.embed_dim = embed_dim

        self.attn = self_attn
        self.ln_1 = nn.LayerNorm(embed_dim)

        self.cross_attention = cross_attention
        self.ln_ca = nn.LayerNorm(embed_dim)

        self.mlp = feed_forward
        self.ln_2 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        decoder_input_mask: torch.Tensor,
        cross_attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(x=self.ln_1(x), decoder_input_mask=decoder_input_mask)
        x = x + self.cross_attention(
            x=self.ln_ca(x), cross_attn_memory=memory, cross_attn_mask=cross_attn_mask
        )
        x = x + self.mlp(x=self.ln_2(x))
        return x
