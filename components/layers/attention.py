import math

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel


# Sourced: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_training: bool = True,
):
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1))

    attn_bias_shape = (attn_mask.size(0), L, S) if attn_mask is not None else (L, S)
    attn_bias = torch.zeros(
        attn_bias_shape, dtype=query.dtype, device=query.device
    )  # Add the batch dimension to the sentence length

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            # unsqueeze to consider the number of heads
            attn_bias = attn_bias.unsqueeze(1)
        else:
            attn_bias = attn_mask + attn_bias

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=is_training)
    return attn_weight @ value, attn_weight


class MultiHeadAttentionBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        dropout: float,
        cross_attn: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.attention_scores = None
        self.dropout = dropout
        self.cross_attn = cross_attn
        self.resid_dropout = nn.Dropout(dropout)

        assert embed_dim % self.n_heads == 0, (
            "Dimension of embedding should be divisible by the number of heads"
        )
        self.d_k = embed_dim // self.n_heads

        if self.cross_attn:
            self.c_attn = nn.Linear(embed_dim, 2 * embed_dim, bias=bias)
            self.q_attn = nn.Linear(embed_dim, embed_dim, bias=bias)
        else:
            self.c_attn = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)

        self.c_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        cross_attn_memory: torch.Tensor | None = None,
        decoder_input_mask: torch.Tensor | None = None,
        cross_attn_mask: torch.Tensor | None = None,
    ):

        batch_size, seq_len, emb_size = x.size()

        if self.cross_attn:
            assert cross_attn_memory is not None, (
                "Cross attention memory must be provided for cross-attention"
            )
            n_patches = cross_attn_memory.size(1)
            patches_embedding = cross_attn_memory.size(2)

            q = self.q_attn(x)
            k, v = self.c_attn(cross_attn_memory).split(patches_embedding, dim=2)

            q = q.view(
                batch_size, seq_len, self.n_heads, emb_size // self.n_heads
            ).transpose(1, 2)  # (B, nh, T, hs)
            k = k.view(
                batch_size, n_patches, self.n_heads, patches_embedding // self.n_heads
            ).transpose(1, 2)  # (B, nh, T, hs)
            v = v.view(
                batch_size, n_patches, self.n_heads, patches_embedding // self.n_heads
            ).transpose(1, 2)  # (B, nh, T, hs)

            with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                x, self.attention_scores = scaled_dot_product_attention(
                    query=q,
                    key=k,
                    value=v,
                    attn_mask=cross_attn_mask,
                    dropout_p=self.dropout,
                    is_training=self.training,
                )
        else:
            # q,k,v
            q, k, v = self.c_attn(x).split(emb_size, dim=2)  # (B, T, 3 * C)

            q = q.view(
                batch_size, seq_len, self.n_heads, emb_size // self.n_heads
            ).transpose(1, 2)  # (B, nh, T, hs)
            k = k.view(
                batch_size, seq_len, self.n_heads, emb_size // self.n_heads
            ).transpose(1, 2)  # (B, nh, T, hs)
            v = v.view(
                batch_size, seq_len, self.n_heads, emb_size // self.n_heads
            ).transpose(1, 2)  # (B, nh, T, hs)

            with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                x, self.attention_scores = scaled_dot_product_attention(
                    query=q,
                    key=k,
                    value=v,
                    attn_mask=decoder_input_mask,
                    dropout_p=self.dropout,
                    is_training=self.training,
                )

        # Convert x to correct shape
        # The contiguous call colocates all the elements of the tensor in contiguous way in memory so we can then call
        # view and pytorch transform the tensor in place
        # -1 lets pytorch automatically calculate the size but will be the max_sentence_length
        # The self.d_model could be obtained by multiplying self.h and self.d_k
        out = x.transpose(1, 2).contiguous().view(batch_size, seq_len, emb_size)
        out = self.resid_dropout(self.c_proj(out))

        del q
        del k
        del v

        # Return shape (batch, max_sentence_length, embedding_dim)
        return out
