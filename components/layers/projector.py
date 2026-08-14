import torch
from torch import nn


class ImageTextProjector(nn.Module):
    def __init__(self, embed_dim: int, expansion_factor: int, activation=nn.GELU()):
        super().__init__()

        self.expansion = nn.Linear(embed_dim, expansion_factor * embed_dim)
        self.act = activation
        self.converger = nn.Linear(expansion_factor * embed_dim, embed_dim)
        self.norm = nn.LayerNorm(
            embed_dim
        )  # LayerNorm applied before the cross-attention like its done for text

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x has shape (batch_size, max_sentence_length, d_model)
        x = self.expansion(x)
        x = self.act(x)
        x = self.converger(x)
        x = self.norm(x)
        return x
