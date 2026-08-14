import torch
import torch.nn.functional as F
from torch import nn


class CCE(nn.Module):
    # Reference: https://github.com/unique-chan/Complement-Cross-Entropy/blob/master/cce.py
    def __init__(self, balancing_factor: int = 1):
        super().__init__()
        self.nll_loss = nn.NLLLoss()
        self.balancing_factor = balancing_factor

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Note: yHat.shape[1] <=> number of classes
        batch_size = len(y)
        # cross entropy
        cross_entropy = self.nll_loss(F.log_softmax(y_hat, dim=1), y)
        # complement entropy
        y_hat = F.softmax(y_hat, dim=1)
        yg = y_hat.gather(dim=1, index=torch.unsqueeze(y, 1))
        px = y_hat / (1 - yg) + 1e-7
        px_log = torch.log(px + 1e-10)
        y_zerohot = torch.ones(batch_size, y_hat.shape[1]).scatter_(
            1, y.view(batch_size, 1).data.cpu(), 0
        )
        output = px * px_log * y_zerohot
        complement_entropy = torch.sum(output) / (
            float(batch_size) * float(y_hat.shape[1])
        )

        return cross_entropy - self.balancing_factor * complement_entropy
