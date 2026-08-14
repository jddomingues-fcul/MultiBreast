import torch
from torch import nn

 
class CrossEntropyLossWithWeights(nn.Module):
    def __init__(self, class_weights: dict, ignore_index: int, block_size: int = 512):
        super().__init__()
        self.class_weights = class_weights
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.block_size = block_size

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        birads_classes: list[str],
        modalities: list[str],
    ) -> torch.Tensor:
        cum_loss = torch.tensor(0.0, device=logits.device)

        n_samples = logits.shape[0] // self.block_size
        for i in range(n_samples):
            curr_preds = logits[i * self.block_size : (i + 1) * self.block_size, :]
            curr_target = target[i * self.block_size : (i + 1) * self.block_size]

            class_key = "-".join([birads_classes[i], modalities[i]])
            class_weight = torch.tensor(
                self.class_weights[class_key], device=logits.device
            )
            cum_loss += class_weight * self.ce_loss(curr_preds, curr_target)

        cum_loss /= n_samples
        return cum_loss
