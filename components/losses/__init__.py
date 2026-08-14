import torch
from omegaconf import DictConfig
from torch import nn

from components.losses.cce import CCE
from components.losses.weighted_ce import CrossEntropyLossWithWeights


def get_loss_fn(loss_configs: DictConfig) -> nn.Module:
    losses = {
        "simple_ce": lambda: nn.CrossEntropyLoss(),
        "ce": lambda: nn.CrossEntropyLoss(
            weight=torch.tensor(loss_configs.weight, dtype=torch.float32)
        ),
        "ce_rg": lambda: nn.CrossEntropyLoss(
            ignore_index=loss_configs.ignore_index,
            label_smoothing=loss_configs.label_smoothing,
        ),
        "ce_rg_with_weights": lambda: CrossEntropyLossWithWeights(
            class_weights=loss_configs.class_weights,
            ignore_index=loss_configs.ignore_index,
        ),
        "cce": lambda: CCE(),
        "bce": lambda: nn.BCELoss(),
        "bce_logits": lambda: nn.BCEWithLogitsLoss(),
    }

    try:
        return losses[loss_configs.name]()
    except KeyError:
        raise ValueError(
            f"Loss function {loss_configs.name} not found. Available options are: {list(losses.keys())}"
        )
