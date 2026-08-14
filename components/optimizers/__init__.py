from functools import partial

import torch
from omegaconf import DictConfig

from components.optimizers.adamw_gpt2 import gpt2_adamw


def get_optimizer(optim_configs: DictConfig) -> torch.optim.Optimizer:
    optimizers = {
        "sgd": lambda: partial(
            torch.optim.SGD,
            lr=optim_configs.lr,
            weight_decay=optim_configs.weight_decay,
            momentum=optim_configs.momentum,
        ),
        "adam": lambda: partial(
            torch.optim.Adam,
            lr=optim_configs.lr,
            weight_decay=optim_configs.weight_decay,
            fused=True,
        ),
        "adamw": lambda: partial(
            torch.optim.AdamW,
            lr=optim_configs.lr,
            weight_decay=optim_configs.weight_decay,
            fused=True,
        ),
        "rsmprop": lambda: partial(
            torch.optim.RMSprop,
            lr=optim_configs.lr,
            weight_decay=optim_configs.weight_decay,
            momentum=optim_configs.momentum,
            foreach=True,
        ),
        "gpt2_adamw": lambda: partial(
            gpt2_adamw,
            weight_decay=optim_configs.weight_decay,
            learning_rate=optim_configs.lr,
            betas=(optim_configs.beta1, optim_configs.beta2),
        ),
    }

    try:
        return optimizers[optim_configs.name]()
    except KeyError:
        raise ValueError(
            f"Invalid optimizer: {optim_configs.name}. Available optimizers: {optimizers.keys()}"
        )
