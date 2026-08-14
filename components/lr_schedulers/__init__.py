from functools import partial

from omegaconf import DictConfig
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
    StepLR,
)

from components.lr_schedulers.linear_warmup_cosine_decay import (
    LinearWarmupWithCosineDecay,
)
from components.lr_schedulers.no_op import NoOpLRScheduler


def get_scheduler(scheduler_configs: DictConfig):
    schedulers = {
        "cosine_annealing": lambda: partial(
            CosineAnnealingLR,
            T_max=scheduler_configs.T_max,
            eta_min=scheduler_configs.eta_min,
        ),
        "consine_annealing_warm_restarts": lambda: partial(
            CosineAnnealingWarmRestarts,
            T_0=scheduler_configs.T_0,
            T_mult=scheduler_configs.T_mult,
            eta_min=scheduler_configs.eta_min,
        ),
        "steplr": lambda: partial(
            StepLR,
            step_size=scheduler_configs.step_size,
            gamma=scheduler_configs.gamma,
        ),
        "on_plateu": lambda: partial(
            ReduceLROnPlateau,
            mode=scheduler_configs.mode,
            patience=scheduler_configs.patience,
        ),
        "linear_warmup_with_cosine_decay": lambda: partial(
            LinearWarmupWithCosineDecay,
            warmup_iters=scheduler_configs.warmup_iters,
            lr_decay_iters=scheduler_configs.lr_decay_iters,
            min_lr_scale=scheduler_configs.min_lr_scale,
        ),
        "none": lambda: partial(NoOpLRScheduler),
    }

    try:
        return schedulers[scheduler_configs.name]()
    except KeyError:
        raise ValueError(
            f"Scheduler {scheduler_configs.name} not supported. Available schedulers: {list(schedulers.keys())}"
        )
