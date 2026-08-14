import math

from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer


class LinearWarmupWithCosineDecay(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_iters: int,
        lr_decay_iters: int,
        min_lr_scale: float,
        last_epoch: int = -1,
    ):
        self.warmup_iters = warmup_iters
        self.lr_decay_iters = lr_decay_iters
        self.min_lr_scale = min_lr_scale
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        current_iter = self.last_epoch
        if current_iter < self.warmup_iters:
            # Linear warmup
            return [
                base_lr * current_iter / self.warmup_iters for base_lr in self.base_lrs
            ]
        elif current_iter > self.lr_decay_iters:
            # Minimum learning rate after decay
            return [self.min_lr_scale * blr for blr in self.base_lrs]
        else:
            # Cosine decay
            decay_ratio = (current_iter - self.warmup_iters) / (
                self.lr_decay_iters - self.warmup_iters
            )
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return [
                self.min_lr_scale * base_lr
                + coeff * (base_lr - self.min_lr_scale * base_lr)
                for base_lr in self.base_lrs
            ]
