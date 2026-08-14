from torch.optim.lr_scheduler import _LRScheduler


class NoOpLRScheduler(_LRScheduler):
    def __init__(self, optimizer):
        super().__init__(optimizer)

    def get_lr(self):
        # Always return the current learning rate as-is
        return [group["lr"] for group in self.optimizer.param_groups]
