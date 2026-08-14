import numpy as np
import torch


def from_numpy(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x)


def div_255(x: torch.Tensor) -> torch.Tensor:
    return x / 255.0


def unsqueeze(x: torch.Tensor) -> torch.Tensor:
    return x.unsqueeze(1)


def repeat_rgb_channels(x: torch.Tensor) -> torch.Tensor:
    return x.repeat(1, 3, 1, 1)


def norm(x):
    mi = x.min()
    ma = x.max()
    return (x - mi) / (ma - mi)
