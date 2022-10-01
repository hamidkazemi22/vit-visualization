import torch
import torch.nn as nn


class Normalizer(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean', torch.Tensor(mean).reshape((1, -1, 1, 1)))
        self.register_buffer('std', torch.Tensor(std).reshape((1, -1, 1, 1)))

    def forward(self, t: torch.tensor) -> torch.tensor:
        return self.get_normal(t)

    def get_normal(self, t: torch.Tensor) -> torch.Tensor:
        return (t - self.mean) / self.std

    def get_unit(self, t: torch.Tensor) -> torch.Tensor:
        return (t * self.std) + self.mean
