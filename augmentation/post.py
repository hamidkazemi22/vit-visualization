import torch
from torch import nn as nn


class ClipSTD(nn.Module):
    @torch.no_grad()
    def forward(self, x: torch.tensor, inflate: float = 1., per_sample: bool = True) -> torch.tensor:
        std = x.std() if not per_sample else x.view(x.shape[0], -1).std(dim=-1).view(-1, 1, 1, 1)
        mean = x.mean() if not per_sample else x.view(x.shape[0], -1).mean(dim=-1).view(-1, 1, 1, 1)
        x = inflate * (x - mean) / (std * 2)
        return x.clamp(min=-0.5, max=0.5) + 0.5


class Clip(nn.Module):
    @torch.no_grad()
    def forward(self, x: torch.tensor) -> torch.tensor:
        return x.clamp(min=0, max=1)


class LInfClip(nn.Module):
    def __init__(self, original: torch.tensor, eps: float = 16 / 255):
        super().__init__()
        self.base = original.detach().clone().cuda()
        self.eps = eps

    @torch.no_grad()
    def forward(self, x: torch.tensor) -> torch.tensor:
        return x + torch.clip(self.base - x, min=-self.eps, max=self.eps)


class L2Clip(nn.Module):
    def __init__(self, original: torch.tensor, eps: float = 16 / 255):
        super().__init__()
        self.base = original.detach().clone().cuda()
        self.eps = eps

    @torch.no_grad()
    def forward(self, x: torch.tensor) -> torch.tensor:
        delta = self.base - x
        norm = delta.norm(p=2)
        delta = self.eps * delta / norm if norm > self.eps else delta
        return x + delta
