import pdb
from torch import nn

import torch


class ActivationHelper:
    def __init__(self, how_many_patches: int):
        self.n_patches = how_many_patches

    @torch.no_grad()
    def __call__(self, score: torch.tensor) -> (int, int):
        score = score.squeeze(0)
        s1, s2 = self.n_patches, score.size(0) // self.n_patches
        score = score.view(s1, s2, s1, s2).transpose(1, 2).mean(dim=-1).mean(dim=-1)
        mx, col = score.max(dim=-1)
        val, row = mx.max(dim=-1)
        top = row.item()
        left = col[top].item()
        return top, left

    @torch.no_grad()
    def most_salient(self, x: torch.tensor, model, patch_size: int) -> torch.tensor:
        act = model(x)
        top, left = self(act)
        top, left = top * patch_size, left * patch_size
        return x[:, :, top:top + patch_size, left:left + patch_size]

    @torch.no_grad()
    def least_salient_pos(self, x: torch.tensor, model, patch_size: int) -> (int, int):
        act = model(x)
        top, left = self(-act)
        return top * patch_size, left * patch_size
