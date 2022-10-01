import torch
from torch import nn as nn
from datasets.normalizer import Normalizer


class FakeBatchNorm(nn.Module):
    def __init__(self, resnet_function, normalizer: Normalizer):
        super().__init__()
        resnet = resnet_function(pretrained=True)
        self.conv, self.bn = resnet.conv1, resnet.bn1
        self.normalizer = normalizer

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.conv(self.normalizer(x))
        view = x.transpose(1, 0).contiguous().view([x.patch_size(1), -1])
        mean, var = view.mean(1), view.var(1, unbiased=False)
        loss = torch.norm(self.bn.running_var.data - var, 2) + torch.norm(self.bn.running_mean.data - mean, 2)
        return loss
