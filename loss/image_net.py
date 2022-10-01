import pdb

import torch

from datasets import Normalizer
from hooks.transformer.vit import ViTAttHookHolder, ViTAbsHookHolder
from .base import InvLoss
from .regularizers import TotalVariation as BaseTotalVariation, FakeColorDistribution as AbstractColorDistribution
from .regularizers import FakeBatchNorm as BaseFakeBN, NormalVariation as BaseNormalVariation
from .regularizers import ColorVariation as BaseColorVariation
from hooks import TimedHookHolder, LayerHook
import numpy as np


class MatchBatchNorm(InvLoss):
    def __init__(self, bn: BaseFakeBN, coefficient: float = 1.):
        super().__init__(coefficient=coefficient)
        self.bn = bn

    def loss(self, x: torch.tensor) -> torch.tensor:
        return self.bn(x)


class TotalVariation(InvLoss):
    def loss(self, x: torch.tensor):
        return self.tv(x) * np.prod(x.shape[-2:]) / self.size

    def __init__(self, p: int = 2, size: int = 224, coefficient: float = 1.):
        super().__init__(coefficient)
        self.tv = BaseTotalVariation(p)
        self.size = size * size


class NormalVariation(InvLoss):
    def loss(self, x: torch.tensor):
        return self.tv(x) * np.prod(x.shape[-2:]) / self.size

    def __init__(self, p: int = 2, size: int = 224, coefficient: float = 1.):
        super().__init__(coefficient)
        self.tv = BaseNormalVariation(p)
        self.size = size * size


class ColorVariation(InvLoss):
    def loss(self, x: torch.tensor):
        return self.tv(x) * np.prod(x.shape[-2:]) / self.size

    def __init__(self, p: int = 2, size: int = 224, coefficient: float = 1.):
        super().__init__(coefficient)
        self.tv = BaseColorVariation(p)
        self.size = size * size


class CrossEntropyLoss(InvLoss):
    def loss(self, x: torch.tensor):
        return self.xent(self.model(x), self.label)

    def __init__(self, model: torch.nn.Module, label: torch.tensor, coefficient: float = 1.):
        super().__init__(coefficient)
        self.model = model
        self.label = label
        self.xent = torch.nn.CrossEntropyLoss()


class BatchAugment(InvLoss):
    def loss(self, x: torch.tensor):
        if self.aug is not None:
            x = self.aug(x)
        return self.other(x)

    def __init__(self, other: InvLoss, aug: torch.tensor = None):
        super().__init__(coefficient=1.0)
        self.other = other
        self.aug = aug


class NetworkPass(InvLoss):
    def __init__(self, model: torch.nn.Module):
        super().__init__(coefficient=0.0)
        self.model = model

    def loss(self, x: torch.tensor):
        self.model(x)
        return torch.tensor(0)


class BatchNorm1stLayer(InvLoss):
    def loss(self, x: torch.tensor) -> torch.tensor:
        return self.hook.get_layer(self.layer)

    def reset(self) -> torch.tensor:
        return self.hook.reset()

    def __init__(self, bn_hook: TimedHookHolder, layer: int = 0, coefficient: float = 1.):
        super().__init__(coefficient=coefficient)
        self.hook = bn_hook
        self.layer = layer


class ViTFeatHook(InvLoss):
    def __init__(self, hook: ViTAbsHookHolder, key: str, coefficient: float = 1.0):
        super().__init__(coefficient)
        self.hook = hook
        self.key = key

    def loss(self, x: torch.tensor):
        d, o = self.hook(x)
        all_feats = d[self.key][0][:, 1:, :].mean(dim=1)  # Exclude CLS
        mn = min(all_feats.shape)
        return - all_feats[:mn, :mn].diag().mean()


class ViTEnsFeatHook(ViTFeatHook):
    def __init__(self, hook: ViTAbsHookHolder, key: str, feat: int = 0, coefficient: float = 1.0):
        super().__init__(hook, key, coefficient)
        self.f = feat

    def loss(self, x: torch.tensor):
        d, o = self.hook(x)
        all_feats = d[self.key][0][:, 1:, :].mean(dim=1)  # Exclude CLS
        mn = min(all_feats.shape)
        return - all_feats[:mn, self.f].diag().mean()


class ReconstructionLoss(ViTFeatHook):
    def __init__(self, hook: ViTAbsHookHolder, x: torch.tensor, key: str, feat: int = 0,
                 coefficient: float = 1.0):
        super().__init__(hook, key, coefficient)
        self.ref = self.hook(x).clone().detach()
        self.f = feat

    def loss(self, x: torch.tensor):
        return (self.hook(x) - self.ref).norm()


class ViTHeadHook(ViTEnsFeatHook):
    def loss(self, x: torch.tensor):
        d, o = self.hook(x)
        all_feats = d[self.key][0][:, 1:, :].mean(dim=1)  # Exclude CLS and average over words, Result is BSx768
        return -all_feats.view(all_feats.shape[0], 12, -1).mean(dim=-1)[:, self.f].mean()


class ViTScoreHook(ViTEnsFeatHook):
    def loss(self, x: torch.tensor):
        d, o = self.hook(x)
        score_head = d[self.key][0][:, self.f, 1:, 1:]
        pw = int(np.sqrt(score_head.shape[-1]))
        patched = score_head.view(-1, pw, pw, pw, pw)
        ret_val = -patched[:, :, :pw // 2, :, pw // 2:].mean()
        return ret_val * 10000


class LayerActivationNorm(InvLoss):
    def __init__(self, hook: LayerHook, model: torch.nn.Module, coefficient: float = 1.0):
        super().__init__(coefficient)
        self.hook, self.model = hook, model

    def loss(self, x: torch.tensor) -> torch.tensor:
        self.model(x)
        return - self.hook()


class ActivationNorm(InvLoss):
    def loss(self, x: torch.tensor):
        return - self.hook.get_layer(self.layer)

    def __init__(self, activation_hook: TimedHookHolder, layer: int, coefficient: float = 1.):
        super().__init__(coefficient)
        self.hook = activation_hook
        self.layer = layer

    def reset(self) -> torch.tensor:
        return self.hook.reset()


class ColorDistribution(InvLoss):
    def loss(self, x: torch.tensor):
        return self.color_loss(x)

    def __init__(self, normalizer: Normalizer, coefficient: float = 1.):
        super().__init__(coefficient)
        self.color_loss = AbstractColorDistribution(normalizer)
