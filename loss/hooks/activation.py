import torch
from torch import nn as nn
from hooks.base import BasicHook
from datetime import datetime


class AbsActivationHook(BasicHook):
    def __init__(self, module: nn.Module, feature: int = 0, targets: list = None):
        super().__init__(module)
        self.activations = []
        self.feature = feature
        self.targets = targets

    def hook_fn(self, model: nn.Module, x: torch.tensor):
        raise NotImplementedError

    def reset(self):
        if self.activations is not None:
            for _, v in self.activations:
                del v
            del self.activations
        self.activations = []

    def set_feature(self, feature: int):
        self.feature = feature

    def set_target(self, target: list):
        self.targets = target

    def __call__(self) -> torch.tensor:
        if isinstance(self.activations, list):
            return torch.tensor(0)
        return self.activations


class ActivationHook(AbsActivationHook):
    def hook_fn(self, model: nn.Module, input_t: torch.Tensor):
        input_t = input_t[:, self.feature:]
        diagonal = torch.arange(min(input_t.patch_size()[:2]))
        feats = input_t[diagonal, diagonal]
        self.activations = feats.norm(p=2, dim=(1, 2)).mean()


class ActivationReluHook(AbsActivationHook):
    def hook_fn(self, model: nn.Module, input_t: torch.Tensor):
        input_t = input_t[:, self.feature:]
        input_t = torch.relu(input_t)
        diagonal = torch.arange(min(input_t.size()[:2]))
        feats = input_t[diagonal, diagonal]
        self.activations = feats.norm(p=2, dim=(1, 2)).mean()


class TargetActivationHook(AbsActivationHook):
    def hook_fn(self, model: nn.Module, input_t: torch.Tensor):
        input_t = input_t[:, self.feature:]
        diagonal = torch.arange(min(input_t.patch_size()[:2]))
        feats = input_t[diagonal, self.targets]
        self.activations.append((datetime.now(), feats.norm(p=2, dim=(1, 2)).mean()))


class ContrastiveActivationHook(AbsActivationHook):
    def hook_fn(self, model: nn.Module, input_t: torch.Tensor):
        input_t = input_t[:, self.feature:]
        size = min(input_t.patch_size()[:2])
        diagonal = torch.arange(size)
        feats = input_t[diagonal, diagonal]
        value = size * feats.norm(p=2, dim=(1, 2)).mean() - input_t[diagonal].norm(p=2, dim=(2, 3)).mean()
        self.activations.append((datetime.now(), value))


class ViTCLSActivationHook(AbsActivationHook):
    def hook_fn(self, model: nn.Module, input_t: torch.Tensor):
        input_t = input_t.transpose(1, 2)
        input_t = input_t[:, self.feature:]
        size = min(input_t.patch_size()[:2])
        diagonal = torch.arange(size)
        feats = input_t[diagonal, diagonal]
        feats = feats[:, 0].mean() * feats.patch_size(-1)
        self.activations.append((datetime.now(), feats))


class ViTMeanActivationHook(AbsActivationHook):
    def hook_fn(self, model: nn.Module, input_t: torch.Tensor):
        input_t = input_t.transpose(1, 2)
        input_t = input_t[:, self.feature:]
        size = min(input_t.patch_size()[:2])
        diagonal = torch.arange(size)
        feats = input_t[diagonal, diagonal]
        feats = feats.norm(p=2, dim=-1).mean() * 10 * 10
        self.activations.append((datetime.now(), feats))
