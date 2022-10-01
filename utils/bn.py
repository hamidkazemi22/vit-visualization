from typing import List

import numpy as np
import torch


class Accumulator:
    def __init__(self):
        self._values = []

    def __add__(self, other):
        other = other.detach().cpu().numpy() if isinstance(other, torch.Tensor) else other
        self._values.append(other)
        return self

    @property
    def value(self) -> np.ndarray:
        self._values = np.concatenate(self._values) if isinstance(self._values, List) else self._values
        return self._values

    def save(self, name: str):
        np.save('{}.npy'.format(name), self.value)


def get_len(model: torch.nn.Module):
    lens = [bn.num_features for bn in model.modules() if isinstance(bn, torch.nn.BatchNorm2d)]
    return sum(lens)


def get_mean(model: torch.nn.Module) -> np.ndarray:
    return np.array([m.running_mean.cpu().numpy() for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)])


def get_var(model: torch.nn.Module) -> np.ndarray:
    return np.array([m.running_var.cpu().numpy() for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)])


def get_stat_params(model: torch.nn.Module, get_all: bool = False) -> List[torch.Tensor]:
    mean = [m.running_mean for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    # var = [m.running_var for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    # mean.extend(var)
    if get_all:
        w = [m.weight for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
        b = [m.bias for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
        mean.extend(w)
        mean.extend(b)
    return mean


def get_error(first: np.ndarray, other: np.ndarray) -> float:
    first = np.concatenate(first, axis=0)
    other = np.concatenate(other, axis=0)
    return np.linalg.norm(first - other) / np.linalg.norm(other)


def replace_bn(model: torch.nn.Module, mean: np.ndarray, var: np.ndarray, last: int):
    index = 0
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean = torch.Tensor(mean[index]).cuda()
            m.running_var = torch.Tensor(var[index]).cuda()
            if index == last:
                break
            index += 1


def replace_bn_with_multiplicative_inverse(model: torch.nn.Module):
    for i, m in enumerate(model.modules()):
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_var = torch.Tensor(1. / m.running_var.detach().cpu().numpy()).cuda()
            m.running_var.requires_grad_()


def random_model(model: torch.nn.Module):
    for i, m in enumerate(model.modules()):
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean = (torch.Tensor(m.running_mean.detach().cpu()) + torch.randn_like(
                m.running_mean.cpu()) * 1.0).cuda()
            m.running_mean.requires_grad_()


def shirt_on_model(model: torch.nn.Module, mean, var):
    count = 0
    for i, m in enumerate(model.modules()):
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean = torch.Tensor(mean[count].detach().cpu().numpy()).cuda()
            m.running_mean.requires_grad_()
            m.running_var = torch.Tensor(var[count].detach().cpu().numpy()).cuda()
            m.running_var.requires_grad_()
            count += 1


def get_var_params(model: torch.nn.Module) -> torch.Tensor:
    # return sum([torch.clamp(running_var) for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)])
    return sum([torch.sum(1 / m.running_var) for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)])
