from typing import Union
from torch import Tensor
import torch.nn as nn


def to_device(x: Union[Tensor, nn.Module, list, tuple, dict], device: str = 'cuda:0') -> \
        Union[Tensor, list, nn.Module, tuple, dict]:
    if isinstance(x, Tensor):
        return x.to(device)
    if isinstance(x, list):
        return [to_device(i, device) for i in x]
    if isinstance(x, tuple):
        return tuple(to_device(i, device) for i in x)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    if isinstance(x, nn.Module):
        return x.to(device)
    return x


def to_cuda(x: Union[Tensor, list, tuple, dict, nn.Module]) -> \
        Union[Tensor, list, tuple, dict, nn.Module]:
    return to_device(x, 'cuda')
