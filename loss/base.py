import torch

_nums = '0123456789'


def _abbreviation(name: str) -> str:
    if len(name) <= 3:
        return name
    abr = ''.join(x for x in name if x.isupper() or x in _nums)
    return abr[:3]


def _round(num: float) -> str:
    if num > 100:
        return str(int(round(num, 0)))
    if num > 10:
        return str(round(num, 1))
    return str(round(num, 2))


class InvLoss:
    def __init__(self, coefficient: float = 1.0):
        self.c = coefficient
        self.name = _abbreviation(self.__class__.__name__)
        self.last_value = 0

    def __call__(self, x: torch.tensor) -> torch.tensor:
        tensor = self.loss(x)
        self.last_value = tensor.item()
        return self.c * tensor

    def loss(self, x: torch.tensor):
        raise NotImplementedError

    def __str__(self):
        return f'{_round(self.c * self.last_value)}({_round(self.last_value)})'

    def reset(self) -> torch.tensor:
        return 0


class LossArray:
    def __init__(self):
        self.losses = []
        self.last_value = 0

    def __add__(self, other: InvLoss):
        self.losses.append(other)
        return self

    def __call__(self, x: torch.tensor):
        tensor = sum(l(x) for l in self.losses)
        self.last_value = tensor.item()
        return tensor

    def header(self) -> str:
        rest = '\t'.join(l.name for l in self.losses)
        return f'Loss\t{rest}'

    def __str__(self):
        rest = '\t'.join(str(l) for l in self.losses)
        return f'{_round(self.last_value)}\t{rest}'

    def reset(self):
        return sum(l.reset() for l in self.losses)
