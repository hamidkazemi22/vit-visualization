import torch
from torch import nn


class LayerHook:
    def __init__(self, classifier: nn.Module, layer_class, layer_depth: int, hook_cls):
        self.layer = [m for m in classifier.modules() if isinstance(m, layer_class)][layer_depth]
        self.hook = hook_cls(self.layer)

    def __call__(self) -> torch.tensor:
        return self.hook()
