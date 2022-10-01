import torch.nn as nn
from utils.iterators import ItemIterator


class HookHolder(ItemIterator):
    def __init__(self, classifier: nn.Module, hook_class, layer_class):
        self.hooks = [hook_class(m) for m in classifier.modules() if isinstance(m, layer_class)]

    @property
    def iterator_item(self):
        return self.hooks

    def check_for_attr(self, attr: str, hook_class):
        for h in self:
            if not hasattr(h, attr):
                raise AttributeError('Class {} does not have attribute {}'.format(hook_class.__name__, attr))

    def _broadcast(self, func_name: str, *to_propagate):
        for i in self:
            func = getattr(i, func_name)
            func(*to_propagate)

    def _gather(self, attr: str) -> list:
        return [getattr(l, attr) for l in self]

    def close(self):
        self._broadcast('close')
