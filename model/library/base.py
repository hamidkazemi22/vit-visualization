import torch
from torch import nn

from datasets import Normalizer, image_net
from utils.iterators import ItemIterator


class InvertModel(ItemIterator):
    _device = 'cuda' if torch.cuda.is_available() else 'cpu'

    @property
    def iterator_item(self):
        return [self]

    def __init__(self, name: str, constructor, constructor_args: dict, image_size: int, batch_size: int,
                 normalizer: Normalizer):
        self.name = name
        self.constructor, self.constructor_args = constructor, constructor_args
        self.image_size, self.batch_size = image_size, batch_size
        self.normalizer = normalizer

    def __call__(self, normalize: bool = True, grad: bool = False) -> (torch.nn.Module, int, int, str):
        model = self.constructor(**self.constructor_args)
        if normalize:
            model = nn.Sequential(self.normalizer, model)
        model.eval()

        if not grad:
            for param in model.parameters():
                param.requires_grad = grad
        return model.to(self._device), self.image_size, self.batch_size, self.name


class TorchVisionModel(InvertModel):
    def __init__(self, constructor, batch_size: int):
        name = f'{self.__class__.__name__}_{constructor.__name__}'
        constructor_args = {'pretrained': True}
        image_size = 224
        normalizer = image_net.normalizer
        super().__init__(name, constructor, constructor_args, image_size, batch_size, normalizer)


class ModelLibrary(ItemIterator):
    @property
    def iterator_item(self):
        return self.models

    def __init__(self, other_models: list):
        self.models = [m for l in other_models for m in l]

    @property
    def names(self) -> dict:
        return {i: self[i].name for i in range(len(self))}

    def __str__(self):
        out = ''
        for i in range(len(self)):
            out +=f'{i}:\t{self[i].name}\n'
            if i % 10 == 9:
                out += '\n'
        return out

    @staticmethod
    def _fw_score(q: str, reference: str) -> int:
        score = 0
        for i in range(len(reference)):
            if reference[i] == q[score]:
                score += 1
                if score == len(q):
                    break
        return score

    @staticmethod
    def _amin_score(q: str, reference: str) -> float:
        return (ModelLibrary._fw_score(q, reference) + ModelLibrary._fw_score(q[::-1], reference[::-1])) / (
                len(q) + len(reference))

    def search(self, query: str, top_k=5):
        scores = [(self._amin_score(query, self[i].name), self[i].name, i) for i in range(len(self))]
        scores = sorted(scores, reverse=True)
        return [(i, name) for _, name, i in scores][:top_k]
