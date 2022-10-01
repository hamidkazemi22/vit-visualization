import pdb

import torch
from typing import Union

from cam.activation_helper import ActivationHelper
from .model import ViTPatchFeatWrapper


class BatchFeed:
    def __init__(self, model: ViTPatchFeatWrapper):
        self.model = model

    @staticmethod
    @torch.no_grad()
    def _batch_gen(x: torch.tensor, batch_size) -> torch.tensor:
        for i in range(0, x.size(0), batch_size):
            yield x[i:min(i + batch_size, x.size(0))]

    @torch.no_grad()
    def __call__(self, x: torch.tensor, batch_size: int = 32, reduction: str = 'mean') -> torch.tensor:
        if reduction == 'mean':
            return torch.cat([self.model(i).mean(dim=-1).mean(dim=-1) for i in self._batch_gen(x, batch_size)])
        return torch.cat([self.model(i) for i in self._batch_gen(x, batch_size)])


class Stitcher:
    _device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __init__(self, model: ViTPatchFeatWrapper, patch_size: int, image_size: int = 384):
        self.model, self.image_size, self.patch_size = model, image_size, patch_size
        self.activation_helper = ActivationHelper(image_size // patch_size)
        self._feed = BatchFeed(self.model)

    @torch.no_grad()
    def max_score(self, ims: Union[torch.tensor, list], score: float, orig_im: torch.tensor) -> (torch.tensor, float):
        ims = torch.cat(ims).to(self._device) if isinstance(ims, list) else ims
        scores = self._feed(ims)
        best_score, best_index = scores.max(dim=-1)
        best_score, best_index = best_score.item(), best_index.item()
        torch.cuda.empty_cache()
        return (ims[best_index].unsqueeze(0).detach().clone(), best_score) if best_score > score else (orig_im, score)

    @torch.no_grad()
    def __call__(self, original: torch.tensor, score: float, x: torch.tensor) -> (torch.tensor, float):
        patch = self.select_patch(x)
        inputs = self.select_images(original, patch)
        return self.max_score(inputs, score, original)

    @torch.no_grad()
    def select_images(self, original, patch) -> list:
        original = original.to(self._device)
        inputs = []
        for i in range(0, self.image_size - self.patch_size + 1, self.patch_size):
            for j in range(0, self.image_size - self.patch_size + 1, self.patch_size):
                temp = original.detach().clone()
                temp[:, :, i:i + self.patch_size, j:j + self.patch_size] = patch
                inputs.append(temp)
        return inputs

    @torch.no_grad()
    def select_patch(self, x) -> torch.tensor:
        x = x.to(self._device)
        x = x if x.dim() == 4 else x.unsqueeze(0)
        return self.activation_helper.most_salient(x, self.model, self.patch_size)


class SniperStitcher(Stitcher):
    @torch.no_grad()
    def select_images(self, original, patch) -> list:
        top, left = self.activation_helper.least_salient_pos(original, self.model, self.patch_size)
        cp = original.detach().clone()
        cp[:, :, top:top + self.patch_size, left:left + self.patch_size] = patch
        return [cp]
