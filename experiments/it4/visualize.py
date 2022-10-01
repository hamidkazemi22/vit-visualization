import pdb
import pickle
import numpy as np

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from hooks.transformer.vit import ViTAttHookHolder
from model import model_library
from saver import ExperimentSaver
from utils import exp_starter_pack
from datasets import weird_image_net
from utils.device import to_cuda

"""
SHAPES: 
    in: [18, 577, 768]
    key: [18, 577, 768]
    query: [18, 577, 768]
    value: [18, 577, 768]
    out: [18, 577, 768]
    score: [18, 12, 577, 577]
"""


class FeatHeap:
    def __init__(self, limit: int = 24 * 24, feat_count: int = 10):
        self.arr = []
        self.limit = limit
        self.n = feat_count

    @torch.no_grad()
    def convert(self, act: torch.tensor, start: int, images: torch.tensor) -> list:
        images = images.detach().clone().cpu()
        act = act.detach().clone()
        act = act[:, :-1, :self.n].transpose(0, -1)  # 768, 576, 18
        feats, words, batch_size = act.shape
        sqrt_words = int(np.sqrt(words))

        index = torch.arange(start, start + batch_size).view(1, 1, -1).repeat(feats, words, 1).view(feats,
                                                                                                    -1).numpy().tolist()
        patch = torch.arange(words).view(1, -1, 1).repeat(feats, 1, batch_size).view(feats, -1)
        xs = (patch // sqrt_words).view(feats, -1).numpy().tolist()
        ys = (patch % sqrt_words).view(feats, -1).numpy().tolist()
        flat = act.reshape(feats, -1).cpu().numpy().tolist()
        arrayed = [
            sorted([(a, ii, x, y) for a, ii, x, y in zip(flat[i], index[i], xs[i], ys[i])], reverse=True)[:self.limit]
            for i in range(feats)]
        patch_size = images.shape[-1] // sqrt_words
        imaged = [[(a, i, x, y,
                    images[i - start, :, x * patch_size:(x + 1) * patch_size, y * patch_size:(y + 1) * patch_size]) for
                   (a, i, x, y) in arr] for arr in arrayed]
        return imaged

    @torch.no_grad()
    def __add__(self, tup: tuple):
        converted = self.convert(*tup)

        if len(self.arr) != len(converted):
            self.arr = [[] for _ in range(len(converted))]

        for i in range(len(converted)):
            self.arr[i] = sorted(self.arr[i] + converted[i], reverse=True)[:self.limit]

        """
        for i in range(len(converted)):
            self.arr[i].sort(reverse=True)
            self.arr[i] = self.arr[i][:self.limit]
        """
        return self


def main():
    exp_name, args, _ = exp_starter_pack()
    model, _, _, _ = model_library[34]()
    layer = args.layer
    loader = DataLoader(weird_image_net.eval(), batch_size=10, shuffle=False)
    hooks = ViTAttHookHolder(model, True, True, True, True, True, True, slice(layer, layer + 1))
    count = 0
    heaps = {}

    keys = ['in', 'key', 'query', 'value', 'score', 'out']
    feat_count = 8
    savers = {f'{f}_{k}': ExperimentSaver(f'DataX{k}X{layer}_{f}', save_id=True, disk_saver=True) for k in keys
              for f in range(feat_count)}

    for ii, data in enumerate(tqdm(loader)):
        xs, ys = to_cuda(data)
        with torch.no_grad():
            h, o = hooks(xs)
        for k in h.keys():
            if k not in heaps.keys():
                heaps[k] = FeatHeap(limit=64, feat_count=feat_count)
        torch.cuda.empty_cache()
        for k, v in h.items():
            heaps[k] += (v[0], count, xs)
            if ii % 100 == -1:
                for f in range(feat_count):
                    savers[f'{f}_{k}'].save([x[-1] for x in heaps[k].arr[f]], f'{ii}')

        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r - a  # free inside reserved
        print(f'f: {f},\tt: {t},\ta: {a},\tr: {r}')
        count += len(xs)


if __name__ == '__main__':
    main()
