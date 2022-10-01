import pdb
import pickle

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from hooks.transformer.vit import ViTAttHookHolder
from model import model_library
from utils import exp_starter_pack
from datasets import weird_image_net
from utils.device import to_cuda


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy().tolist()
    if isinstance(x, list):
        return [to_numpy(i) for i in x]
    if isinstance(x, tuple):
        return tuple(to_numpy(i) for i in x)
    if isinstance(x, dict):
        return {k: to_numpy(v) for k, v in x.items()}
    return x


""" Result: We cannot pass all layers / all data and save activations, we need to update them as we go! """


def main():
    exp_name, args, _ = exp_starter_pack()
    model, _, _, _ = model_library[34]()
    loader = DataLoader(weird_image_net.eval(), batch_size=18, shuffle=False)
    hooks = ViTAttHookHolder(model, True, True, True, True, True, True, slice(1))

    for data in tqdm(loader):
        xs, ys = to_cuda(data)
        with torch.no_grad():
            h, o = hooks(xs)
            print('Done forward, saving', flush=True)
            with open('test.pkl', 'wb') as file: pickle.dump(h, file)
            break


if __name__ == '__main__':
    main()
