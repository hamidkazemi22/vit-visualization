import os

import torchvision.utils
from torch.utils.data import DataLoader
from datasets import weird_image_net
from hooks.transformer.vit import SimpleViTGeLUHook
from model import model_library
from utils import exp_starter_pack
import torch
from tqdm import tqdm


@torch.no_grad()
def main():
    exp_name, args, _ = exp_starter_pack()
    method = 'eval'
    # method = 'train'
    feat_count = 8

    dataset = weird_image_net.eval() if method == 'eval' else weird_image_net.train()
    indices = torch.load(f'{method}.pt')
    indices = indices.view(12, -1)[:, :feat_count]

    par = os.path.join('desktop', method)
    os.makedirs(par, exist_ok=True)
    for l in tqdm(range(12)):
        for f in range(feat_count):
            img, _ = dataset[indices[l, f]]
            torchvision.utils.save_image(img, os.path.join(par, f'{l}_{f}.png'))


if __name__ == '__main__':
    main()
