import os
import pdb

import torchvision.utils
from torch.utils.data import DataLoader
from datasets import weird_image_net, image_net
from hooks.transformer.vit import SimpleViTGeLUHook, SaliencyViTGeLUHook, SaliencyClipGeLUHook
from model import model_library
from utils import exp_starter_pack
import torch
from tqdm import tqdm


@torch.no_grad()
def main():
    exp_name, args, _ = exp_starter_pack()
    feat_count = 128
    patch_size = 32
    network = 98
    layer, feature = args.layer, args.feature
    method = 'eval' if args.grid == 1.0 else 'train'
    dataset = image_net.eval() if method == 'eval' else image_net.train()
    indices = torch.load(f'{method}_{layer}_{feature}.pt')
    model, image_size, _, _ = model_library[network]()
    hook = SaliencyClipGeLUHook(model)

    par_norm = os.path.join('desktop', f'{method}_{layer}_{feature}_mask')
    par_nat = os.path.join('desktop', f'{method}_{layer}_{feature}')
    os.makedirs(par_nat, exist_ok=True)
    os.makedirs(par_norm, exist_ok=True)
    for v, i in tqdm(indices):
        img, _ = dataset[i]
        img = img.cuda().unsqueeze(0)
        act = hook(img, layer, feature).vpiew(1, image_size // patch_size, image_size // patch_size)
        torchvision.utils.save_image(img, os.path.join(par_nat, f'{i}.png'))
        torchvision.utils.save_image(act, os.path.join(par_norm, f'{i}.png'), normalize=True)


if __name__ == '__main__':
    main()
