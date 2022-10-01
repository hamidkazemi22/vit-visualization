import os
import pdb

import torchvision.utils
from torch.utils.data import DataLoader
from datasets import weird_image_net
from hooks.transformer.vit import SimpleViTGeLUHook, SaliencyViTGeLUHook
from model import model_library
from utils import exp_starter_pack
import torch
from tqdm import tqdm


@torch.no_grad()
def main():
    exp_name, args, _ = exp_starter_pack()
    method = 'eval' if args.grid == 1 else 'train'
    feat_count = 256
    network = 35
    patch_size = 32

    dataset = weird_image_net.eval() if method == 'eval' else weird_image_net.train()
    indices = torch.load(f'{method}_{network}.pt')
    indices = indices.view(12, -1)[:, :feat_count]
    model, image_size, _, _ = model_library[network]()
    hook = SaliencyViTGeLUHook(model)

    par_norm = os.path.join('desktop', f'{method}_{network}_mask')
    par_nat = os.path.join('desktop', f'{method}_{network}')
    os.makedirs(par_norm, exist_ok=True)
    os.makedirs(par_nat, exist_ok=True)
    for l in tqdm(range(12)):
        for f in range(feat_count):
            img, _ = dataset[indices[l, f]]
            img = img.cuda().unsqueeze(0)
            act = hook(img, l, f).view(1, image_size // patch_size, image_size // patch_size)
            torchvision.utils.save_image(img, os.path.join(par_nat, f'{l}_{f}.png'))
            torchvision.utils.save_image(act, os.path.join(par_norm, f'{l}_{f}.png'), normalize=True)


if __name__ == '__main__':
    main()
