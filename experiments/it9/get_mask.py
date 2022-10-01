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
    method = 'eval'
    feat_count = 8
    patch_size = 16

    dataset = weird_image_net.eval() if method == 'eval' else weird_image_net.train()
    indices = torch.load(f'{method}.pt')
    indices = indices.view(12, -1)[:, :feat_count]
    model, image_size, _, _ = model_library[34]()
    hook = SaliencyViTGeLUHook(model)

    par_norm = os.path.join('desktop', f'{method}_mask_normalized')
    par = os.path.join('desktop', f'{method}_mask')
    os.makedirs(par, exist_ok=True)
    for l in tqdm(range(12)):
        for f in range(feat_count):
            img, _ = dataset[indices[l, f]]
            img = img.cuda().unsqueeze(0)
            act = hook(img, l, f).view(1, image_size // patch_size, image_size // patch_size)
            torchvision.utils.save_image(act, os.path.join(par_norm, f'{l}_{f}.png'), normalize=True)
            torchvision.utils.save_image(act, os.path.join(par, f'{l}_{f}.png'), normalize=False)


if __name__ == '__main__':
    main()
