import torchvision.utils

from datasets import weird_image_net
from hooks.transformer.vit import SimpleViTGeLUHook
from model import model_library
from utils import exp_starter_pack
import torch
from tqdm import tqdm
import torchvision.transforms as tr
from torch import nn
import pdb
from matplotlib import pyplot as plt
import numpy as np


@torch.no_grad()
def main():
    exp_name, args, _ = exp_starter_pack()
    before = torch.load('shift_before.pt', map_location='cpu')
    after = torch.load('shift_after.pt', map_location='cpu')

    plt.figure()
    nn_f = before.mean(dim=-1)
    nn_s = after.mean(dim=-1)
    plt.bar(np.arange(12), nn_f, label='Original')
    plt.bar(np.arange(12), nn_s, label='Avg 16 XY-Translations')
    plt.xlabel('Layer')
    plt.ylabel('Avg Activation')
    plt.title('Effect of XY-Translation on Activation')
    plt.legend()
    plt.savefig('shift.pdf')
    plt.show()

    plt.cla()
    n_f = (before.view(-1) / before.view(-1)).view(before.shape)
    n_f = [a[(1 - a.isnan().int()).bool()].mean() for a in n_f]
    n_s = (after.view(-1) / before.view(-1)).view(before.shape)
    n_s = [a[(1 - a.isnan().int()).bool()].mean() for a in n_s]
    plt.bar(np.arange(12), n_f, label='Original')
    plt.bar(np.arange(12), n_s, label='Avg 16 XY-Translations')
    plt.xlabel('Layer')
    plt.ylabel('Avg Activation')
    plt.title('Effect of Translation (Normalized by Activation before Translation)')
    plt.legend()
    plt.savefig('shift_normalized.pdf')
    plt.show()


if __name__ == '__main__':
    main()
