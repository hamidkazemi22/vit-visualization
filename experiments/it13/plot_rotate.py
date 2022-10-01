import math
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
from torch import nn
from matplotlib import pyplot as plt
import numpy as np


@torch.no_grad()
def main():
    exp_name, args, _ = exp_starter_pack()
    before = torch.load('rotate_before.pt', map_location='cpu').clamp(min=0)
    after = torch.load('rotate_after.pt', map_location='cpu').clamp(min=0)

    # gelu = nn.GELU()
    # mn_value = min(gelu((torch.arange(1000) / 1000.0) - 1))
    # before -= mn_value
    # after -= mn_value
    # print(mn_value)

    plt.figure()
    nn_f = before.mean(dim=-1)
    nn_s = after.mean(dim=-1)
    plt.bar(np.arange(12), nn_f, label='Original')
    plt.bar(np.arange(12), nn_s, label='Avg 18 Rotations')
    plt.xlabel('Layer')
    plt.ylabel('Avg Activation')
    plt.title('Effect of Rotation on Activation')
    plt.legend()
    plt.savefig('rotation.pdf')
    plt.show()

    plt.cla()
    n_f = (before.view(-1) / before.view(-1)).view(before.shape)
    indices = [a > 0 for a in before]
    n_f = [a[indices[i]].mean() for i, a in enumerate(n_f)]
    n_s = (after.view(-1) / before.view(-1)).view(before.shape)
    n_s = [a[indices[i]].mean() for i, a in enumerate(n_s)]

    # low = min(n_s)
    # high = max(n_s)
    # print(low)
    # plt.ylim([math.floor(low - 0.5 * (high - low)) - 1, math.ceil(high + 0.5 * (high - low)) + 1])
    plt.bar(np.arange(12), n_f, label='Original')
    plt.bar(np.arange(12), n_s, label='Avg 18 Rotations')
    plt.xlabel('Layer')
    plt.ylabel('Avg Activation')
    plt.title('Effect of Rotation on Activation (Normalized by Activation before Rotation)')
    plt.legend()
    plt.savefig('rotation_normalized.pdf')
    plt.show()


if __name__ == '__main__':
    main()
