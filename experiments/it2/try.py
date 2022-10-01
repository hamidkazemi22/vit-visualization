import pdb

import torchvision
import torch
from torchvision.datasets.folder import default_loader


def main():
    to_tensor = torchvision.transforms.ToTensor()
    image = to_tensor(default_loader('try.png'))
    torchvision.utils.save_image((image - image.mean()) * 3 + image.mean(), 't1.png')
    pdb.set_trace()

if __name__ == '__main__':
    main()
