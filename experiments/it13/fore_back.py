import torchvision.utils
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from datasets import weird_image_net
from hooks.transformer.vit import SimpleViTGeLUHook
from model import model_library
from runner.status import Meter
from utils import exp_starter_pack
import torch
from tqdm import tqdm
from datasets.imagenet_boxes import BackgroundForegroundImageNet
import torchvision.transforms as tr
from torch import nn
import pdb

from utils.classification import get_acc, get_acc5
from utils.device import to_cuda


@torch.no_grad()
def main():
    exp_name, args, _ = exp_starter_pack()
    network = args.network
    model, image_size, batch_size, name = model_library[network]()
    trans = tr.Compose([tr.Resize(image_size), tr.CenterCrop(image_size), tr.ToTensor(), ])
    data = DataLoader(BackgroundForegroundImageNet(transform=trans), num_workers=4, shuffle=True, batch_size=batch_size)
    model.eval()
    loss = nn.CrossEntropyLoss()

    loss3 = [Meter(), Meter(), Meter()]
    acc31 = [Meter(), Meter(), Meter()]
    acc35 = [Meter(), Meter(), Meter()]

    model.eval()
    for i, (data) in enumerate(data):
        x, b, f, y = to_cuda(data)
        y_flat = y.view(-1, 1)
        img = [x, b, f]
        for im, ac1, ac5, ls in zip(img, acc31, acc35, loss3):
            output = model(im)
            correct1 = (output.argmax(dim=-1) == y).sum()
            correct5 = (output.topk(k=5, sorted=False)[1] == y_flat).any(dim=-1).sum()
            cur_loss = loss(output, y)
            ac1 += (correct1, len(im))
            ac5 += (correct5, len(im))
            ls += (cur_loss.item() * len(im), len(im))
    with open('FBAcc.txt', 'a') as f:
        print(name, f'x: {acc31[0].value * 100} \t {acc35[0].value * 100} \t {loss3[0].value}\t',
              f'b: {acc31[1].value * 100} \t {acc35[1].value * 100} \t {loss3[1].value} \t',
              f'f: {acc31[2].value * 100} \t {acc35[2].value * 100} \t {loss3[2].value}',
              file=f)
        print(name, f'x: {acc31[0].value * 100} \t {acc35[0].value * 100} \t {loss3[0].value}\t',
              f'b: {acc31[1].value * 100} \t {acc35[1].value * 100} \t {loss3[1].value} \t',
              f'f: {acc31[2].value * 100} \t {acc35[2].value * 100} \t {loss3[2].value}')


if __name__ == '__main__':
    main()
