import pdb
from typing import Union

import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets.folder import default_loader
from tqdm import tqdm

from datasets import weird_image_net
from model import model_library
from model.augmented import AugmentedModel
from utils import exp_starter_pack
import torch
from torch import nn
from pytorch_pretrained_vit import ViT

from utils.device import to_cuda
from utils.statistics import Meter


class ViTInput(nn.Module):
    def __init__(self, vit: Union[ViT, AugmentedModel]):
        super().__init__()
        vit = vit if not isinstance(vit, AugmentedModel) else vit.classifier[1]
        self.patch_embedding = vit.patch_embedding
        self.class_token = vit.class_token
        self.positional_embedding = vit.positional_embedding

    def forward(self, x: torch.tensor) -> torch.tensor:
        b, c, fh, fw = x.shape
        x = self.patch_embedding(x)  # b,d,gh,gw
        x = x.flatten(2).transpose(1, 2)  # b,gh*gw,d
        x = torch.cat((self.class_token.expand(b, -1, -1), x), dim=1)  # b,gh*gw+1,d
        return self.positional_embedding(x)  # b,gh*gw+1,d


class ViTPre(nn.Sequential):
    def __init__(self, vit: AugmentedModel):
        super().__init__(vit.augmentations, vit.classifier[0], ViTInput(vit.classifier[1]))


class ViTTransformer(nn.Module):
    def __init__(self, vit: Union[ViT, AugmentedModel], sl: slice = None):
        super().__init__()
        vit = vit if not isinstance(vit, AugmentedModel) else vit.classifier[1]
        sl = sl if sl is not None else slice(None, None)
        self.blocks = vit.transformer.blocks[sl]

    def forward(self, x: torch.tensor, mask: torch.tensor = None) -> torch.tensor:
        for b in self.blocks:
            x = b(x, mask)
        return x


class ViTHead(nn.Module):
    def __init__(self, vit: Union[ViT, AugmentedModel], sl: slice = None):
        super().__init__()
        vit = vit if not isinstance(vit, AugmentedModel) else vit.classifier[1]
        self.sl = sl if sl is not None else slice(0, 1)
        self.norm = vit.norm
        self.fc = vit.fc

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.norm(x)[:, self.sl].mean(dim=1)  # b,d
        return self.fc(x)


@torch.no_grad()
def main():
    exp_name, args, _ = exp_starter_pack()
    network = args.network
    model_base, image_size, batch_size, name = model_library[network]()
    to_tensor = torchvision.transforms.ToTensor()
    img = to_tensor(default_loader('10_22.png'))
    model = nn.Sequential(ViTPre(model_base), ViTTransformer(model_base), ViTHead(model_base, slice(1, None)))

    loss = nn.CrossEntropyLoss()

    loader = DataLoader(weird_image_net.eval(), num_workers=4, shuffle=True, batch_size=batch_size)
    ls, ac, ac5 = Meter(), Meter(), Meter()

    model.eval()
    for i, (data) in enumerate(tqdm(loader)):
        x, y = to_cuda(data)
        output = model(x)
        correct1 = (output.argmax(dim=-1) == y).sum()
        correct5 = (output.topk(k=5, sorted=False)[1] == y.view(-1, 1)).any(dim=-1).sum()
        cur_loss = loss(output, y)
        ac += (correct1, len(x))
        ac5 += (correct5, len(x))
        ls += (cur_loss.cpu().item() * len(x), len(x))
        if i % 100 == 0:
            print(name, f'AVG(Patches-CLS)_{network}: \t{ac.value * 100} \t {ac5.value * 100} \t {ls.value}')

    with open('Acc.txt', 'a') as f:
        print(name, f'AVG(Patches-CLS)_{network}: \t{ac.value * 100} \t {ac5.value * 100} \t {ls.value}', file=f)
        print(name, f'AVG(Patches-CLS)_{network}: \t{ac.value * 100} \t {ac5.value * 100} \t {ls.value}')


if __name__ == '__main__':
    main()
