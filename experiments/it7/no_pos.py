import pdb

import torchvision.utils
from torch.utils.data import DataLoader
from datasets import weird_image_net
from model import model_library
from utils import exp_starter_pack
from torch import nn
from runner import Evaluator
import torch


class PatchShuffle(nn.Module):
    def __init__(self, image_size: int = 384, patch_size=16):
        super().__init__()
        self.im, self.p, self.dim = image_size, patch_size, image_size // patch_size

    def forward(self, x: torch.tensor) -> torch.tensor:
        bs = x.size(0)
        x = x.reshape(bs, 3, self.dim, self.p, self.dim, self.p).permute(0, 2, 4, 1, 3, 5).reshape(bs, -1, 3, self.p,
                                                                                                   self.p)
        x = x[:, torch.randperm(self.dim * self.dim)]
        x = x.reshape(bs, self.dim, self.dim, 3, self.p, self.p).permute(0, 3, 1, 4, 2, 5).reshape(bs, 3, self.im,
                                                                                                   self.im)
        return x


def main():
    _ = exp_starter_pack()
    network = 35
    model, image_size, batch_size, _ = model_library[network]()
    loader = DataLoader(weird_image_net.eval(), batch_size=6 * batch_size, num_workers=4, shuffle=True)
    evaluator = Evaluator(model, loader, nn.CrossEntropyLoss())
    """
        34: About 84%:
        35: About 81%
    acc, loss = evaluator.eval()
    print(acc, loss)
    """
    """
        34: About 22%
        35: About 58%
    model.classifier[1].positional_embedding.pos_embedding.fill_(0)
    acc, loss = evaluator.eval()
    print(acc, loss) 
    """
    """
        34: About 22%
        35: About 58%
    nn.init.normal_(model.classifier[1].positional_embedding.pos_embedding, std=0.02)
    acc, loss = evaluator.eval()
    print(acc, loss)
    """
    evaluator = Evaluator(model, loader, nn.CrossEntropyLoss(), aug=PatchShuffle(image_size, 32))
    acc, loss = evaluator.eval()
    print(acc, loss)
    """
        34: About 33% 
        35: About 54%
    """


if __name__ == '__main__':
    main()
