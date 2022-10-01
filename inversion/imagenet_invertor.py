import collections

import torch
import torch.nn as nn
import torch.optim as optim

from loss import LossArray

from saver import AbstractSaver


class ImageNetVisualizer:
    def __init__(self, loss_array: LossArray, saver: AbstractSaver = None, pre_aug: nn.Module = None,
                 post_aug: nn.Module = None, steps: int = 2000, lr: float = 0.1, save_every: int = 200,
                 print_every: int = 5, **_):
        self.loss = loss_array
        self.saver = saver

        self.pre_aug = pre_aug
        self.post_aug = post_aug

        self.save_every = save_every
        self.print_every = print_every
        self.steps = steps
        self.lr = lr

    def __call__(self, img: torch.tensor = None, optimizer: optim.Optimizer = None):
        img = img.detach().clone().to('cuda:0').requires_grad_()

        optimizer = optimizer if optimizer is not None else optim.Adam([img], lr=self.lr, betas=(0.5, 0.99), eps=1e-8)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.steps, 0.)

        print(f'#i\t{self.loss.header()}', flush=True)

        for i in range(self.steps + 1):
            optimizer.zero_grad()
            augmented = self.pre_aug(img) if self.pre_aug is not None else img
            loss = self.loss(augmented)

            if i % self.print_every == 0:
                print(f'{i}\t{self.loss}', flush=True)
            if i % self.save_every == 0 and self.saver is not None:
                self.saver.save(img, i)

            # loss.backward(retain_graph=True)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            img.data = (self.post_aug(img) if self.post_aug is not None else img).data

            self.loss.reset()
            torch.cuda.empty_cache()

        optimizer.state = collections.defaultdict(dict)
        return img
