import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from runner.status import Status, Meter
from utils.classification import get_acc
from utils.device import to_cuda
import pdb


class CorrectClassLoss(nn.Module):
    def forward(self, output: torch.tensor, y: torch.tensor):
        return output.gather(1, y.view(-1, 1)).mean()


class CorrectTrainer:
    def __init__(self, model: nn.Module, loader: DataLoader = None, loss: nn.Module = None, optimizer: Optimizer = None,
                 aug: callable = None, steps: int = 8, eps: float = 8. / 255., step_size: float = 0.):
        self.model = model.cuda()
        self.loader = loader
        self.loss = loss.cuda()
        self.optimizer = optimizer
        self.aug = aug.cuda() if aug else None
        self.steps = steps
        self.eps = eps
        self.step_size = step_size or (2. * eps / steps)
        self.adv_loss = CorrectClassLoss()

    def _method_for_one_epoch(self) -> float:
        loss = Meter()
        accuracy = Meter()
        status = Status(self.loader, 3, template='Acc:{} Loss:{} ({})')
        status.print(*['xx.xx'] * 3)

        sample_x, _ = next(iter(self.loader))
        noise = sample_x.cuda().clone().detach().requires_grad_()
        for i, (data) in enumerate(status):
            x, y = to_cuda(data)
            x = self.aug(x) if self.aug else x

            # noise.zero_()
            for rep in range(self.steps):
                adv_x = (x + noise[:len(x)]).clamp(0., 1.)
                self.model.eval()
                output = self.model(adv_x)
                _, correct1 = get_acc(output, y)
                adv_loss = self.adv_loss(output, y)
                self.optimizer.zero_grad()
                adv_loss.backward()
                noise.data = (noise.data + self.step_size * noise.grad.sign()).clamp(-self.eps, self.eps)

                self.model.train()
                adv_x = (x + noise[:len(x)]).clamp(0., 1.)
                output = self.model(adv_x)
                _, correct = get_acc(output, y)
                cur_loss = self.loss(output, y)
                # print(correct, correct1)

                self.optimizer.zero_grad()
                cur_loss.backward()
                self.optimizer.step(closure=None)

                accuracy += (correct, len(x))
                loss += (cur_loss.item() * len(x), len(x))
                status.print(accuracy.value * 100, loss.value, cur_loss.item())
        return accuracy.value * 100

    def train(self) -> float:
        self.model.train()
        with torch.enable_grad():
            return self._method_for_one_epoch()
