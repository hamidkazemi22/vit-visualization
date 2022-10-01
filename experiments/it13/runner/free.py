import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from runner.status import Status, Meter
from utils.classification import get_acc
from utils.device import to_cuda


class FreeTrainer:
    def __init__(self, model: nn.Module, loader: DataLoader = None, loss: nn.Module = None, optimizer: Optimizer = None,
                 aug: callable = None, steps: int = 8, eps: float = 8. / 255., step_size: float = 0.):
        self.model = model.cuda()
        self.loader = loader
        self.loss = loss.cuda()
        self.optimizer = optimizer
        self.aug = aug.cuda() if aug else None
        self.steps = steps
        self.eps = eps
        # self.step_size = step_size or (2. * eps / steps)
        self.step_size = step_size or eps

    def _method_for_one_epoch(self) -> float:
        loss = Meter()
        accuracy = Meter()
        status = Status(self.loader, 3, template='{} {}'.format(self.__class__.__name__, 'Acc:{} Loss:{} ({})'))
        status.print(*['xx.xx'] * 3)

        sample_x, _ = next(iter(self.loader))
        noise = sample_x.cuda().clone().detach().requires_grad_()
        for i, (data) in enumerate(status):
            x, y = to_cuda(data)
            x = self.aug(x) if self.aug else x

            for rep in range(self.steps):
                adv_x = (x + noise[:len(x)]).clamp(0., 1.)
                output = self.model(adv_x)
                _, correct = get_acc(output, y)
                cur_loss = self.loss(output, y)

                if self.optimizer is not None:
                    self.optimizer.zero_grad()
                    cur_loss.backward()
                    self.optimizer.step(closure=None)
                    noise.data = (noise.data + self.step_size * noise.grad.sign()).clamp(-self.eps, self.eps)

                    accuracy += (correct, len(x))
                    loss += (cur_loss.item() * len(x), len(x))
                    status.print(accuracy.value * 100, loss.value, cur_loss.item())
        return accuracy.value * 100

    def train(self) -> float:
        self.model.train()
        with torch.enable_grad():
            return self._method_for_one_epoch()
