import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from runner.adversarial import Modifier
from utils.device import to_cuda
from utils.classification import get_acc
from runner.status import Status, Meter


class BaseRunner:
    def __init__(self, model: nn.Module, loader: DataLoader = None, loss: nn.Module = None,
                 optimizer: Optimizer = None, aug: callable = None, attacker: Modifier = None):
        self.model = model.cuda()
        self.loader = loader
        self.loss = loss.cuda()
        self.optimizer = optimizer
        self.aug = aug if aug else None
        self.attacker = attacker

    def _method_for_one_epoch(self) -> (float, float):
        loss = Meter()
        accuracy = Meter()
        status = Status(self.loader, 3, template='{} {}'.format(self.__class__.__name__, 'Acc:{} Loss:{} ({})'))
        status.print(*['xx.xx'] * 3)

        for i, (data) in enumerate(status):
            x, y = to_cuda(data)
            x = self.attacker(x, y) if self.attacker else x
            x = self.aug(x) if self.aug else x

            output = self.model(x)
            _, correct = get_acc(output, y)
            cur_loss = self.loss(output, y)

            if self.optimizer is not None:
                self.optimizer.zero_grad()
                cur_loss.backward()
                self.optimizer.step(closure=None)

            accuracy += (correct, len(x))
            loss += (cur_loss.item() * len(x), len(x))
            status.print(accuracy.value * 100, loss.value, cur_loss.item())
        return accuracy.value * 100, loss.value


class Trainer(BaseRunner):
    def __init__(self, model: nn.Module, loader: DataLoader = None, loss: nn.Module = None, aug: nn.Module = None,
                 optimizer: Optimizer = None, attacker: Modifier = None):
        super().__init__(model, loader=loader, optimizer=optimizer, loss=loss, aug=aug, attacker=attacker)

    def train(self) -> (float, float):
        self.model.train()
        with torch.enable_grad():
            return self._method_for_one_epoch()


class Evaluator(BaseRunner):
    def __init__(self, model: torch.nn.Module, loader: DataLoader = None, loss: nn.Module = None,
                 aug: nn.Module = None, attacker: Modifier = None):
        super().__init__(model, loader=loader, loss=loss, optimizer=None, aug=aug, attacker=attacker)

    @torch.no_grad()
    def eval(self) -> (float, float):
        self.model.eval()
        return self._method_for_one_epoch()
