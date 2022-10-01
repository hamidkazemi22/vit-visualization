import torchvision
from typing import List, Any
from torchvision import transforms as trans
from torchvision.datasets import VisionDataset
from datasets.base import EasyDataset


class Cifar(EasyDataset):
    def __init__(self, mean: List, std: List, data_class: Any):
        super(Cifar, self).__init__(mean, std)
        self.data_class = data_class

        # self.normalize = cnns.transforms.Normalize(mean=self.mean, std=self.std)
        self.eval_transforms = trans.Compose([trans.ToTensor()])  # , self.normalize])
        self.train_transforms = trans.Compose([trans.RandomCrop(size=32, padding=4), trans.RandomHorizontalFlip(),
                                               trans.ToTensor()])  # , self.normalize])

    def eval(self) -> VisionDataset:
        return self.data_class(root='./data', train=False, download=True, transform=self.eval_transforms)

    def train(self) -> VisionDataset:
        return self.data_class(root='./data', train=True, download=True, transform=self.train_transforms)


cifar10 = Cifar(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010], data_class=torchvision.datasets.CIFAR10)
cifar100 = Cifar(mean=[0.5071, 0.4865, 0.4409], std=[0.2009, 0.1984, 0.2023], data_class=torchvision.datasets.CIFAR100)
