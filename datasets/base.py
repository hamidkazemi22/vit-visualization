from torchvision.datasets import VisionDataset

from datasets.normalizer import Normalizer


class EasyDataset:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.normalizer = Normalizer(mean, std)

    def eval(self) -> VisionDataset:
        raise NotImplementedError

    def train(self) -> VisionDataset:
        raise NotImplementedError
