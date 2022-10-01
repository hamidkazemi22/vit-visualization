from torchvision.datasets import VisionDataset

from datasets.base import EasyDataset
from torchvision import transforms as trans
from torchvision.datasets import ImageFolder


class ImageNet(EasyDataset):
    _root = './data/imagenet/{}'

    def __init__(self):
        super(ImageNet, self).__init__(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.eval_transforms = trans.Compose([trans.Resize(224), trans.CenterCrop(224), trans.ToTensor(), ])
        self.train_transforms = trans.Compose([trans.Resize(224), trans.CenterCrop(224), trans.ToTensor(), ])
        #self.train_transforms = trans.Compose(
        #    [trans.RandomResizedCrop(224), trans.RandomHorizontalFlip(), trans.ToTensor()])

    def eval(self) -> VisionDataset:
        return ImageFolder(root=self._root.format('val'), transform=self.eval_transforms)

    def train(self) -> VisionDataset:
        # return ImageFolder(root=self._root.format('train'), transform=self.train_transforms)
        return ImageFolder(root=self._root.format('train'), transform=self.eval_transforms)


image_net = ImageNet()
