from torch.utils import model_zoo
from torchvision.datasets import ImageFolder
import torch
from torchvision.transforms import ToTensor, ToPILImage


class BackgroundForegroundImageNet(ImageFolder):
    boxes_url = 'https://github.com/AminJun/ImageNet1KBoundingBoxes/releases/download/files/boxes.pt'
    indices_url = 'https://github.com/AminJun/ImageNet1KBoundingBoxes/releases/download/files/indices.pt'

    def __init__(self, root: str = './data/imagenet/train/', download=True, boxes: str = None, indices: str = None,
                 *args, **kwargs):
        assert download or (boxes is not None and indices is not None)
        if download:
            self.boxes = model_zoo.load_url(self.boxes_url, map_location='cpu')
            self.b_indices = model_zoo.load_url(self.indices_url, map_location='cpu')
        else:
            self.boxes = torch.load(boxes)
            self.b_indices = torch.load(indices)

        merged = {}
        for k, v in self.boxes.items():
            merged.update(v)
        self.boxes = merged

        self.pre_transform = ToTensor()
        self.back_transform = ToPILImage()
        print('loading imagenet folders')
        super(BackgroundForegroundImageNet, self).__init__(root, *args, **kwargs)

    def __len__(self):
        return len(self.b_indices)

    def __getitem__(self, item):
        real_i = self.b_indices[item]
        path, target = self.samples[real_i]
        sample = self.pre_transform(self.loader(path))
        background = sample.clone().detach()
        for box in self.boxes[path.split('/')[-1]][0]:
            x1, x2, y1, y2 = box
            background[:, int(y1):int(y2), int(x1):int(x2)] = 0
        foreground = (sample - background).detach().clone()

        sample, background, foreground = self.back_transform(sample), self.back_transform(
            background), self.back_transform(foreground)

        if self.transform is not None:
            sample, background, foreground = self.transform(sample), self.transform(background), self.transform(
                foreground)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, background, foreground, target
