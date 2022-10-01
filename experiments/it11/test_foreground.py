import torchvision.utils
import torch
import torchvision.transforms as trans

from costum_imagenet import BackgroundForegroundImageNet


def test1():
    tr = trans.Compose([trans.Resize(224), trans.CenterCrop(224), trans.ToTensor(), ])
    dataset = BackgroundForegroundImageNet(root='./data/imagenet/train', download=True, transform=tr)
    x, b, f, y = dataset[0]
    torchvision.utils.save_image(torch.stack([x, b, f]), 'test1.png')


def test2():
    tr = trans.Compose([trans.Resize(224), trans.CenterCrop(224), trans.ToTensor(), ])
    dataset = BackgroundForegroundImageNet(root='./data/imagenet/train', download=False, boxes='boxes.pt',
                                           indices='indices.pt', transform=tr)
    x, b, f, y = dataset[1]
    torchvision.utils.save_image(torch.stack([x, b, f]), 'test2.png')


if __name__ == '__main__':
    test1()
    test2()
