import os

import torch
from torch.utils.data import DataLoader

from augmentation import Clip, Tile, Jitter, RepeatBatch, ColorJitter
from augmentation.pre import GaussianNoise
from hooks.transformer.vit import ViTAttHookHolder, ViTGeLUHook, ClipGeLUHook
from inversion import ImageNetVisualizer
from inversion.utils import new_init
from loss import LossArray, TotalVariation
from loss.image_net import ViTFeatHook, ViTEnsFeatHook
from model import model_library
from saver import ExperimentSaver
from utils import exp_starter_pack
import torchvision.transforms as transforms
import torchvision


def get_cifar100_loaders(batch_size=128, train=None):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=train, download=True,
                                                      transform=transform_train)
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=True, num_workers=4, batch_size=batch_size)

    return cifar100_training_loader


def main():
    args = exp_starter_pack()[1]
    layer, feature = args.layer, args.feature
    network = args.network
    tv = args.tv
    model, image_size, _, _ = model_library[network]()
    train_loader = get_cifar100_loaders(train=True)
    val_loader = get_cifar100_loaders(train=False)
    corrects, total = 0, 0
    epochs = 20
    criterion = torch.nn.CrossEntropyLoss()
    model.requires_grad_(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    classifier = torch.nn.Linear(in_features=model.classifier[1].fc.in_features, out_features=100).cuda()
    model.classifier[1].fc = classifier
    model = torch.nn.DataParallel(model)
    for epoch in range(epochs):
        model.train()
        for image, target in train_loader:
            optimizer.zero_grad()
            image, target = image.cuda(), target.cuda()
            output = model(image)
            loss = criterion(output, target)
            preds = output.argmax(dim=-1)
            corrects += (preds == target).sum()
            total += preds.shape[0]
            acc = corrects / total
            print(f'acc train is {acc}, loss is {loss}', end='\r')
        print(f'acc train is {acc}, loss is {loss}')
        scheduler.step()
        model.eval()
        for image, target in train_loader:
            image, target = image.cuda(), target.cuda()
            output = model(image)
            preds = output.argmax(dim=-1)
            corrects += (preds == target).sum()
            total += preds.shape[0]
            acc = corrects / total
            print(f'test acc is {acc}', end='\r')
        print(f'test acc is {acc}')
    cp_dir = 'cp/'
    os.makedirs(cp_dir)
    torch.save(model, os.path.join(cp_dir, 'cifar100.pt'))

if __name__ == '__main__':
    main()
