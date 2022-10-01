import os
import torch
import torchvision.transforms
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader


class GreatInit(Dataset):
    def __init__(self, root: str = 'data/inits/images', subset: str = 'inv_resnet'):
        self.path = os.path.join(root, subset)
        self.files = os.listdir(self.path)
        to_tensor = torchvision.transforms.ToTensor()
        self.images = {f.split('.')[0]: to_tensor(default_loader(os.path.join(self.path, f))) for f in self.files}

    def __len__(self):
        return len(self.images.keys())

    def __getitem__(self, item: int) -> torch.tensor:
        return self.images[str(item)].unsqueeze(0)


inv_dataset = GreatInit()
