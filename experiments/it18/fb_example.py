import torchvision.utils
from torch.utils.data import DataLoader
from saver import ExperimentSaver
from utils import exp_starter_pack
import torch
from datasets.imagenet_boxes import BackgroundForegroundImageNet
import torchvision.transforms as tr
from tqdm import tqdm


@torch.no_grad()
def main():
    exp_name, args, _ = exp_starter_pack()
    trans = tr.Compose([tr.Resize(224), tr.CenterCrop(224), tr.ToTensor(), ])
    data = BackgroundForegroundImageNet(transform=trans)
    saver = ExperimentSaver(f'FB', save_id=True, disk_saver=True)
    count = 1
    for x, b, f, y in tqdm(data):
        saver.save(x, 'x')
        saver.save(b, 'b')
        saver.save(f, 'f')
        count -= 1
        if count == 0:
            break

    x, b, f, y = data[57]
    torchvision.utils.save_image([x, b, f], 'fore_back.pdf', padding=0)
    torchvision.utils.save_image([x, b, f], 'fore_back.png', padding=0)
    torchvision.utils.save_image(x, 'image.png')
    torchvision.utils.save_image(b, 'background.png')
    torchvision.utils.save_image(f, 'foreground.png')


if __name__ == '__main__':
    main()
