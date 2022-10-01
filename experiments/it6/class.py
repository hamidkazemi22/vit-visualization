import os
import pdb

import torchvision
from torchvision.datasets.folder import default_loader

from inversion import ImageNetVisualizer
import json
from model import model_library
from utils import exp_starter_pack


def main():
    exp_name, args, _ = exp_starter_pack()
    network = 34
    model, image_size, _, _ = model_library[network]()
    subset = args.dir

    path = os.path.join('data', subset)
    files = os.listdir(path)
    to_tensor = torchvision.transforms.ToTensor()

    outputs = {}
    for f in files:
        if 'png' not in f:
            continue
        image = to_tensor(default_loader(os.path.join(path, f)))
        image = image.view(1, 3, image_size, image_size).cuda()
        labels = model(image)
        outputs[f] = (labels.topk(k=5)[1][0]).cpu().numpy().tolist()

    js_formatted = json.dumps(outputs)
    with open(os.path.join(path, 'top5.json'), 'w') as f:
        print(f'top5 = {js_formatted}', file=f)




if __name__ == '__main__':
    main()
