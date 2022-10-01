import torch

from hooks.transformer.vit import ViTAttHookHolder
from model import model_library
from utils import exp_starter_pack
from datasets import weird_image_net


def main():
    exp_name, args, _ = exp_starter_pack()
    model, image_size, batch_size, name = model_library[34]()
    eval = weird_image_net.eval()

    hooks = ViTAttHookHolder(model, True, True, True, True, True, True)

    """ Result: 18 """
    # images, labels = [], []
    # for i in range(1000):
    #     x, y = eval[i]
    #     images.append(x)
    #     labels.append(y)
    #
    #     xs = torch.stack(images).cuda()
    #     with torch.no_grad():
    #         _ = hooks(xs)
    #         print(f'Can be done: {len(xs)} w/o grad')

    """ Result: 10 """
    images, labels = [], []
    for i in range(1000):
        x, y = eval[i]
        images.append(x)
        labels.append(y)

        xs = torch.stack(images).cuda()
        cl = xs.clone()
        cl.requires_grad_()
        _, out = hooks(cl)
        out.sum().backward()
        print(f'Can be done: {len(xs)} w/ grad')


if __name__ == '__main__':
    main()
