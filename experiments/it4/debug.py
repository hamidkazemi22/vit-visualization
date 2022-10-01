import torch
from augmentation import Clip, Tile, Jitter
from hooks.transformer.vit import ViTAttHookHolder
from inversion import ImageNetVisualizer
from inversion.utils import new_init
from loss import LossArray, TotalVariation
from loss.image_net import ViTFeatHook
from model import model_library
from saver import ExperimentSaver
from utils import exp_starter_pack


def main():
    exp_name, args, _ = exp_starter_pack()
    layer, method, network, patch, coef = args.layer, args.method, args.network, args.patch, float(args.sign)
    model, image_size, _, _ = model_library[network]()

    saver = ExperimentSaver(f'SGDJitterNet{network}{method}Layer{layer}SGN{coef}_{patch}', save_id=True,
                            disk_saver=True)

    loss = LossArray()
    loss += ViTFeatHook(ViTAttHookHolder(model, True, True, True, True, False, True, sl=slice(layer, layer + 1)),
                        key=method, coefficient=1 * coef)

    pre, post = torch.nn.Sequential(Tile(image_size // patch), Jitter()), Clip()
    # pre, post = None, Clip()
    image = new_init(patch, 8)
    # image = new_init(384, 8)

    visualizer = ImageNetVisualizer(loss, saver, pre, post, print_every=10, lr=.01, steps=400, save_every=100)
    image.data = visualizer(image)
    saver.save(pre(image), 'final')
    # saver.save(image, 'final')


if __name__ == '__main__':
    main()
