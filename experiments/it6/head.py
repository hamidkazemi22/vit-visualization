import torch
from augmentation import Clip, Tile, Jitter, RepeatBatch, ColorJitter
from augmentation.pre import GaussianNoise
from hooks.transformer.vit import ViTAttHookHolder
from inversion import ImageNetVisualizer
from inversion.utils import new_init
from loss import LossArray, TotalVariation
from loss.image_net import ViTFeatHook, ViTEnsFeatHook, ViTHeadHook
from model import model_library
from saver import ExperimentSaver
from utils import exp_starter_pack


def main():
    args = exp_starter_pack()[1]
    layer, feature, method, network, coef = args.layer, args.feature, args.method, args.network, args.sign
    lr = args.lr
    tv = 0.0005 * 0.01
    model, image_size, _, _ = model_library[network]()

    saver = ExperimentSaver(f'Head{layer}H{feature}_{tv}_N{network}_M{method}_S{coef}_LR{lr}', save_id=True,
                            disk_saver=True)

    loss = LossArray()
    loss += ViTHeadHook(ViTAttHookHolder(model, sl=slice(layer, layer + 1), **{method: True}), key=method, feat=feature,
                        coefficient=1 * coef)
    loss += TotalVariation(2, 384, coefficient=tv)

    pre, post = torch.nn.Sequential(RepeatBatch(8), ColorJitter(8, shuffle_every=True),
                                    GaussianNoise(8, True, 0.5, 400), Tile(image_size // image_size), Jitter()), Clip()
    image = new_init(image_size, 1)

    visualizer = ImageNetVisualizer(loss, saver, pre, post, print_every=10, lr=lr, steps=400, save_every=400)
    image.data = visualizer(image)
    saver.save(image, 'final')


if __name__ == '__main__':
    main()
