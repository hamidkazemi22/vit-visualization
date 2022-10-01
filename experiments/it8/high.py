import torch
from augmentation import Clip, Tile, Jitter, RepeatBatch, ColorJitter
from augmentation.pre import GaussianNoise
from hooks.transformer.vit import ViTAttHookHolder, ViTGeLUHook
from inversion import ImageNetVisualizer
from inversion.utils import new_init
from loss import LossArray, TotalVariation
from loss.image_net import ViTFeatHook, ViTEnsFeatHook
from model import model_library
from saver import ExperimentSaver
from utils import exp_starter_pack


def main():
    args = exp_starter_pack()[1]
    layer, feature, grid, method, network = args.layer, args.feature, args.grid, args.method, args.network
    coef, lr = args.sign, args.lr
    method = 'high'
    tv = 0.0005
    model, image_size, _, _ = model_library[network]()

    saver = ExperimentSaver(f'High{layer}F{feature}_{tv}x{grid}_LR{lr}_N{network}_M{method}_S{coef}', save_id=True,
                            disk_saver=True)

    loss = LossArray()
    loss += ViTEnsFeatHook(ViTGeLUHook(model, sl=slice(layer, layer + 1)), key=method, feat=feature,
                           coefficient=1 * coef)
    loss += TotalVariation(2, 384, coefficient=tv * grid)

    pre, post = torch.nn.Sequential(RepeatBatch(8), ColorJitter(8, shuffle_every=True),
                                    GaussianNoise(8, True, 0.5, 400), Tile(image_size // image_size), Jitter()), Clip()
    image = new_init(image_size, 1)

    visualizer = ImageNetVisualizer(loss, saver, pre, post, print_every=10, lr=lr, steps=400, save_every=100)
    image.data = visualizer(image)
    saver.save(image, 'final')


if __name__ == '__main__':
    main()
