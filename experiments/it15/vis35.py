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
    layer, feature, grid, network = args.layer, args.feature, args.grid, args.network
    model, image_size, _, _ = model_library[network]()
    import pdb; pdb.set_trace()
    tv = args.tv

    saver = ExperimentSaver(f'VisL{layer}_F{feature}_N{network}_TV{tv}', save_id=True, disk_saver=True)

    loss = LossArray()
    loss += ViTEnsFeatHook(ViTGeLUHook(model, sl=slice(layer, layer + 1)), key='high', feat=feature, coefficient=1)
    loss += TotalVariation(2, image_size, coefficient=0.0005 * tv)

    pre, post = torch.nn.Sequential(RepeatBatch(8), ColorJitter(8, shuffle_every=True),
                                    GaussianNoise(8, True, 0.5, 400), Tile(image_size // image_size), Jitter()), Clip()

    image = new_init(image_size, 1)
    visualizer = ImageNetVisualizer(loss, None, pre, post, print_every=10, lr=0.1, steps=400, save_every=100)
    image.data = visualizer(image)
    saver.save(image, 'final')


if __name__ == '__main__':
    main()
