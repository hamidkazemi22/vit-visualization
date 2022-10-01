import torch
from augmentation import Clip, Tile, Jitter, RepeatBatch, ColorJitter
from augmentation.pre import GaussianNoise
from hooks.transformer.vit import ViTAttHookHolder, ViTGeLUHook, ReconstructionViTGeLUHook
from inversion import ImageNetVisualizer
from inversion.utils import new_init
from loss import LossArray, TotalVariation
from loss.image_net import ViTFeatHook, ViTEnsFeatHook, ReconstructionLoss
from model import model_library
from saver import ExperimentSaver
from utils import exp_starter_pack
from datasets import weird_image_net


def main():
    args = exp_starter_pack()[1]
    layer, feature, grid, network = args.layer, args.feature, args.grid, args.network
    model, image_size, _, _ = model_library[network]()

    saver = ExperimentSaver(f'RecL{layer}_Eval{feature}_N{network}_TV', save_id=True, disk_saver=True)
    data = weird_image_net.eval()
    x, y = data[feature]
    x = x.unsqueeze(0).cuda()

    loss = LossArray()
    loss += ReconstructionLoss(ReconstructionViTGeLUHook(model, sl=slice(layer, layer + 1)), x, key='high',
                               feat=feature, coefficient=1)
    loss += TotalVariation(2, image_size, coefficient=1.)

    pre, post = torch.nn.Sequential(
        # RepeatBatch(8),
        # ColorJitter(8, shuffle_every=True),
        # GaussianNoise(8, True, 0.5, 400),
        # Jitter()
    ), Clip()

    image = new_init(image_size, 1)
    visualizer = ImageNetVisualizer(loss, None, pre, post, print_every=10, lr=0.1, steps=400, save_every=100)
    image.data = visualizer(image)
    saver.save(torch.cat([x, image]), 'final')


if __name__ == '__main__':
    main()
