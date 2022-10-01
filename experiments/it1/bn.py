from torchvision.models import resnet18

from augmentation import Clip, Jitter, Centering, RepeatBatch, Zoom, ColorJitter
from datasets import image_net
from hooks import LayerHook
from inversion import ImageNetVisualizer
from inversion.utils import new_init
from loss import LossArray, TotalVariation, LayerActivationNorm, ActivationHook, MatchBatchNorm
from loss.regularizers import FakeBatchNorm
from model import model_library
from saver import ExperimentSaver
from utils import exp_starter_pack
import torch


def main():
    exp_name, args, _ = exp_starter_pack()
    network = args.network
    layer, feature = args.layer, args.feature
    model, image_size, batch_size, name = model_library[network]()
    grid = args.grid
    batch_size = min(64, batch_size)

    saver = ExperimentSaver(f'GridSearchBN{layer}{name}_{grid}_{feature}', save_id=True, disk_saver=True)

    loss = LossArray()

    layer_hook = LayerHook(model, torch.nn.BatchNorm2d, layer, ActivationHook)
    loss += LayerActivationNorm(layer_hook, model, coefficient=1)

    resnet_bn = FakeBatchNorm(resnet18, image_net.normalizer).cuda()
    loss += MatchBatchNorm(resnet_bn, coefficient=50.0 * grid)
    loss += TotalVariation(size=image_size, coefficient=0.05)

    pre, post = None, torch.nn.Sequential(Clip())
    image = new_init(image_size, batch_size)

    visualizer = ImageNetVisualizer(loss, None, pre, post, print_every=10, lr=.01, steps=400, save_every=10)
    image.data = visualizer(image).data
    saver.save(image[:4])


if __name__ == '__main__':
    main()
