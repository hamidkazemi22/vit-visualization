from augmentation import Clip
import pdb 
from augmentation.post import ClipSTD
from augmentation.pre import Layered
from hooks import LayerHook
from inversion import ImageNetVisualizer
from inversion.utils import new_init
from loss import LossArray, TotalVariation, LayerActivationNorm, NormalVariation
from loss.hooks.activation import ActivationReluHook
from model import model_library
from saver import ExperimentSaver
from utils import exp_starter_pack
import torch


def main():
    exp_name, args, _ = exp_starter_pack()
    network, layer, feature = args.network, args.layer, args.feature
    model, image_size, batch_size, name = model_library[network]()
    layer_hook = LayerHook(model, torch.nn.BatchNorm2d, layer, ActivationReluHook)
    batch_size = min(64, batch_size)

    saver = ExperimentSaver(f'OneShot2V{layer}{name}_{feature}', save_id=True, disk_saver=True)

    clipper = ClipSTD()

    image = new_init(image_size, batch_size, last=None)
    pdb.set_trace()

    pre, post = None, torch.nn.Sequential(Clip(), ClipSTD(), Clip())
    pre, post = None, Clip()

    saver.save(image, 'pre')
    loss = LossArray()
    loss += LayerActivationNorm(layer_hook, model, coefficient=1)
    loss += TotalVariation(size=image_size, coefficient=0.05)
    visualizer = ImageNetVisualizer(loss, saver, pre, post, print_every=10, lr=.01, steps=200, save_every=40)
    image.data = visualizer(image).data
    saver.save(image, 'noise')
    delta = clipper(image)
    saver.save(delta , 'total')


if __name__ == '__main__':
    main()
