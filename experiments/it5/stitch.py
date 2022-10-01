from torch.utils.data import DataLoader
from tqdm import tqdm
from cam.base import Stitcher, SniperStitcher
from cam.model import ViTPatchFeatWrapper
from datasets import weird_image_net
from hooks.transformer.vit import ViTAttHookHolder
from model import model_library
from saver import ExperimentSaver
from utils import exp_starter_pack
import torch


def get_patches(image_size: int, patch_size: int):
    im = image_size // patch_size  # 24
    while im != 1:
        yield im * patch_size
        for i in range(2, im+1):
            if im % i == 0:
                im = im // i
                break
    yield patch_size


def main():
    exp_name, args, _ = exp_starter_pack()
    loader = DataLoader(weird_image_net.eval(), batch_size=1, num_workers=1, shuffle=True)
    layer, feature = args.layer, args.feature
    patch_size = 16
    network, method = 34, 'in_feat'
    model, image_size, _, _ = model_library[network]()
    hook = ViTPatchFeatWrapper(ViTAttHookHolder(model, sl=slice(layer, layer + 1), **{method: True}), method, feature)
    saver = ExperimentSaver(f'Stitch{layer}F{feature}', save_id=True, disk_saver=True)

    canvas = torch.zeros((1, 3, image_size, image_size)).cuda()
    score = 0
    for i, p_size in enumerate(get_patches(image_size, patch_size)):
        stitcher = SniperStitcher(hook, p_size, image_size)
        for j, (x, y) in enumerate(tqdm(loader)):
            canvas, score = stitcher(canvas, score, x)
        saver.save(canvas, 'finished', i)
    saver.save(canvas, 'finished_all')


if __name__ == '__main__':
    main()
