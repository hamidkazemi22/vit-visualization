import pdb

from torch.utils.data import DataLoader
from datasets import weird_image_net, image_net
from hooks.transformer.vit import SimpleViTGeLUHook, SimpleClipGeLUHook
from model import model_library
from utils import exp_starter_pack
import torch
from tqdm import tqdm


@torch.no_grad()
def main():
    exp_name, args, _ = exp_starter_pack()
    network, batch_size = 99, 24
    dim_all = 12 * 768 * 4
    layer, feature = args.layer, args.feature
    mod = 'eval' if args.grid == 1.0 else 'train'
    data = image_net.eval() if mod == 'eval' else image_net.train()

    loader = DataLoader(data, batch_size=batch_size, num_workers=4, shuffle=False)
    model, image_size, _, _ = model_library[network]()
    hook = SimpleClipGeLUHook(model)
    indices = []

    for i, (x, y) in enumerate(tqdm(loader)):
        x = x.cuda()
        act = hook(x)
        act = act.view(act.shape[0], 12, -1)[:, layer, feature]
        cur_i = torch.arange(act.shape[0]) + (i * batch_size)
        cur = act.cpu().numpy().tolist()
        cur_arr = [(a, ind) for a, ind in zip(cur, cur_i)]
        indices = indices + cur_arr
        indices = sorted(indices, reverse=True)[:20]
        print(indices[0][0])
        if i % 100 == 0:
            print(f'{i}/{len(loader)} from {mod}')
    torch.save(indices, f'{mod}_{layer}_{feature}.pt')
    pdb.set_trace()


if __name__ == '__main__':
    main()
