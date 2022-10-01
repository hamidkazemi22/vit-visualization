from torch.utils.data import DataLoader
from datasets import weird_image_net
from hooks.transformer.vit import SimpleViTGeLUHook
from model import model_library
from utils import exp_starter_pack
import torch
from tqdm import tqdm


@torch.no_grad()
def main():
    exp_name, args, _ = exp_starter_pack()
    network, batch_size = 34, 18
    dim_all = 12 * 768 * 4

    loader = DataLoader(weird_image_net.train(), batch_size=batch_size, num_workers=4, shuffle=False)
    model, image_size, _, _ = model_library[network]()
    hook = SimpleViTGeLUHook(model)
    value = torch.zeros(size=(dim_all,)).cuda() - 1.
    index = torch.zeros(size=(dim_all,)).long().cuda() - 1

    for i, (x, y) in enumerate(tqdm(loader)):
        x = x.cuda()
        act = hook(x)
        cur_v, cur_i = act.max(dim=0)
        update_i = value > cur_v
        value[update_i] = cur_v[update_i]
        index[update_i] = (cur_i + (i * batch_size))[update_i]
    torch.save(index, 'train.pt')


if __name__ == '__main__':
    main()
