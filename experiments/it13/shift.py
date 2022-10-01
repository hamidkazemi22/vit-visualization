import torchvision.utils

from datasets import weird_image_net
from hooks.transformer.vit import SimpleViTGeLUHook
from model import model_library
from utils import exp_starter_pack
import torch
from tqdm import tqdm
import torchvision.transforms as tr
from torch import nn
import pdb


class Shift(nn.Module):
    def __init__(self, count_per_dim: int = 4, shift_per_dim: int = 4):
        super().__init__()
        self.c, self.add = count_per_dim, shift_per_dim

    def forward(self, x: torch.tensor) -> torch.tensor:
        assert x.dim() == 4
        rep = x.repeat(self.c * self.c, 1, 1, 1)
        for i in range(self.c):
            for j in range(self.c):
                real_ind, xi, yi = i * self.c + j, i * self.add, j * self.add
                rep[real_ind] = torch.roll(rep[real_ind], shifts=(xi, yi), dims=(1, 2))
        return rep


@torch.no_grad()
def main():
    exp_name, args, _ = exp_starter_pack()
    method = 'eval'
    dataset = weird_image_net.eval() if method == 'eval' else weird_image_net.train()
    indices = torch.load(f'{method}.pt')
    n_layer = 12
    indices = indices.view(n_layer, -1)
    n_feat = indices.shape[-1]
    before = torch.zeros(indices.shape).float().cuda()
    after = torch.zeros(indices.shape).float().cuda()

    model, image_size, _, _ = model_library[34]()
    hook = SimpleViTGeLUHook(model)

    shift = Shift(4, 4)
    images = shift(dataset[0][0].unsqueeze(0))
    torchvision.utils.save_image(images, 'shift.png')
    shift = shift.cuda()
    print(indices.shape)

    for l in range(n_layer):
        for f in tqdm(range(n_feat)):
            img, _ = dataset[indices[l, f]]
            img = img.cuda().unsqueeze(0)
            act = hook(shift(img)).view(-1, n_layer, n_feat)[:, l, f]
            before[l, f] = act[0]
            after[l, f] = act.mean()
            if f == 10:
                break
    torch.save(before, 'shift_before.pt')
    torch.save(after, 'shift_after.pt')
    pdb.set_trace()


if __name__ == '__main__':
    main()
