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


class Rotation(nn.Module):
    def __init__(self, count: int = 360 // 30):
        super().__init__()
        self.deg = 360. / count
        self.rotates = [tr.RandomRotation((i, i + 1)) for i in range(0, 360, int(self.deg))]

    def forward(self, x: torch.tensor) -> torch.tensor:
        assert x.dim() == 4
        rep = x.repeat(len(self.rotates), 1, 1, 1)
        for i, rot in enumerate(self.rotates):
            rep[i] = rot(rep[i])
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

    rotate = Rotation(18)
    images = rotate(dataset[0][0].unsqueeze(0))
    torchvision.utils.save_image(images, 'rotation.png')
    rotate = rotate.cuda()
    print(indices.shape)

    for l in range(n_layer):
        for f in tqdm(range(n_feat)):
            img, _ = dataset[indices[l, f]]
            img = img.cuda().unsqueeze(0)
            act = hook(rotate(img)).view(-1, n_layer, n_feat)[:, l, f]
            before[l, f] = act[0]
            after[l, f] = act.mean()
    torch.save(before, 'rotate_before.pt')
    torch.save(after, 'rotate_after.pt')
    pdb.set_trace()


if __name__ == '__main__':
    main()
