from torch import nn as nn
import torch
import random


class Gray4D(nn.Module):
    def __init__(self, n_channels: int = 3):
        super().__init__()
        self.n = n_channels

    def forward(self, x: torch.tensor) -> torch.tensor:
        shape = tuple([1] * (4 - x.dim())) + x.shape
        return x.view(shape).repeat(1)


class Layered(nn.Module):
    def __init__(self, x: torch.tensor):
        super().__init__()
        self.x = x

    def forward(self, x: torch.tensor) -> torch.tensor:
        return x + self.x


class Jitter(nn.Module):
    def __init__(self, lim: int = 32):
        super().__init__()
        self.lim = lim

    def forward(self, x: torch.tensor) -> torch.tensor:
        off1 = random.randint(-self.lim, self.lim)
        off2 = random.randint(-self.lim, self.lim)
        return torch.roll(x, shifts=(off1, off2), dims=(2, 3))


class ColorJitter(nn.Module):
    def __init__(self, batch_size: int, shuffle_every: bool = False, mean: float = 1., std: float = 1.):
        super().__init__()
        self.batch_size, self.mean_p, self.std_p = batch_size, mean, std
        self.mean = self.std = None
        self.shuffle()
        self.shuffle_every = shuffle_every

    def shuffle(self):
        self.mean = (torch.rand((self.batch_size, 3, 1, 1,)).cuda() - 0.5) * 2 * self.mean_p
        self.std = ((torch.rand((self.batch_size, 3, 1, 1,)).cuda() - 0.5) * 2 * self.std_p).exp()

    def forward(self, img: torch.tensor) -> torch.tensor:
        if self.shuffle_every:
            self.shuffle()
        return (img - self.mean) / self.std


class GaussianNoise(nn.Module):
    def __init__(self, batch_size: int, shuffle_every: bool = False, std: float = 1., max_iter: int = 400):
        super().__init__()
        self.batch_size, self.std_p, self.max_iter = batch_size, std, max_iter
        self.std = None
        self.rem = max_iter - 1
        self.shuffle()
        self.shuffle_every = shuffle_every

    def shuffle(self):
        self.std = torch.randn(self.batch_size, 3, 1, 1).cuda() * self.rem * self.std_p / self.max_iter
        self.rem = (self.rem - 1 + self.max_iter) % self.max_iter

    def forward(self, img: torch.tensor) -> torch.tensor:
        if self.shuffle_every:
            self.shuffle()
        return img + self.std


class ColorJitterR(ColorJitter):
    def forward(self, img: torch.tensor) -> torch.tensor:
        if self.shuffle_every:
            self.shuffle()
        return (img * self.std) + self.mean


class Centering(nn.Module):
    def __init__(self, size: int, std: float):
        super().__init__()
        self.size = size
        self.std = std

    def forward(self, img: torch.tensor) -> torch.tensor:
        pert = (torch.rand(2) * 2 - 1) * self.std
        w, h = img.shape[-2:]
        x = (pert[0] + w // 2 - self.size // 2).long().clamp(min=0, max=w - self.size)
        y = (pert[1] + h // 2 - self.size // 2).long().clamp(min=0, max=h - self.size)
        return img[:, :, x:x + self.size, y:y + self.size]


class Zoom(nn.Module):
    def __init__(self, out_size: int = 384):
        super().__init__()
        self.up = torch.nn.Upsample(size=(out_size, out_size), mode='bilinear', align_corners=False).cuda()

    def forward(self, img: torch.tensor) -> torch.tensor:
        return self.up(img)


class Tile(nn.Module):
    def __init__(self, rep: int = 384 // 16):
        super().__init__()
        self.rep = rep

    def forward(self, x: torch.tensor) -> torch.tensor:
        dim = x.dim()
        if dim < 3:
            raise NotImplementedError
        elif dim == 3:
            x.unsqueeze(0)
        final_shape = x.shape[:2] + (x.shape[2] * self.rep, x.shape[3] * self.rep)
        return x.unsqueeze(2).unsqueeze(4).repeat(1, 1, self.rep, 1, self.rep, 1).view(final_shape)


class RepeatBatch(nn.Module):
    def __init__(self, repeat: int = 32):
        super().__init__()
        self.size = repeat

    def forward(self, img: torch.tensor):
        return img.repeat(self.size, 1, 1, 1)


class MaskBatch(nn.Module):
    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.other(x[:self.count] if self.count > 0 else x)

    def __init__(self, count: int = -1):
        super().__init__()
        self.count = count


class Flip(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: torch.tensor) -> torch.tensor:
        return torch.flip(x, dims=(3,)) if random.random() < self.p else x
