import pdb

import torchvision.utils
from torch.utils.data import DataLoader
from datasets import weird_image_net
from model import model_library
from utils import exp_starter_pack
from torch import nn
import numpy as np
from runner import Evaluator
import torch


def roll_n(X, axis, n):
    f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None) for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None) for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)


def fftshift(X):
    real, imag = X.chunk(chunks=2, dim=-1)
    real, imag = real.squeeze(dim=-1), imag.squeeze(dim=-1)
    for dim in range(2, len(real.size())):
        real = roll_n(real, axis=dim, n=int(np.ceil(real.size(dim) / 2)))
        imag = roll_n(imag, axis=dim, n=int(np.ceil(imag.size(dim) / 2)))
    real, imag = real.unsqueeze(dim=-1), imag.unsqueeze(dim=-1)
    X = torch.cat((real, imag), dim=1)
    return torch.squeeze(X)


def ifftshift(X):
    real, imag = X.chunk(chunks=2, dim=-1)
    real, imag = real.squeeze(dim=-1), imag.squeeze(dim=-1)

    for dim in range(len(real.size()) - 1, 1, -1):
        real = roll_n(real, axis=dim, n=int(np.floor(real.size(dim) / 2)))
        imag = roll_n(imag, axis=dim, n=int(np.floor(imag.size(dim) / 2)))
    real, imag = real.unsqueeze(dim=-1), imag.unsqueeze(dim=-1)
    X = torch.cat((real, imag), dim=1)
    return torch.squeeze(X)


def h_pass(img: torch.tensor, filter_rate: float = 0.95):
    fft_img = torch.fft.fft(img)
    print(fft_img.shape)  # torch.Size([512, 512])
    fft_shift_img = fftshift(fft_img)
    print(fft_shift_img.shape)

    h, w = fft_shift_img.shape[:2]  # height and width
    cy, cx = int(h / 2), int(w / 2)  # centerness
    rh, rw = int(filter_rate * cy), int(filter_rate * cx)  # filter_size
    # the value of center pixel is zero.
    fft_shift_img[cy - rh:cy + rh, cx - rw:cx + rw] = 0
    ifft_shift_img = ifftshift(fft_shift_img)
    ifft_img = torch.fft.ifft(ifft_shift_img)
    return ifft_img


def l_pass(img: torch.tensor, filter_rate: float = 0.95) -> torch.tensor:
    raise NotImplementedError


class FreqPass(nn.Module):
    def __init__(self, high: bool = True, rate: float = 0.95):
        super().__init__()
        self.f, self.r = h_pass if high else l_pass, rate

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = torch.stack([torch.stack([self.f(c, self.r) for c in i]) for i in x])
        return x.real


def main():
    _ = exp_starter_pack()

    # network = 34
    # model, image_size, batch_size, _ = model_library[network]()
    # loader = DataLoader(weird_image_net.eval(), batch_size=6 * batch_size, num_workers=4, shuffle=False)
    # x, y = next(iter(loader))
    # x, y = x[:1], y[:1]
    # torchvision.utils.save_image(x, 'orig.png')
    # hp = FreqPass(True, 0.75)
    # nx = hp(x)
    # torchvision.utils.save_image(nx, 'h95.png')


if __name__ == '__main__':
    main()
