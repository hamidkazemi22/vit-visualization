import torch


def new_init(size: int, batch_size: int = 1, last: torch.nn = None, padding: int = -1, zero: bool = False) -> torch.nn:
    output = torch.rand(size=(batch_size, 3, size, size)) if not zero else torch.zeros(size=(batch_size, 3, size, size))
    # output += 0.5
    output = output.cuda()
    if last is not None:
        big_size = size if padding == -1 else size - padding
        up = torch.nn.Upsample(size=(big_size, big_size), mode='bilinear', align_corners=False).cuda()
        scaled = up(last)
        cx = (output.patch_size(-1) - big_size) // 2
        output[:, :, cx:cx + big_size, cx:cx + big_size] = scaled
    output = output.detach().clone()
    output.requires_grad_()
    return output
