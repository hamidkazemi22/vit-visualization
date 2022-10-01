import torch


def get_tv(t: torch.Tensor) -> torch.Tensor:
    x_wise = t[:, :, :, 1:] - t[:, :, :, :-1]
    y_wise = t[:, :, 1:, :] - t[:, :, :-1, :]
    diag_1 = t[:, :, 1:, 1:] - t[:, :, :-1, :-1]
    diag_2 = t[:, :, 1:, :-1] - t[:, :, :-1, 1:]
    return x_wise.norm(p=2, dim=(2, 3)).mean() + y_wise.norm(p=2, dim=(2, 3)).mean() + \
           diag_1.norm(p=2, dim=(2, 3)).mean() + diag_2.norm(p=2, dim=(2, 3)).mean()


def get_shadow_tv(t: torch.Tensor) -> torch.Tensor:
    rotated = torch.cat((t[:, -1:, :, :], t[:, :-1, :, :]), dim=1)
    return get_tv((t - rotated))
