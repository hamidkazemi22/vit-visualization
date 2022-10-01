import torch
import os


def save_model(model: torch.nn.Module, exp_name: str, epoch: int, root: str = 'checkpoints'):
    par = os.path.join(root, exp_name)
    os.makedirs(par, exist_ok=True)
    path = os.path.join(par, '{}.pt'.format(epoch))
    torch.save(model.state_dict(), path)


def load_model(model: torch.nn.Module, checkpoint: str, root='checkpoints'):
    path = os.path.join(root, '{}.pt'.format(checkpoint))
    model.load_state_dict(torch.load(path))
    model.eval()
