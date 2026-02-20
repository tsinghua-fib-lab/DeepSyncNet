import torch
from torch import nn


def normal_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight, mean=0, std=0.01)
        torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv1d):
        torch.nn.init.normal_(m.weight, mean=0, std=0.01)
        torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'bias' in name:
                torch.nn.init.zeros_(param)
            elif 'weight' in name:
                torch.nn.init.normal_(param, mean=0, std=0.01)