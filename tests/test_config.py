import torch


def gaussian(size):
    return torch.randn(size)


print(gaussian([3, 4]))