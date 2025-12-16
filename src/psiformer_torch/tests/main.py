import torch

A = torch.rand(4, 4)
print(A)
B = A.reshape(-1)
print(B)