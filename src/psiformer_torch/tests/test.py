import torch


A = torch.rand(3, 2, 2, 3)
sign, C = (torch.linalg.slogdet(A))
b = torch.logsumexp
print(C.shape)
print(sign.shape)
