import torch

x = torch.rand(2,3)
y = torch.rand(2,3)

if ((x<y)[1,2]):
    print("asd")