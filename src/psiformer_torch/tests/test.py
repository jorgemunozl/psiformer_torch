import torch


# Stack and concat

x = torch.stack(
    [torch.tensor([2.0*_]) for _ in range(4)]
)

print(x)
print(x.mean())
