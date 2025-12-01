import torch
import matplotlib.pyplot as plt


x = torch.linspace(3, 10, 2)
z = torch.linspace(4, 5, 2)
x.requires_grad_(True)
y = x ** 2 + z

dy_dx = torch.autograd.grad(
    y, x, grad_outputs=torch.tensor([3.0,4.0]), create_graph=True, retain_graph=True
)

# d2y = torch.autograd.grad(
# dy_dx, x, grad_outputs=torch.ones_like(dy_dx)
# )

print(dy_dx)
