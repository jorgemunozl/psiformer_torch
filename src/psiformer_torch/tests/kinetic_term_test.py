import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
from torch.autograd.functional import jacobian


def vectorial_function(x: torch.Tensor) -> torch.Tensor:
    # Let's say that it has a shape (2,)
    print(x.shape)
    y0 = torch.exp(-x[0]**2 - x[1]**2)
    y1 = torch.sin(x[0]) * torch.cos(x[1])
    return torch.stack([y0, y1])


def jacobian_manual(f: Callable, x: torch.Tensor) -> torch.Tensor:
    # Returns the jacobian matrix
    x = x.clone().detach().requires_grad_(True)
    y = f(x)
    jacobian_matrix = []
    for i in range(2):
        grad_i = torch.autograd.grad(y[i], x, create_graph=True)[0]
        jacobian_matrix.append(grad_i)
    return torch.stack(jacobian_matrix)  # Shape (output_dim, input_dim)


def function(x: torch.Tensor) -> torch.Tensor:
    # Let's say that it has a shape (2,)
    print(x.shape)
    return torch.exp(-x[0]**2-x[1]**2)


def analytic_laplacian(x: torch.Tensor) -> torch.Tensor:
    return -4 * torch.exp(-x[0]**2 - x[1]**2)*(1-x[0]-x[1])


def gradient(f: Callable, x: torch.Tensor):
    # Returns the gradient
    x = x.clone().detach().requires_grad_(True)
    y = f(x)
    grad = torch.autograd.grad(y, x, create_graph=True)[0]
    return grad.sum()


def laplacian(f, x):
    # x: shape (..., n), f(x): scalar
    x = x.clone().detach().requires_grad_(True)
    y = f(x)  # scalar tensor
    # first derivatives
    grad = torch.autograd.grad(y, x, create_graph=True)[0]  # same shape as x
    # second derivatives: grad outputs is a tensor of ones matching grad
    lap = 0.0
    for g in grad.view(-1):
        lap_term = torch.autograd.grad(g, x, retain_graph=True, allow_unused=False)[0]
        lap += lap_term.sum()
    return lap


def plot_scalar_field(f) -> None:
    plt.figure()
    plt.imshow(f)
    plt.show()


def main():
    x = torch.linspace(0, 1, 100)
    y = torch.linspace(0, 1, 100)
    X, Y = torch.meshgrid(x, y)
    XY = torch.stack([X, Y])
    AL = analytic_laplacian(XY)
    torchL = laplacian(function, XY)
    plot_scalar_field(torchL)


if __name__ == "__main__":
    main()
