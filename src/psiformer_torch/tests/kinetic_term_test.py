import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
from torch.autograd.functional import jacobian, hessian


def vectorial_function(x: torch.Tensor) -> torch.Tensor:
    # Let's say that it has a shape (2,)
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
    return torch.exp(-x[0]**2-x[1]**2)


def analytic_laplacian(x: torch.Tensor) -> torch.Tensor:
    return -4 * torch.exp(-x[0]**2 - x[1]**2)*(1-x[0]-x[1])


def gradient(f: Callable, x: torch.Tensor):
    # Returns the gradient
    x = x.clone().detach().requires_grad_(True)
    y = f(x)
    grad = torch.autograd.grad(y, x, create_graph=True)[0]
    return grad.sum()


def laplacian(f: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    # Compute the Laplacian at a single point by tracing the Hessian
    x = x.clone().detach().requires_grad_(True)
    hess = hessian(f, x)  #  This would be efficient? I mean this computes 
    #  more derivatives that I want
    print(hess)
    return torch.einsum("ii->", hess)


def laplacian_on_grid(f: Callable[[torch.Tensor],
                      torch.Tensor], grid: torch.Tensor) -> torch.Tensor:
    """
    Evaluate the Laplacian of f over a grid of coordinates.
    grid has shape (dim, ...). We treat each spatial location independently.
    """
    dim = grid.shape[0]
    flat_points = grid.reshape(dim, -1).transpose(0, 1)
    results = torch.zeros(flat_points.shape[0], dtype=grid.dtype,
                          device=grid.device)
    for idx, point in enumerate(flat_points):
        results[idx] = laplacian(f, point)
    return results.reshape(grid.shape[1:])


def plot_scalar_field(f) -> None:
    if isinstance(f, torch.Tensor):
        f = f.detach().cpu().numpy()
    plt.figure()
    plt.imshow(f)
    plt.show()


def plot_scalar_fields(fields, titles, extent=(0, 1, 0, 1)) -> None:
    fig, axes = plt.subplots(1, len(fields), figsize=(10, 4), constrained_layout=True)
    for ax, field, title in zip(np.ravel(axes), fields, titles):
        data = field.detach().cpu().numpy() if torch.is_tensor(field) else np.asarray(field)
        im = ax.imshow(data, origin="lower", extent=extent)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.8)
    plt.show()


def main():
    x = torch.linspace(0, 1, 100)
    y = torch.linspace(0, 1, 100)
    value = torch.tensor([1.0, 2.0])
    print(laplacian(function, value))
#    X, Y = torch.meshgrid(x, y) #, indexing="ij")
#    XY = torch.stack([X, Y])
#    AL = analytic_laplacian(XY)
#    torchL = laplacian_on_grid(function, XY)


if __name__ == "__main__":
    main()
