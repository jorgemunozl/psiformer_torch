import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)  # better for derivatives

class ToyNet(nn.Module):
    """
    f: R^{n x n} -> R
    First operation: determinant(X), then an MLP on that scalar.
    """
    def __init__(self, n: int, hidden_dim: int = 16):
        super().__init__()
        self.n = n
        # MLP that takes a scalar det(X) as input and outputs a scalar
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (n, n) matrix (no batch, just a single sample)
        returns: scalar tensor
        """
        s, det_x = torch.slogdet(x)                # scalar
        det_x = det_x.view(1, 1)            # shape (1, 1) for Linear
        out = self.mlp(det_x)               # shape (1, 1)
        return out.squeeze()                # scalar


def laplacian_scalar_wrt_matrix(f: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Compute the Laplacian of scalar f wrt matrix x entries:
        Δ f = sum_{i,j} ∂² f / ∂ x_{ij}²

    f: scalar tensor = f(x)
    x: (n, n) with requires_grad=True
    """
    assert f.ndim == 0, "f must be a scalar tensor"
    assert x.requires_grad, "x must have requires_grad=True"

    # First gradient: g_ij = ∂ f / ∂ x_ij
    grad_x = torch.autograd.grad(f, x, create_graph=True)[0]  # shape (n, n)

    lap = torch.zeros((), dtype=x.dtype)
    n, m = x.shape

    # Laplacian = trace of Hessian = sum_{ij} ∂² f / ∂ x_ij²
    for i in range(n):
        for j in range(m):
            g_ij = grad_x[i, j]
            # second derivative wrt x_ij
            grad2_x = torch.autograd.grad(g_ij, x, retain_graph=True)[0]
            lap = lap + grad2_x[i, j]

    return lap


def main():
    n = 3  # size of the input matrix
    net = ToyNet(n)

    # Create a random input matrix and tell PyTorch it's a variable
    x = torch.randn(n, n, requires_grad=True)
    x = torch.tensor([[2.0, 4.0, 3.0],
                      [1.0, 2.0, 1.5],
                      [1.2, .3, .4]], requires_grad=True)
    # Forward pass: scalar output
    f = net(x)

    # Compute Laplacian wrt all entries of x
    lap = laplacian_scalar_wrt_matrix(f, x)

    print("Input matrix X =")
    print(x.detach().numpy())
    print("\nf(X) =", f.item())
    print("Laplacian Δ f(X) =", lap.item())


if __name__ == "__main__":
    main()
