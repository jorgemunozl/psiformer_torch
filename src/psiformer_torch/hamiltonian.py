import torch
from torch.autograd import grad
from typing import Callable


class Potential():
    """
    For the hidrogen atom, the only nucleous is fixed at (0,0,0).
    Broadcast.
    """
    def __init__(self, r_e: torch.Tensor):
        # Compute the potential between the hidrogen proton and electron
        self.r_e = r_e[..., :3]  # only spatial coords

    def potential(self) -> torch.Tensor:
        eps = 1e-12
        r = torch.linalg.norm(self.r_e, dim=-1)
        return -1/(r+eps)


class Hamiltonian():
    def __init__(self, log_psi_fn: Callable[[torch.Tensor], torch.Tensor]):
        self.log_psi_fn = log_psi_fn

    def local_energy(self, sample: torch.Tensor) -> torch.Tensor:
        # Hydrogen: potential from proton/electron distance
        V = Potential(sample).potential()
        g = self.grad_log_psi(sample)
        lap = self.laplacian_log_psi(sample)
        kinetic = -0.5 * (lap + (g * g).sum())
        return kinetic + V

    def grad_log_psi(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gradient of log psi with graph retained for higher order derivatives.
        """
        x_req = x.clone().detach().requires_grad_(True)
        y = self.log_psi_fn(x_req)
        (g,) = grad(y, x_req, create_graph=True)
        return g

    def laplacian_log_psi(self, x: torch.Tensor) -> torch.Tensor:
        """
        Laplacian of log psi via second derivatives of each dimension.
        """
        x_req = x.clone().detach().requires_grad_(True)
        y = self.log_psi_fn(x_req)
        (g,) = grad(y, x_req, create_graph=True, retain_graph=True)

        second_terms = []
        for i in range(x_req.numel()):
            (g_i,) = grad(g[i], x_req, retain_graph=True)
            second_terms.append(g_i[i])
        return torch.stack(second_terms).sum()
