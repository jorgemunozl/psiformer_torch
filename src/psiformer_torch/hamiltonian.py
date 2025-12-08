import torch
from torch.autograd import grad
from typing import Callable


class Potential():
    """
    For the Helium atom, the only nucleous is fixed at (0,0,0).
    Broadcast.
    """
    def __init__(self, coords: torch.Tensor, Z: int = 2):
        # coords: (B, n_elec, 3)
        self.coords = coords
        self.Z = Z

    def potential(self) -> torch.Tensor:
        eps = 1e-5
        r_i = torch.linalg.norm(self.coords, dim=-1)  # (B, n_elec)
        nuc_term = -self.Z*(1/(r_i+eps)).sum(dim=-1)
        diff = self.coords[:, 0, :] - self.coords[:, 1, :]
        r_12 = torch.linalg.norm(diff)
        e_e_term = 1/(r_12 + eps)

        # (B, )
        return nuc_term + e_e_term


class Hamiltonian():
    def __init__(self, log_psi_fn: Callable[[torch.Tensor], torch.Tensor],
                 n_elec: int = 2, Z: int = 2
                 ):
        self.log_psi_fn = log_psi_fn
        self.n_elec = n_elec
        self.Z = Z

    def local_energy(self, sample: torch.Tensor) -> torch.Tensor:
        # sample : (B, n_elec, 3)
        V = Potential(sample, self.Z).potential()
        g = self.grad_log_psi(sample)
        lap = self.laplacian_log_psi(sample)
        # kinetic : (B, )
        kinetic = -0.5 * (lap + (g * g).sum())
        return kinetic + V

    def grad_log_psi(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gradient of log psi with graph retained for higher order derivatives.
        x: (B, n_elec, 3)
        """
        x_req = x.clone().detach().requires_grad_(True)
        y = self.log_psi_fn(x_req)  # y: (B,)
        (g,) = grad(
            y, x_req, grad_outputs=torch.ones_like(y),
            create_graph=True
        )
        return g

    def laplacian_log_psi(self, x: torch.Tensor) -> torch.Tensor:
        """
        Laplacian of log psi via second derivatives of each dimension.
        x: (B, n_elec, 3)
        """
        x_req = x.clone().detach().requires_grad_(True)
        y = self.log_psi_fn(x_req)  # (B,)
        (g,) = grad(y, x_req, grad_outputs=torch.ones_like(y),
                    create_graph=True, retain_graph=True)

        second_terms = []
        g_flat = g.reshape(g.shape[0], -1)
        for b in range(g_flat.shape[0]):
            lap_b = 0.0
            for j in range(g_flat.shape[1]):
                second = grad(g_flat[b, j], x_req, retain_graph=True)[0]
                lap_b = lap_b + second.reshape(g_flat.shape[0], -1)[b, j]
            second_terms.append(lap_b)
        return torch.stack(second_terms)
