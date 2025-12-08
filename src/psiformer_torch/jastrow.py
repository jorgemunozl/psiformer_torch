import torch
import torch.nn as nn


class Jastrow(nn.Module):
    """
    Docstring for Jastrow:
    """
    def __init__(self):
        super().__init__()
        self.alpha_anti = nn.Parameter(torch.rand(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the antisymmetric Jastrow factor for two electrons."""
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.size(1) < 2:
            raise ValueError("Jastrow requires at least two electrons.")

        # Helium
        diff = x[:, 0, :] - x[:, 1, :]
        dist = torch.linalg.norm(diff, dim=-1)
        fac = -0.5 * self.alpha_anti**2/(self.alpha_anti + dist)
        return fac
