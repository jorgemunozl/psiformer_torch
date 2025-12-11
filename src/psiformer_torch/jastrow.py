import torch
import torch.nn as nn


class Jastrow(nn.Module):
    """
    Docstring for Jastrow:
    """
    def __init__(self, spin_up: int, spin_down: int):
        super().__init__()
        self.alpha_anti = nn.Parameter(torch.rand(1))
        self.alpha_par = nn.Parameter(torch.rand(1))
        self.spin_up = spin_up
        self.spin_down = spin_down

    @staticmethod
    def _same_spin_sum(position: torch.Tensor,
                       coeff: float, alpha: torch.Tensor) -> torch.Tensor:
        """
        position: (B, n, 3) n: up or down
        """
        batch_size, n, _ = position.shape
        if n < 2:
            return position.new_zeros(batch_size)

        # Pairwise distance with broadcasting
        diff = position[:, :, None, :] - position[:, None, :, :]
        # Use a soft norm to avoid NaNs in higher-order derivatives when
        # electrons coincide (norm gradient is undefined at zero).
        dists = torch.sqrt(diff.pow(2).sum(dim=-1) + 1e-12)

        i, j = torch.triu_indices(n, n, offset=1, device=position.device)
        pair_dists = dists[:, i, j]  # (B, num_pairs)
        terms = coeff * alpha.pow(2) / (alpha+pair_dists)
        return terms.sum(dim=1)

    @staticmethod
    def _diff_spin_sum(up: torch.Tensor, down: torch.Tensor,
                       coeff: float, alpha: torch.Tensor) -> torch.Tensor:
        """
        up: (B, n_up, 3)
        down: (B, n_down, 3)
        returns (B,)
        """
        batch_size, n_up, _ = up.shape
        _, n_down, _ = down.shape

        if n_up == 0 or n_down == 0:
            return up.new_zeros(batch_size)

        diff = up[:, :, None, :] - down[:, None, :, :]
        dists = torch.sqrt(diff.pow(2).sum(dim=-1) + 1e-12)

        pair_dists = dists.reshape(batch_size, -1)
        terms = coeff * alpha.pow(2) / (alpha + pair_dists)
        return terms.sum(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the antisymmetric Jastrow factor for n electrons.
        x : (B, n_elec, 3)
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.size(1) < 2:
            raise ValueError("Jastrow requires at least two electrons.")

        # Any atom
        up = x[:, :self.spin_up, :]  # (B, up, 3)
        down = x[:, self.spin_up:self.spin_down + self.spin_up, :]  # (B, d, 3)

        same_up = self._same_spin_sum(up, -0.25, self.alpha_par)
        same_down = self._same_spin_sum(down, -0.25, self.alpha_par)

        diff_spin = self._diff_spin_sum(up, down,
                                        coeff=-0.5, alpha=self.alpha_anti)

        return same_up + same_down + diff_spin
