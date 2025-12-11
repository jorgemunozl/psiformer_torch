import torch
from typing import Callable
from config import Train_Config
from psiformer import get_device


class MH():
    """
    Implementation for the Metropolis Hasting (MH) algorithm
    using a gaussian kernel. Returns a list a samples from
    the target distribution.
    We work with the log form!.
    """
    def __init__(self, target: Callable[[torch.Tensor], torch.Tensor],
                 config: Train_Config, n_elec: int,
                 device: torch.device | None = None):
        self.target = target
        self.config = config
        self.n_elec = n_elec
        self.device = device or get_device()

    def generate_trial(self, state: torch.Tensor) -> torch.Tensor:
        # state: (B, n_elec_, 3)
        return state + self.config.step_size * torch.randn_like(state)

    def accept_decline(self, trial: torch.Tensor,
                       current_state: torch.Tensor) -> torch. Tensor:
        # Sampling does not need gradients; keep it detached from autograd.
        with torch.no_grad():
            # alpha: (B,)
            alpha = 2*(self.target(trial) - self.target(current_state))
            log_accept = torch.min(alpha, torch.zeros_like(alpha))
            log_u = torch.log(torch.rand_like(alpha))
        return log_u < log_accept  # (B,)

    @torch.inference_mode()
    def sampler(self) -> torch.Tensor:
        """
        Generate monte_carlo_size samples times batch
        """
        # Thermalization
        # x: (B, n_elec, 3)
        B, n_e, dim = self.config.batch_size, self.n_elec, self.config.dim
        x = torch.randn(B, n_e, dim, device=self.device)

        for _ in range(self.config.burn_in_steps):
            trial = self.generate_trial(x)  # (B, n_e, 3)
            accept_mask = self.accept_decline(trial, x)  # (B,)
            x = torch.where(accept_mask[:, None, None], trial, x)  # Broadcast

        samples_eq = torch.zeros(
            self.config.monte_carlo_length, B, n_e, dim, device=self.device
        )
        samples_eq[0] = x

        for i in range(1, self.config.monte_carlo_length):
            trial = self.generate_trial(x)
            acc = self.accept_decline(trial, x)
            # x: (B, n_e, dim)
            x = torch.where(acc[:, None, None], trial, x)  # Broad
            samples_eq[i] = x

        return samples_eq
