import torch
from typing import Callable


class MH():
    """
    Implementation for the Metropolis Hasting (MH) algorithm
    using a gaussian kernel. Returns a list a samples from
    the target distribution.
    We work with the log form. !
    """
    def __init__(self, target: Callable[[torch.Tensor], torch.Tensor],
                 eq_steps: int,
                 num_samples: int,
                 dim: int,
                 batch_size: int,
                 step_size: float = 1.0,
                 ):
        self.target = target
        self.eq_steps = eq_steps
        self.num_samples = num_samples
        self.dim = dim
        self.step_size = step_size
        self.batch_size = batch_size

    def generate_trial(self, state: torch.Tensor) -> torch.Tensor:
        return state + self.step_size * torch.randn_like(state)

    def accept_decline(self, trial: torch.Tensor,
                       current_state: torch.Tensor) -> bool:
        # Sampling does not need gradients; keep it detached from autograd.
        with torch.no_grad():
            alpha = 2*(self.target(trial) - self.target(current_state))
        if torch.rand(()) < torch.exp(torch.minimum(alpha, torch.tensor(0.0))):
            return True
        return False

    @torch.no_grad()
    def sampler(self) -> torch.Tensor:
        """
        Generate monte_carlo_size samples times batch
        """
        # Thermalization
        x = torch.randn(self.batch_size, self.num_samples, 3)

        # Here the first configuration is sampled from
        for _ in range(self.eq_steps):
            trial = self.generate_trial(x)
            if self.accept_decline(trial, x):
                x = trial

        # Sampling
        samples = torch.zeros(self.num_samples, self.dim)
        samples[0] = x

        for i in range(1, self.num_samples):
            trial = self.generate_trial(x)
            if self.accept_decline(trial, x):
                x = trial
            samples[i] = x

        return samples
