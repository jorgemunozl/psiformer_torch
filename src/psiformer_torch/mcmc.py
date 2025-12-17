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
        self._state: torch.Tensor | None = None

    def _init_state(self) -> torch.Tensor:
        B, n_e, dim = (
            self.config.batch_size,
            self.n_elec,
            self.config.dim,
        )
        return torch.randn(B, n_e, dim, device=self.device)

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

    def _mh_step(self, state: torch.Tensor) -> torch.Tensor:
        trial = self.generate_trial(state)
        accept_mask = self.accept_decline(trial, state)
        updated = torch.where(accept_mask[:, None, None], trial, state)
        return updated

    def _run_steps(self, state: torch.Tensor, steps: int) -> torch.Tensor:
        for _ in range(max(1, steps)):
            state = self._mh_step(state)
        return state

    @torch.inference_mode()
    def sampler(self) -> torch.Tensor:
        """
        Generate monte_carlo_size samples times batch.
        Keeps the latest accepted state to avoid re-thermalizing every call
        and performs several MH steps between stored samples to reduce
        duplicate configurations (which make the energy variance zero).
        """
        if self._state is None:
            self._state = self._init_state()
            self._state = self._run_steps(
                self._state, self.config.burn_in_steps
            ).detach()

        B, n_e, dim = self.config.batch_size, self.n_elec, self.config.dim
        samples_eq = torch.empty(
            self.config.monte_carlo_length, B, n_e, dim, device=self.device
        )

        for i in range(self.config.monte_carlo_length):
            self._state = self._run_steps(
                self._state,
                self.config.mh_steps_per_sample,
            ).detach()
            samples_eq[i] = self._state

        # Detach to make sure future updates don't backprop accidentally
        self._state = self._state.detach()
        return samples_eq
