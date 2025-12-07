import os
import sys
import torch

try:
    from psiformer_torch.psiformer import Hamiltonian
except ImportError:
    # Fallback for running as a standalone script: add src/psiformer_torch to path.
    sys.path.append(os.path.abspath(os.path.dirname(__file__)))
    from psiformer import Hamiltonian


def log_psi_100(x: torch.Tensor) -> torch.Tensor:
    """
    Ground-state hydrogen (1s) log wavefunction.
    psi_100(r) = (1/sqrt(pi)) * exp(-r)  =>  log psi = -r - 0.5 log(pi)
    """
    r = torch.linalg.norm(x, dim=-1)
    return -r - 0.5 * torch.log(torch.tensor(torch.pi))


def local_energy_from_psiformer(x: torch.Tensor) -> torch.Tensor:
    # Hamiltonian expects a log-psi function and single-sample inputs.
    hamil = Hamiltonian(log_psi_100)
    energies = []
    for sample in x:
        energies.append(hamil.local_energy(sample))
    return torch.stack(energies)


def main():
    x = torch.randn(1024, 3)
    E_loc = local_energy_from_psiformer(x)
    print(f"Mean local energy: {E_loc.mean().item():.4f} Hartree (target -0.5)")


if __name__ == "__main__":
    main()
