import numpy as np
import torch


class Potential:
    def __init__(self, parameters):
        self.parameters = parameters


def kinetic_from_log(f, x):
    """Compute -1/2 \nabla^2 psi / psi from log|psi|."""
    grad = torch.tensor(f, x)
    hess = torch.tensor(f, x)
    laplacian = torch.trace(hess)
    kinetic = -0.5 * (laplacian + torch.dot(grad, grad))
    return kinetic


def operators(atoms,
              nelectrons,
              potential_epsilon=0.0):
    """Creates kinetic and potential operators of Hamiltonian in atomic units.

    Args:
    atoms: list of Atom objects for each atom in the system.
    nelectrons: number of electrons
    potential_epsilon: Epsilon used to smooth the divergence of the 1/r
      potential near the origin for algorithms with numerical stability issues.

    Returns:
    The functions that generates the kinetic and
    potential energy as a PyTorch op.
    """
    vnn = 0.0
    for i, atom_i in enumerate(atoms):
        for atom_j in atoms[i+1:]:
            qij = float(atom_i.charge * atom_j.charge)
            vnn += qij / np.linalg.norm(
                atom_i.coords_array - atom_j.coords_array
                )
