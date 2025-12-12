import logging
import torch
import matplotlib.pyplot as plt
import numpy as np

from psiformer import PsiFormer
from config import Model_Config, Train_Config
from psiformer_torch.train2 import train_config
from mcmc import MH
from hamiltonian import Hamiltonian


def load_checkpoint() -> PsiFormer:
    check_point = train_config.checkpoint_name
    print("Using checkpoint: ", check_point)
    path = train_config.init_checkpoint()
    print("Checkpoint path: ", path)

    loaded = torch.load(path, map_location=torch.device("cpu"))
    model = PsiFormer(Model_Config())
    if isinstance(loaded, dict) and "model_state_dict" in loaded:
        state_dict = loaded["model_state_dict"]
    else:
        # Backwards compatibility for checkpoints saved as raw state_dict
        state_dict = loaded

    model.load_state_dict(state_dict)
    model.eval()

    return model


def compute_energy(monte_carlo, burn_in, step_size) -> float:
    model = load_checkpoint()
    eval_config = Train_Config(
       monte_carlo_length=monte_carlo,
       burn_in_steps=burn_in,
       step_size=step_size
    )
    mh = MH(model, eval_config, model.config.n_electron_num)
    hamilton = Hamiltonian(model)
    samples = mh.sampler()

    # Local Energies:
    local_energies = torch.stack(
        [hamilton.local_energy(s) for s in samples]
    )

    # Energy Local Expection
    E_mean = local_energies.mean().detach()
    return float(E_mean)


def monte_carlo_step_dense(start, end, step):
    """
    In each step we init a completely new Markov Chain.
    """
    montecarlos = np.arange(start, end, step)
    burn_in = 10
    step_size = 2
    data = []
    print("Number of steps: ", (end-start)/step)
    for i, montecarlo in enumerate(montecarlos):
        print("Computing step", montecarlo)
        E = compute_energy(montecarlo, burn_in, step_size)
        data.append(E)

    plt.plot(montecarlos, data)
    plt.xlabel("Monte Carlo Length")
    plt.ylabel("Energy (a.u)")
    plt.savefig("Monte Carlo Variation")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(nam)s %(levelname)s: %(message)s"
                        )
    logger = logging.getLogger("Evaluation")
    logger.info("Starting Evaluation")
    monte_carlo_step_dense(1000, 6000, 200)
