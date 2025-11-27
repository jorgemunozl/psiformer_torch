from hamiltonian import Hamiltonian, Potential
import wandb
import logging


# Compute Local Energy


def train():
    wandb.init(project="psiformer_torch_example")
    logging.info("WandB initialized for project: psiformer_torch_example")
    wandb.finish()


if __name__ == "__main__":
    train()
