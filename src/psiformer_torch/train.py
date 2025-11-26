import wandb
import logging


def train():
    wandb.init(project="psiformer_torch_example")
    logging.info("WandB initialized for project: psiformer_torch_example")
    
    # Training logic

    wandb.finish()


if __name__ == "__main__":
    train()