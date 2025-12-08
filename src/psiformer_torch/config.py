from dataclasses import dataclass, asdict
import os
import wandb


@dataclass
class Model_Config():
    n_layer: int = 4
    n_head: int = 8
    n_embd: int = 128
    n_features: int = 3  # Hydrogen coordinates (x, y, z)
    n_determinants: int = 1
    n_electron_num: int = 2
    n_spin_down: int = 1
    n_spin_up: int = 1
    envelope_beta: float = 1.0  # exp(-beta * r) envelope strength


@dataclass
class Train_Config():
    train_steps: int = 100
    checkpoint_step: int = 10
    batch_size: int = 1
    checkpoint_name: str = ""

    dim: int = 3  # Three spatial cordinates
    lr: float = 1e-3

    # Wandb
    entity: str = "alvaro18ml-university-of-minnesota"
    project: str = "Psiformer"
    run_name: str = "Add envelope on the train script"

    # MCMC
    monte_carlo_length: int = 1000  # Num samples
    burn_in_steps: int = 1
    step_size: float = 1.0

    def init_checkpoint(self):
        CHECKPOINT_DIR = "checkpoints/"
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        check_path = os.path.join(CHECKPOINT_DIR, self.checkpoint_name)
        return check_path

    def init_wandb(self):
        return wandb.init(
            entity=self.entity,
            project=self.project,
            name=self.run_name,
            config=asdict(self)
        )
