from dataclasses import dataclass, asdict
import os
import wandb

CHECKPOINT_DIR = "checkpoints/"
CHECKPOINT_NAME = "last_checkpoint_1.pth"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, CHECKPOINT_NAME)


@dataclass
class Model_Config():
    n_layer: int = 4
    n_head: int = 8
    n_embd: int = 128
    n_features: int = 4  # Hydrogen coordinates (x, y, z)
    n_out: int = 1
    n_determinants: int = 1
    n_electron_num: int = 2
    n_spin_down: int = 1
    n_spin_up: int = 1
    envelope_beta: float = 1.0  # exp(-beta * r) envelope strength


@dataclass
class Train_Config():
    train_steps: int = 100
    checkpoint_step: int = 10
    monte_carlo_length: int = 4000  # Num samples
    burn_in_steps: int = 1
    checkpoint_path: str = CHECKPOINT_PATH
    dim: int = 3  # Hydrogen coordinates only
    lr: float = 1e-3
    entity: str = "alvaro18ml-university-of-minnesota"
    project: str = "Psiformer"
    run_name: str = "Add envelope on the train script"

    def init_wandb(self):
        return wandb.init(
            entity=self.entity,
            project=self.project,
            name=self.run_name,
            config=asdict(self)
        )
