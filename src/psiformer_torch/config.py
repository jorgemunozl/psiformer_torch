from dataclasses import dataclass, asdict
import os
import wandb


@dataclass
class Model_Config():
    n_layer: int = 2
    n_head: int = 16
    n_embd: int = 256
    n_features: int = 3  # Hydrogen coordinates (x, y, z)
    n_determinants: int = 3
    n_electron_num: int = 2
    n_spin_down: int = 1
    n_spin_up: int = 1


@dataclass
class Train_Config():
    train_steps: int = 150
    checkpoint_step: int = 49
    batch_size: int = 2
    checkpoint_name: str = ""

    dim: int = 3  # Three spatial cordinates
    lr: float = 0.5e-3

    # Wandb
    entity: str = "alvaro18ml-university-of-minnesota"
    project: str = "Psiformer"
    run_name: str = ""

    # MCMC
    monte_carlo_length: int = 6000  # Num samples
    burn_in_steps: int = 100
    step_size: float = 1.0

    def init_checkpoint(self):
        CHECKPOINT_DIR = "checkpoints/"
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        check_path = os.path.join(CHECKPOINT_DIR, self.checkpoint_name)
        return check_path

    def check_name(self):
        """
        Check if name ends with .pth
        """
        if self.checkpoint_name.endswith(".pth"):
            return self.checkpoint_name
        else:
            return self.checkpoint_name + ".pth"

    def init_wandb(self, model_config: Model_Config):
        """
        Config and model configuration to syncz on wandb
        """
        model_dict = asdict(model_config)
        train_config = asdict(self)
        full = model_dict | train_config

        return wandb.init(
            entity=self.entity,
            project=self.project,
            name=self.run_name,
            config=full
        )
