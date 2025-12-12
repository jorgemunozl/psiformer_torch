from dataclasses import dataclass, asdict
import os
import wandb


@dataclass
class Model_Config():
    n_layer: int = 4
    n_head: int = 16
    n_embd: int = 256
    n_features: int = 3  # Electron Coordinates (x, y, z)
    n_determinants: int = 4
    n_electron_num: int = 2
    n_spin_down: int = 1
    n_spin_up: int = 1
    nuclear_charge = 2


@dataclass
class Train_Config():
    train_steps: int = 500
    checkpoint_step: int = 300
    batch_size: int = 4
    checkpoint_name: str = ""

    dim: int = 3  # Three spatial cordinates
    lr: float = 0.5e-3

    # Wandb
    entity: str = "alvaro18ml-university-of-minnesota"
    project: str = "Psiformer"
    run_name: str = "Train"
    wand_mode: str = "online"

    # MCMC
    monte_carlo_length: int = 1028  # Num samples
    burn_in_steps: int = 16
    step_size: float = 0.8
    energy_batch_size: int = 256

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
        options = ("online", "offline", "disabled")
        return wandb.init(
            entity=self.entity,
            project=self.project,
            name=self.run_name,
            config=full,
            mode=self.wand_mode if self.wand_mode in options else "online"
        )
