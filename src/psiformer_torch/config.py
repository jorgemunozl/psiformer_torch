from dataclasses import dataclass, asdict
import os
import wandb


@dataclass
class Model_Config():
<<<<<<< HEAD
    n_layer: int = 6
=======
    n_layer: int = 4
>>>>>>> 189f7cffd92618f4e7eaf624248ef4fa11fb375a
    n_head: int = 16
    n_embd: int = 256
    n_features: int = 3  # Electron Coordinates (x, y, z)
    n_determinants: int = 4
    n_electron_num: int = 3
    n_spin_up: int = 2
    n_spin_down: int = 1
    nuclear_charge: int = 3  # Z for single nucleus (default: Lithium)


@dataclass
class Train_Config():
    train_steps: int = 1000
<<<<<<< HEAD
    checkpoint_step: int = 600
    batch_size: int = 2
    checkpoint_name: str = ""
    energy_batch_size: int = 5128  # how many MCMC samples to score per GPU pass
=======
    checkpoint_step: int = 300
    batch_size: int = 4
    checkpoint_name: str = ""
    energy_batch_size: int = 256  # how many MCMC samples to score per GPU pass
>>>>>>> 189f7cffd92618f4e7eaf624248ef4fa11fb375a

    dim: int = 3  # Three spatial cordinates
    lr: float = 1e-3

    # Wandb
    entity: str = "alvaro18ml-university-of-minnesota"
    project: str = "Psiformer"
    run_name: str = "Train"
    wand_mode: str = "online"

    # MCMC
<<<<<<< HEAD
    monte_carlo_length: int = 1024  # Num samples
    burn_in_steps: int = 4
    step_size: float = 1.0
=======
    monte_carlo_length: int = 1025  # Num samples stored per update
    burn_in_steps: int = 64
    mh_steps_per_sample: int = 32  # MH transitions between stored samples
    step_size: float = .8
>>>>>>> 189f7cffd92618f4e7eaf624248ef4fa11fb375a

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
            mode=self.wand_mode if self.wand_mode in options else "online",
        )
