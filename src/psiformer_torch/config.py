from dataclasses import dataclass, asdict
import os
import wandb


@dataclass
class Model_Config():
    n_layer: int = 4
    n_head: int = 32
    n_embd: int = 512
    n_features: int = 3  # Electron Coordinates (x, y, z)
    n_determinants: int = 1
    n_electron_num: int = 3
    n_spin_up: int = 2
    n_spin_down: int = 1
    nuclear_charge: int = 3  # Z for single nucleus (default: Lithium)


@dataclass
class Train_Config():
    train_steps: int = 1000
    checkpoint_step: int = 333
    batch_size: int = 2
    checkpoint_name: str = ""
    energy_batch_size: int = 128  # how many MCMC samples to score per GPU pass
    dim: int = 3  # Three spatial cordinates
    lr: float = 3e-4

    # Wandb
    entity: str = "alvaro18ml-university-of-minnesota"
    project: str = "Psiformer"
    run_name: str = "Train"
    wand_mode: str = "online"

    # MCMC
    monte_carlo_length: int = 1024  # Num samples
    burn_in_steps: int = 4
    step_size: float = 1.0
    mh_steps_per_sample: int = 32  # MH transitions between stored samples

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


Psiformer_Torch_Small_Model = Model_Config(
    n_layer=1,
    n_head=16,
    n_embd=64,
    n_determinants=1,
    n_electron_num=3,
    n_spin_up=2,
    n_spin_down=1,
    nuclear_charge=3  # Change depends on the atom
)

Psiformer_Torch_Small_Conf = Train_Config(
    # train
    batch_size=2,
    checkpoint_step=33,
    train_steps=301,
    energy_batch_size=2048,

    # MCMC
    monte_carlo_length=128,
    burn_in_steps=4,
    step_size=1.0,
    mh_steps_per_sample=128,
)


Psiformer_Torch_Large_Model = Model_Config(
    n_layer=2,
    n_head=32,
    n_embd=256,
    n_determinants=2,
    n_electron_num=3,
    n_spin_up=2,
    n_spin_down=1,
    nuclear_charge=3  # Change depends on the atom
)

Psiformer_Torch_Large_Conf = Train_Config(
    # train
    batch_size=2,
    checkpoint_step=50,
    train_steps=175,
    energy_batch_size=512,

    # MCMC
    monte_carlo_length=512,
    burn_in_steps=4,
    step_size=0.8,
    mh_steps_per_sample=128,
)


Psiformer_Torch_Debug_Model = Model_Config(
    n_layer=1,
    n_head=2,
    n_embd=4,
    n_determinants=1,
    n_electron_num=3,
    n_spin_up=2,
    n_spin_down=1,
    nuclear_charge=3  # Change depends on the atom
)

Psiformer_Torch_Debug_Conf = Train_Config(
    # train
    batch_size=1,
    checkpoint_step=2,
    train_steps=10,
    energy_batch_size=4,

    # MCMC
    monte_carlo_length=4,
    burn_in_steps=4,
    step_size=1.0,
    mh_steps_per_sample=4,
)

large_conf = (Psiformer_Torch_Large_Model, Psiformer_Torch_Large_Conf)
small_conf = (Psiformer_Torch_Small_Model, Psiformer_Torch_Small_Conf)
debug_conf = (Psiformer_Torch_Debug_Model, Psiformer_Torch_Debug_Conf)
