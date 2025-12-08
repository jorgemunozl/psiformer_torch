import torch
from psiformer import get_device, PsiFormer
from config import Train_Config, Model_Config
import torch.optim as optim
from mcmc import MH
import logging
from hamiltonian import Hamiltonian


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(nam)s %(levelname)s: %(message)s"
                    )
logger = logging.getLogger("Beginning")
logger.info("Starting")


class Trainer():
    def __init__(self, model, config: Train_Config):
        self.model = model.to(get_device())
        self.config = config
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.device = get_device()

    def log_psi(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        n_elec = self.model.config.n_electron_num
        expected_dim = n_elec * 3
        if x.numel() != expected_dim:
            raise ValueError(
                f"Sample has {x.numel()} entries, expected {expected_dim} "
                "(n_electron_num * 3 coordinates)."
            )
        x = x.view(1, n_elec, 3)
        return self.model(x).squeeze(0)

    def save_checkpoint(self, step):
        if step % self.config.checkpoint_step == 0:
            # Check if father directory checkpoint_path exist.
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "step": step,
                },
                self.config.checkpoint_path,
            )
            print(f"Saved checkpoint at step {step}")

    def train(self):
        """
        Create samples, using those computes the E_mean, E,
        Then using model carlo you can compute the derivative of the loss.
        Important the detach.
        """
        mh = MH(self.log_psi, self.config.burn_in_steps,
                self.config.monte_carlo_length, self.config.dim, step_size=1.0)
        hamilton = Hamiltonian(self.log_psi)
        run = self.config.init_wandb()
        for step in range(self.config.train_steps):
            # Sampling
            samples = mh.sampler().to(self.device)

            # Local Energies
            local_energies = torch.stack(
                [hamilton.local_energy(s) for s in samples]
            )

            # Log Psi
            log_psi_vals = torch.stack([self.log_psi(s) for s in samples])

            # Energy Local Expection
            E_mean = local_energies.mean().detach()

            # Derivative of the Loss
            loss = 2*((local_energies.detach() - E_mean) * log_psi_vals).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.save_checkpoint(step)
            # Logging
            logger.info(f"Step {step}: E_mean = {E_mean.item():.6f}")
            logger.info(f"Loss = {loss.item():.6f}")
            run.log({"Energy": E_mean, "loss": loss})

        run.finish()


def train():
    device = get_device()
    print(f"Using {device}")

    # Model
    model_config = Model_Config()
    model = PsiFormer(model_config)

    # Train
    train_config = Train_Config(run_name="Envelope inside the model")

    # Keep dim consistent with number of electron coordinates (n_elec * 3)
    train_config.dim = model_config.n_electron_num * 3
    trainer = Trainer(model, train_config)

    # train the model
    trainer.train()


train()
