import torch
import time
from psiformer import get_device, PsiFormer
from config import Train_Config, Model_Config
import torch.optim as optim
from mcmc import MH
import logging
from hamiltonian import Hamiltonian

#torch.autograd.set_detect_anomaly(True)
torch.set_float32_matmul_precision("high")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s: %(message)s"
                    )
logger = logging.getLogger("Beginning")
logger.info("Starting")


class Trainer():
    def __init__(self, model: PsiFormer, config: Train_Config):
        self.model = model.to(get_device())
        self.config = config
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)
        self.device = get_device()
        self.mh = MH(
            self.log_psi,
            self.config,
            self.model.config.n_electron_num,
            device=self.device,
        )
        self.hamilton = Hamiltonian(
            self.log_psi,
            n_elec=self.model.config.n_electron_num,
            Z=self.model.config.nuclear_charge,
        )
        print(model.config.n_electron_num)

    def log_psi(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_elec, 3)
        if x.device != self.device:
            x = x.to(self.device)
        return self.model(x)

    def _batched_energy_eval(
        self, samples: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Score MCMC samples in larger batches to keep the GPU busy.
        samples: (mc_steps, B, n_elec, 3)
        """
        flat = samples.reshape(-1, samples.size(-2), samples.size(-1))

        logpsis: list[torch.Tensor] = []
        local_es: list[torch.Tensor] = []

        for chunk in flat.split(self.config.energy_batch_size):
            try:
                logpsi = self.log_psi(chunk)
            except ValueError as e:
                logger.warning(
                    f"Skipping chunk due to log_psi error: {e}"
                )
                continue

            if not torch.isfinite(logpsi).all():
                logger.warning("Skipping chunk with non-finite log_psi")
                continue

            local_energy = self.hamilton.local_energy(chunk)
            finite_mask = torch.isfinite(local_energy)
            if not finite_mask.all():
                logger.warning("Dropping non-finite local_energy entries")
                logpsi = logpsi[finite_mask]
                local_energy = local_energy[finite_mask]

            if logpsi.numel() == 0:
                continue

            logpsis.append(logpsi)
            local_es.append(local_energy)

        if len(logpsis) == 0:
            return None, None

        return torch.cat(logpsis, dim=0), torch.cat(local_es, dim=0)

    def save_checkpoint(self, step):
        if step % self.config.checkpoint_step == 0:
            # Check if father directory checkpoint_path exist.
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "step": step,
                },
                self.config.init_checkpoint(),
            )
            print(f"Saved checkpoint at step {step}")

    def train(self):
        """
        Create samples, using those computes the E_mean, E,
        Then using model carlo you can compute the derivative of the loss.
        Important the detach.
        """
        run = self.config.init_wandb(self.model.config)
        train_start = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
        for step in range(self.config.train_steps):
            step_start = time.perf_counter()
            # samples: (monte_carlo, B, n_e, 3)
            samples = self.mh.sampler()

            log_psi_vals, local_energies = self._batched_energy_eval(samples)
            if log_psi_vals is None or local_energies is None:
                logger.warning(
                    f"No valid samples at step {step}; resampling next step."
                )
                continue

            # Energy Local Expection
            E_mean = local_energies.mean().detach()

            # Derivative of the Loss
            loss = 2*((local_energies.detach() - E_mean) * log_psi_vals).mean()

            # Optimizer Step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # self.save_checkpoint(step)

            # Print info
            logger.info(f"Step {step}: E_mean = {E_mean.item():.6f}")
            logger.info(f"Loss = {loss.item():.6f}")

            # Wandb sync information

            env_up = self.model.orbital_head.envelope_up
            env_down = self.model.orbital_head.envelope_down

            metrics = {
                "Energy": E_mean,
                "loss": loss,
                "step_time_sec": time.perf_counter() - step_start,
                "env_up_pi_norm": env_up.pi.detach().norm().item(),
                "env_up_sigma_norm": env_up.raw_sigma.detach().norm().item(),
                "env_down_pi_norm": env_down.pi.detach().norm().item(),
                "env_down_sigma_n": env_down.raw_sigma.detach().norm().item(),
            }
            if torch.cuda.is_available():
                torch.cuda.synchronize(self.device)
                metrics.update(
                    {
                        "gpu/mem_allocated_mb": torch.cuda.memory_allocated(
                            self.device
                        ) / 2**20,
                        "gpu/mem_reserved_mb": torch.cuda.memory_reserved(
                            self.device
                        ) / 2**20,
                        "gpu/max_mem_allocated_mb": torch.cuda.max_memory_allocated(
                            self.device
                        ) / 2**20,
                    }
                )
                torch.cuda.reset_peak_memory_stats(self.device)
            run.log(metrics)
        total_time = time.perf_counter() - train_start
        inf = f"Total train time:{total_time/60:.2f} min ({total_time:.1f}sec)"
        logger.info(inf)
        run.log({"total_training_time_sec": total_time})
        run.finish()


train_config = Train_Config(
        run_name="MANY_ELECTRONS",
        checkpoint_name="MANY_ELECTRONS.pth",
        wand_mode="offline"
    )

if __name__ == "__main__":
    device = get_device()
    print(f"Using {device}")

    # Model
    model_config = Model_Config()
    model = PsiFormer(model_config)

    trainer = Trainer(model, train_config)

    # train the model
    trainer.train()
