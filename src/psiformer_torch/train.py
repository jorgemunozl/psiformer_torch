import torch
import time
import torch.optim as optim
import logging
from dataclasses import replace

from psiformer_torch.psiformer import get_device, PsiFormer
from psiformer_torch.config import Train_Config
from psiformer_torch.mcmc import MH
from psiformer_torch.hamiltonian import Hamiltonian
from psiformer_torch.config import debug_conf, small_conf, large_conf


# torch.autograd.set_detect_anomaly(True)
torch.set_float32_matmul_precision("high")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s: %(message)s"
                    )
logger = logging.getLogger("Beginning")
logger.info("Starting")


class Trainer():
    def __init__(self, model: PsiFormer, config: Train_Config,
                 push: bool):
        self.model = model.to(get_device())
        self.config = config
        self.push = push
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            betas=(0.9, 0.95),
            weight_decay=1e-4,
            amsgrad=True,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.train_steps, eta_min=config.lr * 0.1
        )
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
            logpsi = self.log_psi(chunk)  # (B, )
            local_energy = self.hamilton.local_energy(chunk)
            # local_energy: (B, )
            logpsis.append(logpsi)
            local_es.append(local_energy)

        return torch.cat(logpsis, dim=0), torch.cat(local_es, dim=0)
        # logpsis: (mc_steps * B, )
        # local_es: (mc_steps * B, )

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
            # Energy Local Expection
            E_mean = local_energies.mean().detach()
            # Derivative of the Loss using Log Derivative Trick
            loss = 2*((local_energies.detach() - E_mean) * log_psi_vals).mean()

            # Optimizer Step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            self.save_checkpoint(step)

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
                "lr": self.optimizer.param_groups[0]["lr"],
                "e_up_pi_norm": env_up.pi.detach().norm().item(),
                "e_up_sigma_norm": env_up.raw_sigma.detach().norm().item(),
                "e_down_pi_norm": env_down.pi.detach().norm().item(),
                "e_down_sigma_norm": env_down.raw_sigma.detach().norm().item(),
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
                    }
                )
                torch.cuda.reset_peak_memory_stats(self.device)
            run.log(metrics)
        total_time = time.perf_counter() - train_start
        inf = f"Total train time:{total_time/60:.2f} min ({total_time:.1f}sec)"
        logger.info(inf)
        run.log({"total_training_time_sec": total_time})
        run.finish()


def wrapper(
    preset: str,
    run_name: str = "",
    checkpoint_name: str = "",
    wand_mode: str = "",
) -> tuple:
    """
    Select a (model_config, train_config) pair by preset name.

    Important: returns *copies* of the preset configs so per-run overrides
    don't mutate the module-level singletons in `psiformer_torch.config`.
    """
    key = (preset or "debug").lower()
    if key == "large":
        base_model_config, base_train_config = large_conf
        suffix = "_LARGE"
    elif key == "small":
        base_model_config, base_train_config = small_conf
        suffix = "_SMALL"
    elif key in ("debug", ""):
        base_model_config, base_train_config = debug_conf
        suffix = "_DEBUG"
    else:
        raise ValueError(
            f"Unknown preset {preset!r}; expected 'debug', 'small' or 'large'."
        )

    model_config = replace(base_model_config)
    train_config = replace(base_train_config)

    base_run_name = run_name or train_config.run_name
    train_config.run_name = f"{base_run_name}{suffix}"

    base_checkpoint_name = checkpoint_name or train_config.checkpoint_name
    if base_checkpoint_name:
        train_config.checkpoint_name = f"{base_checkpoint_name}{suffix}"
    else:
        train_config.checkpoint_name = f"{train_config.run_name}"

    if wand_mode:
        train_config.wand_mode = wand_mode

    logger.info(
        "Selected preset=%s run_name=%s checkpoint_name=%s",
        key,
        train_config.run_name,
        train_config.checkpoint_name,
    )
    print(model_config)
    return model_config, train_config


if __name__ == "__main__":
    # Get device
    device = get_device()
    print(f"Using {device}")

    # Model
    model_config, train_config = wrapper("small",
                                         run_name="Helium",
                                         checkpoint_name="Helium",
                                         wand_mode="offline")
    model = PsiFormer(model_config)

    # Train
    trainer = Trainer(model, train_config, True)

    # train the model
    trainer.train()
