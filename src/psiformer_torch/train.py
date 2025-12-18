import torch
import time
import torch.optim as optim
import logging

from psiformer_torch.psiformer import get_device, PsiFormer
from psiformer_torch.config import Train_Config
from psiformer_torch.mcmc import MH
from psiformer_torch.hamiltonian import Hamiltonian
from psiformer_torch.config import debug_conf, small_conf, large_conf
from psiformer_torch.utils.UPLOAD_HF import REPO_ID


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
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                       max_norm=10.0)
            self.optimizer.step()
            self.scheduler.step()

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
                "grad_norm": grad_norm.item() if grad_norm is not None else .0,
                "lr": self.optimizer.param_groups[0]["lr"],
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
                    }
                )
                torch.cuda.reset_peak_memory_stats(self.device)
            run.log(metrics)
        total_time = time.perf_counter() - train_start
        inf = f"Total train time:{total_time/60:.2f} min ({total_time:.1f}sec)"
        logger.info(inf)
        run.log({"total_training_time_sec": total_time})
        run.finish()

        if self.push:
            self.model.push_to_hub(REPO_ID)


def wrapper(type,
            run_name='',
            checkpoint_name='',
            wand_mode=''):
    if type == "large":
        print("Training Large")
        model_config = large_conf[0]
        train_config = large_conf[1]
        run_name = "_LARGE"
    if type == '':
        print("Training Debug")
        model_config = debug_conf[0]
        train_config = debug_conf[1]
        run_name = ''
    if type == 'small':
        print("Training Small")
        model_config = small_conf[0]
        train_config = small_conf[1]
        run_name = "_SMALL"
    print(model_config)

    # Train Config Update
    train_config.run_name = run_name + run_name
    train_config.checkpoint_name = checkpoint_name + run_name
    train_config.wand_mode = wand_mode

    return model_config, train_config


if __name__ == "__main__":
    device = get_device()
    print(f"Using {device}")

    # Model
    model_configs = wrapper("large",
                            run_name="CArbonl",
                            checkpoint_name="CArbonL",
                            wand_mode="online",)
    model = PsiFormer(model_configs[0])

    # Train
    trainer = Trainer(model, model_configs[1], True)

    # train the model
    trainer.train()
