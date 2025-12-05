from __future__ import annotations
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad

import math
from typing import Callable


CHECKPOINT_PATH = "checkpoints/"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def grad_log_psi(log_psi_fn: Callable[[torch.Tensor], torch.Tensor],
                 x: torch.Tensor) -> torch.Tensor:
    """
    Gradient of log psi with graph retained for higher order derivatives.
    """
    x_req = x.clone().detach().requires_grad_(True)
    y = log_psi_fn(x_req)
    (g,) = grad(y, x_req, create_graph=True)
    return g


def laplacian_log_psi(log_psi_fn: Callable[[torch.Tensor], torch.Tensor],
                      x: torch.Tensor) -> torch.Tensor:
    """
    Laplacian of log psi via second derivatives of each dimension.
    """
    x_req = x.clone().detach().requires_grad_(True)
    y = log_psi_fn(x_req)
    (g,) = grad(y, x_req, create_graph=True, retain_graph=True)

    second_terms = []
    for i in range(x_req.numel()):
        (g_i,) = grad(g[i], x_req, retain_graph=True)
        second_terms.append(g_i[i])
    return torch.stack(second_terms).sum()


class MHA(nn.Module):
    def __init__(self, config: Model_Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x: torch.Tensor):
        T, C = x.size()  # sequence length, Embedding dim
        # imposes that our x is 3D, i.e.,
        # (batch_size, seq_len, embedding_dim)

        # get query, key, values from single linear projection
        qkv = self.c_attn(x)
        # print(qkv.size())
        q, k, v = qkv.split(self.n_embd, dim=1)

        # print("K Before View:", k.shape)
        head_dim = C // self.n_head

        # dim (heads, T, head_dim)
        k = k.view(T, self.n_head, head_dim).permute(1, 0, 2)
        q = q.view(T, self.n_head, head_dim).permute(1, 0, 2)
        v = v.view(T, self.n_head, head_dim).permute(1, 0, 2)

        # head, T, head_dim x head, head_dim , T -> head, T, T
        att = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
        att = F.softmax(att, dim=-1)
        y = att @ v  # heads, T, T x heads, T , head_dim -> heads , T, head_dim

        # Back to (T, heads, head_dim)
        y = y.permute(1, 0, 2).contiguous().view(T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config: Model_Config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Layer(nn.Module):
    # Combines MHA and MLP with residual connections
    def __init__(self, config: Model_Config):
        super().__init__()
        self.attn = MHA(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class Model_Config():
    n_layer: int = 4
    n_head: int = 8
    n_embd: int = 128
    n_features: int = 4  # For Hidrogen atom, vec r, distance
    n_out: int = 1


class Train_Config():
    train_steps: int = 100
    checkpoint_step: int = 10
    monte_carlo_length: int = 4000  # Num samples
    burn_in_steps: int = 1
    checkpoint_name: str = "last_checkpoint.pth"
    dim: int = 3  # For Hidrogen Atom


class PsiFormer(nn.Module):
    def __init__(self, config: Model_Config):
        super().__init__()
        self.f_1 = nn.Linear(config.n_features, config.n_embd)
        self.f_h = Layer(config)
        self.f_n = nn.Linear(config.n_embd, config.n_out)

    def build_features(self, r_electron=torch.rand(3),
                       r_proton=torch.rand(3)) -> torch.Tensor:
        """
        Hidrogen atom, simple.
        """
        h_0_1 = r_electron-r_proton
        h_0_2 = torch.norm(h_0_1)
        return torch.cat([h_0_1, torch.tensor([h_0_2])])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)

        # From input features to embedding dimension
        x = self.f_1(x)

        # print("Input Hidden States:", first.shape)
        x = self.f_h(x)

        # print("Output Hidden States:", output.shape)
        x = self.f_n(x)
        # Return scalar log-psi per sample
        return x.squeeze(-1).mean(dim=-1)


class MH():
    """
    Implementation for the Metropolis Hasting (MH) algorithm
    using a gaussian kernel. Returns a list a samples from
    the target distribution.
    We work with the log form. !
    """
    def __init__(self, target: Callable[[torch.Tensor], torch.Tensor],
                 eq_steps: int,
                 num_samples: int,
                 dim: int,
                 step_size: float = 1.0
                 ):
        self.target = target
        self.eq_steps = eq_steps
        self.num_samples = num_samples
        self.dim = dim

    def generate_trial(self, state: torch.Tensor) -> torch.Tensor:
        sample = state + torch.randn_like(state)
        return sample

    def accept_decline(self, trial: torch.Tensor,
                       current_state: torch.Tensor) -> bool:
        # Sampling does not need gradients; keep it detached from autograd.
        with torch.no_grad():
            alpha = 2*(self.target(trial) - self.target(current_state))
        if torch.rand(()) < torch.exp(torch.minimum(alpha, torch.tensor(0.0))):
            return True
        return False

    @torch.no_grad()
    def sampler(self) -> torch.Tensor:
        # Thermalization
        x = torch.randn(self.dim)  # Here the first configuration is sampled from a normal distribution is n.

        for _ in range(self.eq_steps):
            trial = self.generate_trial(x)
            if self.accept_decline(trial, x):
                x = trial

        # Sampling

        samples = torch.zeros(self.num_samples, self.dim)
        samples[0] = x

        for i in range(1, self.num_samples):
            trial = self.generate_trial(x)
            if self.accept_decline(trial, x):
                x = trial
            samples[i] = x

        return samples


class Potential():
    """
    For the hidrogen atom, the only nucleous is fixed at (0,0,0).
    Broadcast.
    """
    def __init__(self, r_e: torch.Tensor):
        # Compute the potential between the hidrogen proton and electron
        self.r_e = r_e

    def potential(self) -> torch.Tensor:
        eps = 1e-12
        r = torch.linalg.norm(self.r_e, dim=-1)
        return -1/(r+eps)


class Hamiltonian():
    def __init__(self, log_psi_fn: Callable[[torch.Tensor], torch.Tensor]):
        self.log_psi_fn = log_psi_fn

    def local_energy(self, sample: torch.Tensor) -> torch.Tensor:
        # Hydrogen: potential from proton/electron distance
        V = Potential(sample).potential()
        g = grad_log_psi(self.log_psi_fn, sample)
        lap = laplacian_log_psi(self.log_psi_fn, sample)
        kinetic = -0.5 * (lap + (g * g).sum())
        return kinetic + V


class Trainer():
    def __init__(self, model, config: Train_Config):
        self.model = model.to(get_device())
        self.config = config
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.device = get_device()

    def log_psi(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x.to(self.device))

    def save_checkpoint(self, step):
        if step % self.config.checkpoint_step == 0:
            # Check if father directory checkpoint_path exist.
            os.makedirs(CHECKPOINT_PATH, exist_ok=True)
            path = os.path.join(CHECKPOINT_PATH, self.config.checkpoint_name)
            torch.save(self.model.state_dict(), path)
            print(f"Saved checkpoint at step {step}")

    def train(self):
        """
        Create samples, using those computes the E_mean, E,
        Then using model carlo you can compute the derivative of the loss.
        Important the detach.
        """
        mh = MH(self.log_psi, self.config.burn_in_steps,
                self.config.monte_carlo_length, self.config.dim, step_size=0.1)
        hamilton = Hamiltonian(self.log_psi)

        for step in range(self.config.train_steps):
            samples = mh.sampler().to(self.device)
            local_energies = torch.stack(
                [hamilton.local_energy(s) for s in samples]
                )
            log_psi_vals = torch.stack([self.log_psi(s) for s in samples])

            E_mean = local_energies.mean().detach()
            loss = 2*((local_energies.detach() - E_mean) * log_psi_vals).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.save_checkpoint(step)

            if step % 2 == 0:
                print(f"step {step} | loss {loss.item()} | energy {E_mean}")


def main():
    device = get_device()
    print(f"Using {device}")

    # Model
    model_config = Model_Config()
    model = PsiFormer(model_config)

    # Train
    train_config = Train_Config()
    # keep dim consistent with model input size
    train_config.dim = model_config.n_features
    trainer = Trainer(model, train_config)
    # train the model
    trainer.train()


if __name__ == "__main__":
    main()
