from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as Optimizer
from torch.autograd.functional import hessian

import math
from typing import Callable


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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
        x += self.attn(x)
        x += self.mlp(x)
        return x


class Model_Config():
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 32
    n_features: int = 4  # For Hidrogen atom, vec r plus distance
    n_out: int = 3


class Train_Config():
    train_steps: int = 10000
    checkpoint_step: int = 100
    monte_carlo_length: int = 20  # Num samples
    burn_in_steps: int = 20
    checkpoint_name: str = "last_checkpoint.pth"
    dim: int = 4  # For Hidrogen Atom


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
        if x.ndim == 0:
            x = x.unsqueeze(0)
        f1 = self.f_1(x)

        # print("Input Hidden States:", first.shape)
        output = self.f_h(f1)

        # print("Output Hidden States:", output.shape)
        f_n = self.f_n(output)
        # print("Last dim", f_n.shape)
        return f_n


def kinetic_from_log(model: PsiFormer, x: torch.Tensor) -> torch. Tensor:
    """
    Returns the value H psi / psi
    """
    # derivative = torch.autograd.grad(f(x).sum(), x, create_graph=True)[0]
    log = torch.log(model(x))
    return laplacian(log, x) + torch.square(jacobian(log, x))


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
        alpha = self.target(trial) - self.target(current_state)
        if torch.rand(()) < torch.exp(torch.minimum(alpha, torch.tensor(0.0))):
            return True
        return False

    def sampler(self) -> torch.Tensor:
        # Thermalization
        x = torch.randn(self.dim)

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

    def montecarlo_estimator(self, func: Callable) -> torch.Tensor:
        avg = torch.tensor([0.0])    
        samples = self.sampler() # Samples is a Tensor recall that
        for sample in samples:
            avg += func(sample)
        return avg


class Potential():
    """
    For now just of the hidrogen atom.
    """
    def __init__(self, r_e: torch.Tensor, r_p: torch.Tensor):
        # Compute the potential between the hidrogen proton and electron
        self.r_e = r_e
        self.r_p = r_p
 
    def potential(self) -> torch.Tensor:
        return (torch.norm(self.r_e-self.r_p))**(-1)


class Hamiltonian():
    def __init__(self, func: Callable[[torch.Tensor], torch.Tensor]):
        self.func = func

    def laplacian(self, f: Callable[[torch.Tensor],
                  torch.Tensor], x: torch.Tensor) -> torch.Tensor:
        # Compute the Laplacian at a single point by tracing the Hessian
        x = x.clone().detach().requires_grad_(True)
        hess = hessian(f, x)
        #  more derivatives that I want
        return torch.einsum("ii->", hess)

    def local_energy(self, sample: torch.Tensor) -> torch.Tensor:
        V = Potential(sample[2:], sample[:2])
        V = V.potential()
        K = (-0.5 * self.laplacian(self.func, sample)) # This guy should 
        return K + V

   
class train():
    def __init__(self, model, config: Train_Config):
        self.model = model
        self.config = config

    def model_square(self, x: torch.Tensor) -> torch.Tensor:
        psi2 = (torch.norm(self.model(x))**2)
        return psi2

    def expectation_EL(self, local_energy) -> torch.Tensor:
        
        return avg.mean()

    def save_checkpoint(self, step):
        if step % self.config.checkpoint_step == 0:
            print("Saving checkpoint")
            # Overwrite the last checkpoint
    def derivative_loss(self, local_energy) -> torch.Tensor:
        # (Local Energy - expectiation) log psi
        e = self.expectation_EL(local_energy)
        avg = torch.tensor([0.0])
        for sample in samples:
            avg += 
        return


    def train(self):
        mh = MH(self.model, self.config.burn_in_steps,
                self.config.monte_carlo_length, self.config.dim)
        optimizer = Optimizer.Adam(self.model.parameters(), lr=1e-3)
        
        for step in range(self.config.train_steps):
            hamilton = Hamiltonian(self.model)
            local_energy = hamilton.local_energy
            expectation_local_energy = mh.montecarlo_estimator(local_energy)
            deviation = (local_energy - expectation_local_energy)*torch.log(self.model)
            derivative_loss = mh.montecarlo_estimator(deviation)
            
            optimizer.zero_grad()
            optimizer.step()
            if step % 100 == 0:
                print("Loss", loss)


def main():
    device = get_device()
    print(f"Using {device}")

    # Model
    model_config = Model_Config()
    model = PsiFormer(model_config)

    # Train
    train_config = Train_Config()
    trainer = train(model, train_config)
    # train the model
    trainer.train()


if __name__ == "__main__":
    main()
