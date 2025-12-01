from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as Optimizer
from torch.autograd.functional import jacobian


import math
from typing import List


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
    steps: int = 10000
    checkpoint_step: int = 100
    monte_carlo_size: int = 20
    burn_in: int = 20
    checkpoint_name: str = "last_checkpoint.pth"


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


def potential(r_e: torch.Tensor, r_p: torch.Tensor) -> torch.Tensor:
    # Compute the potential between the hidrogen proton and electron
    return (torch.norm(r_e-r_p))**(-1)


def kinetic_from_log(model: PsiFormer, x: torch.Tensor) -> torch. Tensor:
    """
    Returns the value H psi / psi
    """
    # derivative = torch.autograd.grad(f(x).sum(), x, create_graph=True)[0]
    log = torch.log(model(x))
    return laplacian(log, x) + torch.square(jacobian(log, x))


def laplacian(f, x: torch.Tensor) -> torch.Tensor:
    return jacobian(jacobian(f, x)[0], x)[0]


def kinetic(model: PsiFormer, x: torch.Tensor) -> torch.Tensor:
    """Computes the raw kinetic term"""
    return laplacian(model, x)


class train():
    def __init__(self, model: PsiFormer, config: Train_Config):
        self.model = model
        self.config = config

    def model_square(self, x: torch.Tensor) -> torch.Tensor:
        psi2 = (torch.norm(self.model(x))**2)
        return psi2

    def local_energy(self, sample: torch.Tensor) -> torch.Tensor:
        V = potential(sample[2:], sample[:2])
        K = kinetic_from_log(self.model, sample)
        return K + V

    def generate_trial(self, state: torch.Tensor) -> torch.Tensor:
        sample = state + torch.randn_like(state)
        return sample

    def accept_decline(self, trial: torch.Tensor,
                       current_state: torch.Tensor) -> bool:
        quotient = self.model_square(trial)/self.model_square(current_state)
        min = torch.min(torch.ones_like(quotient), quotient)
        if torch.rand_like(quotient) < min:
            return True
        return False

    def thermalize(self) -> List[torch.Tensor]:
        markov_chain = [torch.rand(1, 4)]
        while len(markov_chain) < self.config.burn_in:
            trial = self.generate_trial(markov_chain[-1])
            if self.accept_decline(trial, markov_chain[-1]):
                markov_chain.append(trial)
        return markov_chain

    def sampler(self, markov_chain_state: torch.Tensor) -> List[torch.Tensor]:
        samples = [markov_chain_state]
        while len(samples) < (self.config.monte_carlo_size):
            trial = self.generate_trial(samples[-1])
            if self.accept_decline(trial, samples[-1]):
                samples.append(trial)
        return samples

    def compute_loss_mc(self) -> torch.Tensor:
        samples = self.sampler(self.thermalize()[-1])
        avg = torch.zeros_like(samples[0])
        for sample in samples:
            avg += self.local_energy(sample)
        return avg.mean()

    def save_checkpoint(self, step):
        if step % self.config.checkpoint_step == 0:
            print("Saving checkpoint")
            # Overwrite the last checkpoint

    def train(self):

        optimizer = Optimizer.Adam(self.model.parameters(), lr=1e-3)
        for step in range(1000):
            self.save_checkpoint(step)
            loss = self.compute_loss_mc()
            optimizer.zero_grad()
            loss.backward()
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

    # Metropolist Test, 
    k

    # train the model
    #trainer.train()


if __name__ == "__main__":
    main()
