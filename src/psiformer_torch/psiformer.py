from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Model_Config
import math


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
        B, T, C = x.size()  # sequence length, Embedding dim
        # imposes that our x is 3D, i.e.,
        # (batch_size, seq_len, embedding_dim)

        # get query, key, values from single linear projection
        qkv = self.c_attn(x)
        # print(qkv.size())
        q, k, v = qkv.split(self.n_embd, dim=1)

        # print("K Before View:", k.shape)
        head_dim = C // self.n_head

        # dim (B, heads, T, head_dim)
        k = k.view(B, T, self.n_head, head_dim).permute(1, 0, 2)
        q = q.view(B, T, self.n_head, head_dim).permute(1, 0, 2)
        v = v.view(B, T, self.n_head, head_dim).permute(1, 0, 2)

        # B, head, T, head_dim x head, head_dim , T -> B, head, T, T
        att = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
        att = F.softmax(att, dim=-1)
        y = att @ v  # B, heads, T, T x heads, T , head_dim -> B, heads , T, head_dim

        # Back to (B, T, heads, head_dim)
        y = y.permute(1, 0, 2).contiguous().view(T, C)  # Fix because the batch.
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



class Orbital_Head(nn.Module):
    """
    This guy create k matrix to takes the determinants.
    """
    def __init__(self, n_elec, n_det) -> None:
        super().__init__()
        self.n_elec = n_elec
        self.n_det = n_det
    def forward(self, x):
        

class PsiFormer(nn.Module):
    """
    Generate the hidden dimensions for Orbital Head
    """
    def __init__(self, config: Model_Config):
        super().__init__()
        self.config = config
        self.f_1 = nn.Linear(config.n_features, config.n_embd)  # From input feature -> hidden dimension
        self.f_h = Layer(config) # Hidden Dimension

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

        r = torch.linalg.norm(x, dim=-1)
        envelope = -self.config.envelope_beta * r

        R_up = x[:, :0, :]
        R_down = x[:, :1, :]

        # From input features to embedding dimension
        R_up = self.f_1(R_up)
        R_down = self.f_1(R_down)

        R_up = self.f_h(R_up)
        R_down = self.f_h(R_down)

        # Return the sum of determinants
        # d = torch.tensor([0])
        #for k in range(self.config.n_determinants):
        #    orbital = make_orbital()
        #    for i in range(n_orbitals):
        #        phi_ij = 2
        #    d += torch.det(orbital_matrix)

        return (x+envelope).squeeze(-1).mean(dim=-1)
