from __future__ import annotations
from jastrow import Jastrow
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
        B, T, C = x.size()  # Batch, sequence length, Embedding dim
        # Imposes that our x is 3D, i.e.,
        # (batch_size, seq_len=n_electrons, embedding_dim)

        # Get query, key, values from single linear projection.
        qkv = self.c_attn(x)

        q, k, v = qkv.split(self.n_embd, dim=2)

        head_dim = C // self.n_head

        # dim (B, T    , heads, head_dim) -> trans
        # dim (B, heads, T    , head_dim)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        # (B, heads, T, head_dim) x (B, heads, head_dim , T)->(B, head, T, T)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
        att = F.softmax(att, dim=-1)
        # (B, heads, T, T) x (B, heads, T , head_dim)->(B, heads, T, head_dim)
        y = att @ v

        # Back to (B, T, heads, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
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


class Envelope(nn.Module):
    def __init__(self, natom: int, det_spin: int, sigma_init: float = 0.5):
        super().__init__()
        self.pi = nn.Parameter(torch.ones(natom, det_spin))

        # Start with a configurable decay rate; learnable during training.
        self.raw_sigma = nn.Parameter(
            torch.full((natom, det_spin), sigma_init)
        )

    def forward(self, r_ae: torch.Tensor) -> torch.Tensor:
        """
        n_spin: up/down
        r_ae: (B, n_spin, natom, 1) distances between electrons and nuclei
        return: (B, n_spin, det_spin).
        The last dimension of r_ae is important for broadcasting.
        1 -> n_spin
        """

        sigma = F.softplus(self.raw_sigma) + 1e-6

        # Broadcasting for B, n_elec, _ , 1
        return torch.sum(torch.exp(-r_ae * sigma) * self.pi, dim=2)


class Orbital_Head(nn.Module):
    """
    This guy create k matrix to takes the determinants.
    """
    def __init__(self, config: Model_Config) -> None:
        super().__init__()
        self.n_det = config.n_determinants
        self.n_spin_up = config.n_spin_up
        self.n_spin_down = config.n_spin_down
        # Atom: nucleus at origin -> natom = 1
        self.n_atom = 1
        self.envelope_up = Envelope(
            self.n_atom, self.n_det * self.n_spin_up
        )
        self.envelope_down = Envelope(
            self.n_atom, self.n_det * self.n_spin_down
        )
        self.n_embd = config.n_embd

        # Heads for up/down
        self.orb_up = nn.Linear(self.n_embd, self.n_det*self.n_spin_up)
        self.orb_down = nn.Linear(self.n_embd, self.n_det*self.n_spin_down)
        self.det_logits = nn.Parameter(torch.zeros(self.n_det))

    def build_orbital_matrix(self, h: torch.Tensor,
                             r_ae: torch.Tensor, spin: str) -> torch.Tensor:
        """
        n_spin = up / down
        r_ae_up = (B, n_spin, n_atom, 1)
        h_up/down: (B, n_spin, n_embd)
        return: (B, n_det, n_spin, n_spin)
        """
        # From where the H comes from.

        # Take in account the spin
        if spin == "up":
            out = self.orb_up(h)
            env = self.envelope_up
            n_spin = self.n_spin_up

        else:
            out = self.orb_down(h)
            env = self.envelope_down
            n_spin = self.n_spin_down

        # out: (B, N_spin, n_det * n_spin_up/down), no broadcasting
        out = out * env(r_ae)

        B, N, _ = out.shape

        return out.view(B, N, self.n_det, n_spin).transpose(1, 2)

    def slogdet_sum(self, mats: torch.Tensor
                    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        mats: (B, n_det, n_spin, n_spin)
        returns stable sum of determinants
        """
        # Add a tiny diagonal to keep matrices nonsingular at init.
        eps = 1e-6
        eye = torch.eye(
            mats.size(-1), device=mats.device, dtype=mats.dtype
        ).unsqueeze(0).unsqueeze(0)
        mats = mats + eps * eye

        signs = []
        logabs = []
        for k in range(self.n_det):
            # (B, ndet)
            sign, logs_abs = torch.linalg.slogdet(mats[:, k])
            signs.append(sign)
            logabs.append(logs_abs)

        # (B, n det)
        signs = torch.stack(signs, dim=-1)
        logabs = torch.stack(logabs, dim=-1)

        weights = torch.softmax(self.det_logits, dim=0)
        
        max_logabs, _ = logabs.max(dim=-1, keepdim=True)
        weighted = weights * signs * torch.exp(logabs - max_logabs)
        summed = weighted.sum(dim=-1)
        sign_total = torch.sign(summed + 1e-12)
        logabs_total = (max_logabs.squeeze(-1) +
                        torch.log(summed.abs() + 1e-12)
                        )
        return sign_total, logabs_total

    def forward(self, h, spin_up_idx, spin_down_idx, r_ae_up, r_ae_down):
        """
        h: (B, n_elec, n_embd)
        spin_up_idx: [n_up]
        spin_down_idx: [n_down]
        r_ae_up: (B, n_up_,n_atom,1)
        r_ae_down: (B,n_down,n_atom,1)
        """
        h_up = h[:, spin_up_idx, :]
        h_down = h[:, spin_down_idx, :]
        phi_up = self.build_orbital_matrix(h_up, r_ae_up, "up")
        phi_down = self.build_orbital_matrix(h_down, r_ae_down, "down")

        sign_logdet_up = self.slogdet_sum(phi_up)
        sign_logdet_down = self.slogdet_sum(phi_down)
        return sign_logdet_up, sign_logdet_down


class PsiFormer(nn.Module):
    """
    Generate the hidden dimensions for Orbital Head
    Convention: The first spin are up, the last are down
    """
    def __init__(self, config: Model_Config):
        super().__init__()
        self.config = config
        self.l_0 = nn.Linear(config.n_features+1, config.n_embd)
        self.layers = nn.ModuleList(
            [Layer(config) for _ in range(config.n_layer)]
        )
        self.orbital_head = Orbital_Head(config)
        self.jastrow = Jastrow(config.n_spin_up, config.n_spin_down)
        self.spin_up_idx = list(range(self.config.n_spin_up))
        self.spin_down_idx = list(
            range(self.config.n_spin_up,
                  self.config.n_spin_up+self.config.n_spin_down)
        )
        assert set(self.spin_up_idx).isdisjoint(self.spin_down_idx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, n_electron ,3)
        """
        if x.shape[1:] != (self.config.n_electron_num, self.config.n_features):
            error = f"x shape: {x.shape}"
            error += f"{self.config.n_electron_num, self.config.n_features}"
            raise ValueError("Input model shape mismatch", error)

        r = torch.linalg.norm(x, dim=-1, keepdim=True)  # (B, n_elec, 1)
        features = torch.cat([x, r], dim=-1)            # (B, n_electron, 4)

        h = self.l_0(features)  # (B, n_electron, n_embd)

        for layer in self.layers:
            h = layer(h)  # (B, n_electron, n_embd)

        # Electron-nucleus distances
        # (nucleus assumed at origin): (B, n_elec, natom, 1)
        r_ae = torch.linalg.norm(x[:, :, None, :],
                                 dim=-1, keepdim=True)  # |r-R|

        r_ae_up = r_ae[:, self.spin_up_idx, :, :]
        r_ae_down = r_ae[:, self.spin_down_idx, :, :]

        sign_logdet_up, sign_logdet_down = self.orbital_head(
            h, self.spin_up_idx, self.spin_down_idx, r_ae_up, r_ae_down
        )
        sign_up, logdet_up = sign_logdet_up
        sign_down, logdet_down = sign_logdet_down

        jastrow_term = self.jastrow(x)
        # Guard against singular determinant blocks
        if (not torch.isfinite(logdet_up).all()
                or not torch.isfinite(logdet_down).all()):
            raise ValueError("Non-finite log determinant detected")

        sum_det = logdet_up + logdet_down
        log_sign = torch.log(sign_up * sign_down + 1e-12)
        log_psi = sum_det + log_sign + jastrow_term

        # log_psi: (B, )
        return log_psi
