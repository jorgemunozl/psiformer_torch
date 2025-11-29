import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class MHA(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
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
        y = att @ v  # heads, T, T x heads, T , head_dim -> T, head_dim

        # Back to (T, emb_dim)
        y = y.permute(1, 0, 2).contiguous().view(T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
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
    def __init__(self, config):
        super().__init__()
        self.attn = MHA(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x += self.attn(x)
        x += self.mlp(x)
        return x


class Config():
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 124


def main():
    device = get_device()
    print(f"Using {device}")
    config = Config()
    layer = Layer(config)
    hidden_states = torch.randn(3, config.n_embd)  # (seq_len, embedding_dim)
    print("Input Hidden States:", hidden_states.shape)
    output = layer(hidden_states)
    print("Output Hidden States:", output.shape)


if __name__ == "__main__":
    main()


class PsiFormer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = [Layer(config) for _ in config.n_layer]

    def forward(self, x):
        # return self.layers(x)
        pass

    def _build(self):
        feature = torch.rand(1, 1)
        return feature


def potential(x):
    return x**2


def kinetic_log(f, x):
    derivative = torch.autograd.grad(f(x).sum(), x, create_graph=True)[0]
    return derivative**2


def local_energy(f, x):
    V = potential(x)
    K_log = kinetic_log(f, x)
    return K_log + V


def monte_carlo(local_energy, samples):
    average_energy = 0
    for sample in samples:
        average_energy += local_energy(sample)
    return average_energy / len(samples)


def metropolis_hastings(target, proposal):
    samples = []
    return samples
