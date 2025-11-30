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
        y = att @ v  # heads, T, T x heads, T , head_dim -> heads , T, head_dim

        # Back to (T, heads, head_dim)
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


class First_Layer(nn.Module):
    def __init__(self, n_features, n_emb):
        super().__init__()
        self.f1 = nn.Linear(n_features, n_emb)

    def forward(self, x):
        return self.f1(x)


class Last_Layer(nn.Module):
    def __init__(self, n_emb, output_dim) -> None:
        super().__init__()
        self.f1 = nn.Linear(n_emb, output_dim)

    def forward(self, x):
        return self.f1(x)


class Config():
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 32
    n_features: int = 4
    n_out: int = 3


class PsiFormer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.f_1 = First_Layer(config.n_features, config.n_embd)
        self.f_h = Layer(config)
        self.f_n = Last_Layer(config.n_embd, config.n_out)

    def build_features(self, r_electron=torch.rand(3),
                       r_proton=torch.rand(3)) -> torch.Tensor:
        """
        Hidrogen atom, simple.
        """
        h_0_1 = r_electron-r_proton
        h_0_2 = torch.norm(h_0_1)
        return torch.cat([h_0_1, torch.tensor([h_0_2])])

    def forward(self):
        features = self.build_features()
        f1 = self.f_1(features)
        first = torch.unsqueeze(f1, 0)
        # print("Input Hidden States:", first.shape)
        output = self.f_h(first)
        # print("Output Hidden States:", output.shape)
        f_n = self.f_n(output)
        # print("Last dim", f_n.shape)
        return f_n


def main():
    device = get_device()
    print(f"Using {device}")
    config = Config()
    model = PsiFormer(config)
    output = model()
    print("Output", output)


if __name__ == "__main__":
    main()


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
