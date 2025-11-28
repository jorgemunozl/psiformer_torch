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
        assert config.n_embd % config.n_head == 0  # 768/12 = 64 features

        # key, query, value projections for all heads
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # create causal mask buffer (lower triangular matrix)
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer(
            "bias",
            mask.view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x):
        B, T, C = x.size()  # batch, sequence length, Context Length, this imposes that our x is 3D, i.e., (batch_size, seq_len, embedding_dim)

        # get query, key, values from single linear projection
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        # reshape and transpose for attention calculation
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # compute attention scores
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0,
                              float('-inf'))
        att = F.softmax(att, dim=-1)
        # apply attention to values
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


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
    block_size: int = 10
    n_head: int = 3  # It must divide n_embd
    n_embd: int = 9


config = Config()
layer = Layer(config)
value = torch.randn(1, 1, config.n_embd)

B, T, C = value.size()
print(B, T, C)
# output = layer(value)