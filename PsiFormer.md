---
tags:
  - idea
author: Jorge
date: 2025-09-16 08:53
modified: 2025-12-06 15:56
---
Google Deepmind's implementation with [[TensorFlow]] is a good guided. This work relies on [[PyTorch]]. You are going to learn a ton doing this or at least you are going to present it everywhere you can. So a strong basis is completely necessary, use [[Hugging Face Transformers]] is over-killed.

- [[Psiformer MVP Roadmap]]
- [[Psiformer Milestone 1]]

--- 
# Model

We use [[FermiNet Network]] like guide. Basically a `class Psiformer` that returns the form:
$$
\Psi_{\theta}(\mathbf{x})= \exp\big(\mathcal{J}_{\theta}(\mathbf{x})\big)\, \sum_{k=1}^{N_{\det}}\det[\boldsymbol{\Phi}^{k}_{\theta}(\mathbf{x})], 
$$
Where the Jastrow factor is an instance from  a `class Jastrow` and the I don't have idea.
 
## Input of the Model

The features are variable. I mean you train a different model for each molecule, which I consider is kind of a waste, it would be amazing a single model for dominate all. Or at least a model for each group in the atomic table, at the end the atoms who belong to the same group. Share the same properties, that somehow is encoded in the wave function.

## Hamiltonian and Potential

The inputs from both comes from the samples that **Metropolis Hasting** make. 

### Hamiltonian

Model the Hamiltonian is a lot of fun in the sense that you have to know something about how the [[PyTorch Computational Graph]] works. Because you are going to need how compute the **Laplacian** from a scalar field, when the computational graphs are liberated and how retained it.

On `hamiltonian.py`, we are going to compute $\nabla \psi$ and the other guy. So it's going to be quite important take the partial derivatives from the Net output. This is formulated with [[PyTorch Grad]].

The core is:

```python
class Hamiltonian():
    def __init__(self, log_psi_fn: Callable[[torch.Tensor], torch.Tensor]):
        self.log_psi_fn = log_psi_fn

    def local_energy(self, sample: torch.Tensor) -> torch.Tensor:
        # Hydrogen: potential from proton/electron distance
        V = Potential(sample).potential()
        g = self.grad_log_psi(sample)
        lap = self.laplacian_log_psi(sample)
        kinetic = -0.5 * (lap + (g * g).sum())
        return kinetic + V

    def grad_log_psi(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gradient of log psi with graph retained for higher order derivatives.
        """
        x_req = x.clone().detach().requires_grad_(True)
        y = self.log_psi_fn(x_req)
        (g,) = grad(y, x_req, create_graph=True)
        return g

    def laplacian_log_psi(self, x: torch.Tensor) -> torch.Tensor:
        """
        Laplacian of log psi via second derivatives of each dimension.
        """
        x_req = x.clone().detach().requires_grad_(True)
        y = self.log_psi_fn(x_req)
        (g,) = grad(y, x_req, create_graph=True, retain_graph=True)

        second_terms = []
        for i in range(x_req.numel()):
            (g_i,) = grad(g[i], x_req, retain_graph=True)
            second_terms.append(g_i[i])
        return torch.stack(second_terms).sum()

```


### Potential

Let's begin with the potential for the Hidrogen atom. I didn't think on fixing the movement of the proton. Another important matter is the units, recall that for distances we are working with the Bohr radius. 

Like also the potential $V_{nn},V_{e n},V_{e e}$. 

The proton position vector are fixed to the zero vector. This have a lot of  sense when dealing with a single electron but when you have more than one, you fixed all of them to the zero vector? That is practical? 

The potential energy that comes from the repulsion between electrons in that case would become zero.

#### Hidrogen Baseline

For our baseline the Hydrogen Atom we obtain simply obtain a single term. Important to consider the negative sign and the fact that the hydrogen proton is fixed on the origin. Once that we obtain the model how can we visualize it? How? First that recall that the model is actually biased! Is working in the log space. And the envelope is just in the training. So that is wrong.

Recall that the Hydrogen atom has spherical symmetry. 

Important to check the spherical symmetry.


## Slater Determinants

When working with many electrons

## Jastrow Factor - Envelope

The envelope is a must, with envelope I refer to multiply, by a exponential  decay factor. In Fermi net this is not necessary because 

But in Psiformer, the factor (plus the envelope is completely necessary)


And it was a very nice jump the one, which after I add the envelope for the hydrogen. It goes from energies from $-0.1$ to a nice ones $-0.49$, which is actually the energy to the Hidrogen atom in Hartree units. And that it''s actually amazing.

I mean I think that watch the envelope like this is more clearly, than the paper.  But it's important to differentiate between the Jastrow factor and the envelope  that always appears. In the training is where I am I am adding the envelope, which is wrong but that doesn't matter now. 

```python
# This went into the training
def log_psi(self, x: torch.Tensor) -> torch.Tensor:
     x = x.to(self.device)
     r = torch.linalg.norm(x, dim=-1)

        # Important: Add the envelope
     envelope = -self.model.config.envelope_beta * r

     return self.model(x) + envelope
```

But If I put the envelope in the model itself then thing begin to broke, recall that we are using Psiformer for the Sampling also.


## MLPs

The MLPs are easy to build. I mean torch basically make all the dirty work.

```python
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

```


## Transformers

The main piece is the **Attention Part**, first we are not using the three dimension this is we are not using batches. Why I want them?

```python
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
```


But here have to be careful, because the follow questions are important ones.

- In the hydrogen at, what concept we can find to the attention steps?
- When more electrons and protons, how create the attention between protons and electrons? Electrons, electrons only? 

## Molecule Configuration

A atom has a `atomic_number` and all that stuff that is super useful when you want 
- [[FermiNet Molecule Atom Configuration]]

## Training

[[Wandb]] for supervision, saving [[PyTorch Checkpoints]].
[[FermiNet Metropolis Hasting]]. 

Here the tricky relies on the `.detach()` that you have to make. And how is that the derivative is completely necessary. I mean, the big question that you have to make is: Where you want to back propagate and where not. You want to propagate  on $\log \psi$ the argument of the expectation. You are going to threat $\mathbb{E}(E_{L}(X))$, like constant? If that were the case, you would make a `detach()`.  Or a `with torch.no_grad()`. 

Or more precisely you are going to threat $E_{L}(R)$, like a constant. One of the first reason is because if we are not doing that then higher order derivatives are computed. ($E_{L}$ is computed from the laplacian. From psi), then you need three order higher derivatives. Which is actually a lot of error accumulated. 

- We don't want that, so we treat the samples $\mathbf{R}_{k}$ as constants.
- Treat **local energies** as constants.
- We only use the backward guy to get $\nabla_{\theta}\log(\psi)$. Everything else is like just number that our computational graph does not see.


At the end you are going to make:

```python
loss.backward()
```


## Variational Principle 

It says that the energy from our Ansatz should be greater than the ground state, but we don't observe that, why?


## Metropolis Hasting

Now, I think that we have two options for save the list, I should use a `list` or a tensor? Using [[PyTorch Stack]], then you can make `.mean()` and becomes more easy.

Another important matter is how you propose your first initial configuration. For instance I do it sampling from the **Normal Distribution**.

```python
    def sampler(self) -> torch.Tensor:
        # Thermalization
        x = torch.randn(self.dim)
        # Here the first configuration is
        # sampled from a normal distribution is n.

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
```

That is going to induce a distance to a origin how it looks like that PDF? Interesting the truth, that thing have bohr radios units?

And here it's important realize that we are working in the log space. We are assuming that the model input already belong to the log space. (Important to consider if we want to plot it. Although we can potentiate to obtain the regular scale.)

So in that sense we are doing:

```python
def accept_decline(self, trial: torch.Tensor,
                       current_state: torch.Tensor) -> bool:
        # Sampling does not need gradients; keep it detached from autograd.
    with torch.no_grad():
        alpha = 2*(self.target(trial) - self.target(current_state))
    if torch.rand(()) < torch.exp(torch.minimum(alpha, torch.tensor(0.0))):
        return True
    return False
```

In the regular space I was taking the quotient. Here assuming that we are dealing with `target=model`. With
$$
\text{model}=2\log |\psi|
$$

## Optimizer KFCA

---

I think that there exist a library. I am not sure
Important methods:

`class PsiformerOptions`
`make_layer_norm`
`make_multi_head_attention`
`make_mlp`
`make_self_attention_block`
`make_psiformer_layers`
`make_fermi_net`


