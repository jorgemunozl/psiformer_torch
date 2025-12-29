---
tags:
  - idea
author: Jorge
date: 2025-09-16 08:53
modified: 2025-12-18 06:55
---
Implementation on [[PyTorch]] from DeepMind's Fermi Net. Their work was implemented on [[TensorFlow]]. 

# Model

The output of the model is:
$$
\Psi_{\theta}(\mathbf{x})= \exp\big(\mathcal{J}_{\theta}(\mathbf{x})\big)\, \sum_{k=1}^{N_{\det}}\det[\boldsymbol{\Phi}^{k}_{\theta}(\mathbf{x})], 
$$
Where $\mathbf{x}$ is a configuration, (more detailed later). $\mathcal{J}$ is the and the orbitals. In practice we are working in the **log space**, basically we are applied the **log** operation to all the elements:

$$
\text{model output}=\log(\Psi_{\theta})=\mathcal{J}_{\theta}+\log\left( \sum_{k=1}^{N_{\det}}\det[\boldsymbol{\Phi}^{k}_{\theta}(\mathbf{x})]\right)
$$

Start with the input model:

## Input of the Model

It receives a tensor with shape $(B, n_{\text{electrons}}, 3)$, where is the **batch size** arbitrary, (useful for parallelization). The numbers of electron depends on what atom you want to use. And the three because each electron live in the three dimensional space.
After receive that tensor, we are going to compute the distance between the electrons and the nucleus, in the most simple case, atoms, the nucleus is fixed at the origin. Once that the distance of the electrons are computed, we concatenate over the last dimension, thus the input for the first layer is $(B, n_{\text{electron}},4)$ (Our **input features**).

```bash
# Psiformer forward
# x: input
r = torch.linalg.norm(x, dim=-1, keepdim=True)  # (B, n_elec, 1)
features = torch.cat([x, r], dim=-1)            # (B, n_electron, 4)
```

The model model architecture is composed like follow:

```python
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
```

Important the fact where that the sum of **spin up** and **spin down** is equal to the numbers of electrons.

The `self.l_0` is the first layer, which is going to take our **input features** $(B,n_{\text{electrons}},4)$ and take to the **embedding space** $(B,n_{\text{elec}},n_{\text{emb}})$. We are going to call **hidden feature** to this.

Then it comes the `self.layers`, which is a sequence of `Layers`. Each layer implement a **Multi Head Attention** along a **MLP**.

## Multi Head Attention

The **MHA** step takes our **hidden features** to another dimension three times bigger this is $(B, n_{e},3n_{emb})$, doing a `split`, over the last dimension we obtain the so called **key, queries and values**. Then we make a `view` adding a four dimension, for slicing the `q,k,v` in heads. And a `transpose` to prepare the matrix multiplication that comes later.

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
        B, n_elec, C = x.size()  # Batch, sequence length, Embedding dim
        # Imposes that our x is 3D, i.e.,
        # (batch_size, seq_len=n_electrons, embedding_dim)

        # Get query, key, values from single linear projection.
        qkv = self.c_attn(x)

        q, k, v = qkv.split(self.n_embd, dim=2)

        head_dim = C // self.n_head

        # dim (B, n_elec , heads, head_dim) -> trans
        # dim (B, heads, n_elec , head_dim)
        k = k.view(B, n_elec, self.n_head, head_dim).transpose(1, 2)
        q = q.view(B, n_elec, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, n_elec, self.n_head, head_dim).transpose(1, 2)

        # (B, heads, n_elec, head_dim) x (B, heads, head_dim , n_elec)->
        # (B, head, n_elec, n_elec)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
        att = F.softmax(att, dim=-1)
        # (B, heads, n_elec, n_elec) x (B, heads, n_elec , head_dim)->
        # (B, heads, n_elec, head_dim)
        y = att @ v

        # Back to (B, n_elec, heads, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, n_elec, C)
        return self.c_proj(y)
```

We compute the **logits** using the matrix multiplication and dividing by the square root of the **head dimension**. Conceptually we are applying the dot product between the vector which represents the relation between nucleus, with another ones.

- **Here** is important to ask, we would obtain the same results with pairs of electron instead pair electron-nucleus? Like Fermi Net?

Apply **Softmax** in the last dimension, $(n_{elec})$

## MLP

The **MLP** take the **hidden features** go to a more bigger dimension.

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

We are using the [[PyTorch Activation Function]] **Gelu**, with `approximate='tanh'` which means:

We are unifying the both part on a `class Layer` like follow:

```python
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
```

We consider add a [[Pytorch Layer Normalization]].

Now we are going to explain what we are going to do with this processed **hidden features**. But first explain the **Envelope**.

## Envelope

A initial condition that need to fulfill is:
$$
\lim_{ \lvert \vec{x} \rvert  \to \infty } \psi(\vec{x})=0
$$
For that matter we are going to force this using a exponential decay term,  for instance for the hydrogen this term took the form of:

$$
E=e^{ -\beta \lvert x \rvert  }
$$

When considering more electrons to consider all the elements:

$$
\pi \Sigma
$$

```python
class Envelope(nn.Module):
    def __init__(self, natom: int, det_times_spin: int,
                 sigma_init: float = 0.5):
        super().__init__()
        self.pi = nn.Parameter(torch.ones(natom, det_times_spin))

        # Start with a configurable decay rate; learnable during training.
        self.raw_sigma = nn.Parameter(
            torch.full((natom, det_times_spin), sigma_init)
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
        sigma = torch.clamp(sigma, min=1e-3, max=1e3)
        pi = torch.clamp(self.pi, min=1e-3, max=1e3)

        # Broadcasting for B, n_elec, _ , 1
        # r_ae * sigma = (B. n_spin, natom, det_times_spin)
        # * pi = (B, n_spin, natom, det_times_spin), sum to natom
        # (B, n_spin, det_times_spin)
        return torch.sum(torch.exp(-r_ae * sigma) * pi, dim=2)
```

Here we return $(B,n_{\text{spin}}, \det \times \text{spin})$ that is going to be 


## Orbital Heads

With the **hidden features** that we obtain from the layers we are going to create matrices  to apply the **determinant operation**. The class is defined as:

```python
class Orbital_Head(nn.Module):
    """
    This guy create k matrix to takes the determinants.
    """
    def __init__(self, config: Model_Config) -> None:
        super().__init__()
        self.n_det = config.n_determinants
        self.n_embd = config.n_embd
        self.n_spin_up = config.n_spin_up
        self.n_spin_down = config.n_spin_down

        # Atom: nucleus at origin -> natom = 1
        # Molecules: natom != 1
        self.n_atom = 1

        # Envelope
        self.envelope_up = Envelope(
            self.n_atom, self.n_det * self.n_spin_up)
        self.envelope_down = Envelope(
            self.n_atom, self.n_det * self.n_spin_down)

        # Heads for up/down
        self.orb_up = nn.Linear(self.n_embd, self.n_det*self.n_spin_up)
        self.orb_down = nn.Linear(self.n_embd, self.n_det*self.n_spin_down)
        nn.init.constant_(self.orb_up.bias, 1e-3)
        nn.init.constant_(self.orb_down.bias, 1e-3)

        # Weights for the determinant
        self.det_logits = nn.Parameter(torch.zeros(self.n_det))
```

We are going to take our **hidden feature** $(B,n_{elec},h)$, we are going to split it into two Tensors, one for spin up, and another for spin down, are we are going to give it to  `self.orb_up` and (down), this are going to create the sufficient values to create the `n_det` matrices.   $(B,n_{elec}, n_{\det}\cdot n_{\text{spin up}})$

The lines:
```python
nn.init.constant_(self.orb_up.bias, 1e-3)
nn.init.constant_(self.orb_down.bias, 1e-3)
```

Indicate that the bias of the `Linear` operation are initialized on `1e-3`, this is made it for stability,  recall that without the bias will be initialized on random numbers. 

This is how precisely build the `n_det` determinants:

```python
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
```

This returns a tensor for **up/down**, $(B,n_{\det},\text{spin}, \text{spin})$.  So we would obtain two tensors, one for up and another for down. We need to take the determinant to the matrices of the last two dimension of those tensors, for that matter we are going to use the follow **method**.

```python
def slogdet_sum(self, mats: torch.Tensor) -> torch.Tensor:
        """
        mats: (B, n_det, n_spin, n_spin)
        returns stable sum of determinants
        """
        det_logs = []
        for k in range(self.n_det):
            sign, logs_abs = torch.linalg.slogdet(mats[:, k, :, :])
            det_logs.append(logs_abs)
        det_logs = torch.stack(det_logs, dim=-1)
        return torch.logsumexp(det_logs, dim=-1)
```

And here is a **big trap!**. First since we are working on the **log space**, we are going to take the **logarithm** of the determinant, how the determinant could be negative we take the absolute value.

Finally the **forward** for the `class Orbital_Head` is:

```python
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

        logdet_up = self.slogdet_sum(phi_up)
        logdet_down = self.slogdet_sum(phi_down)
        return logdet_up, logdet_down
```


## Jastrow Term

The Jastrow term focuses on the electron-electron interactions. Google Deep Mind proposes:

$$
\mathcal{J}_{\theta}(\mathbf{x})
=
\sum_{i<j;\,\sigma_{i}=\sigma_{j}}
-\frac{1}{4}\frac{\alpha^{2}_{\mathrm{par}}}{\alpha_{\mathrm{par}}+\lvert \mathbf{r}_{i}-\mathbf{r}_{j} \rvert }
\;+\;
\sum_{i,j;\,\sigma_{i}\neq \sigma_{j}}
-\frac{1}{2}\frac{\alpha^{2}_{\mathrm{anti}}}{\alpha_{\mathrm{anti}}+\lvert \mathbf{r}_{i}-\mathbf{r}_{j} \rvert }.
$$

```python
class Jastrow(nn.Module):
    """
    Docstring for Jastrow:
    """
    def __init__(self, spin_up: int, spin_down: int):
        super().__init__()
        # Learnable Parameters
        self.alpha_anti = nn.Parameter(torch.rand(1))
        self.alpha_par = nn.Parameter(torch.rand(1))

		# Index
        self.spin_up = spin_up
        self.spin_down = spin_down
```

We initialize both parameters sampled from a uniform distribution between zero and one. 

For the sum for electrons with same spin, we continue like follow:

```python
    @staticmethod
    def _same_spin_sum(position: torch.Tensor,
                       coeff: float, alpha: torch.Tensor) -> torch.Tensor:
        """
        position: (B, n, 3) n: up or down
        """
        batch_size, n, _ = position.shape
        if n < 2:
            return position.new_zeros(batch_size)

        # Pairwise distance with broadcasting
        diff = position[:, :, None, :] - position[:, None, :, :]
        dists = diff.norm(dim=-1) + 1e-12

        i, j = torch.triu_indices(n, n, offset=1, device=position.device)
        pair_dists = dists[:, i, j]  # (B, num_pairs)
        terms = coeff * alpha.pow(2) / (alpha+pair_dists)
        return terms.sum(dim=1)
```

The **built in** `torch.triu_indices` return the indices of a triangular upper matrix, and offset for the condition $i<j$.

```python
@staticmethod
    def _diff_spin_sum(up: torch.Tensor, down: torch.Tensor,
                       coeff: float, alpha: torch.Tensor) -> torch.Tensor:
        """
        up: (B, n_up, 3)
        down: (B, n_down, 3)
        returns (B,)
        """
        batch_size, n_up, _ = up.shape
        _, n_down, _ = down.shape

        if n_up == 0 or n_down == 0:
            return up.new_zeros(batch_size)

        diff = up[:, :, None, :] - down[:, None, :, :]
        dists = torch.sqrt(diff.pow(2).sum(dim=-1) + 1e-12)

        pair_dists = dists.reshape(batch_size, -1)
        terms = coeff * alpha.pow(2) / (alpha + pair_dists)
        return terms.sum(dim=1)
```

Here is similar but here we don't need index because we are taking all. Use `reshape` for convert the $n\times n$ matrix, to a $n^{2}$ vector.

Finally the **forward** is:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the antisymmetric Jastrow factor for n electrons.
        x : (B, n_elec, 3)
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.size(1) < 2:
            raise ValueError("Jastrow requires at least two electrons.")

        # Any atom
        up = x[:, :self.spin_up, :]  # (B, up, 3)
        down = x[:, self.spin_up:self.spin_down + self.spin_up, :]  # (B, d, 3)

        same_up = self._same_spin_sum(up, -0.25, self.alpha_par)
        same_down = self._same_spin_sum(down, -0.25, self.alpha_par)

        diff_spin = self._diff_spin_sum(up, down,
                                        coeff=-0.5, alpha=self.alpha_anti)

        return same_up + same_down + diff_spin
```

## Psi former Forward

```python
# Psiformer class
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, n_electron ,3)
        """
        # Flatten any leading dims (e.g., monte_carlo, batch) into batch
        if x.dim() > 3:
            x = x.reshape(-1, x.size(-2), x.size(-1))

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

        logdet = self.orbital_head(
            h, self.spin_up_idx, self.spin_down_idx, r_ae_up, r_ae_down
        )
        # print("sign_det", _sign_det)
        jastrow_term = self.jastrow(x)

        # Guard against singular determinant blocks
        if not torch.isfinite(logdet).all():
            raise ValueError("Non-finite log determinant detected")

        # The _sign_det you can use later for whatever you want.

        log_psi = logdet + jastrow_term

        # log_psi: (B, )
        return log_psi
```

# Formulating the Training

The first step for the training is define well the **Loss Function**.

First the `class Trainer`.

```python
class Trainer():
    def __init__(self, model: PsiFormer, config: Train_Config):
        self.model = model.to(get_device())
        self.config = config
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)
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
```

Here note that we are using the **Adam optimizer**, calling the `class MH`. Which is going to be useful for sampling and train the model.

And the `class Hamiltonian` which is going to be useful for compute the **Hamiltonian** of the system.

A simple method is:

```python
    def log_psi(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_elec, 3)
        if x.device != self.device:
            x = x.to(self.device)
        return self.model(x)
```

Another super important is the method:

```python
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
```

Which improves the GPU's parallelization.

Another classical method is:

```python
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
```

Before to pass to the **main method** `def train(self):`, we are going to explain how the **Metropolis Hasting Algorithm** is do it.

## Metropolis Hasting

The **MH** is initialized like:

```python
class MH():
    """
    Implementation for the Metropolis Hasting (MH) algorithm
    using a gaussian kernel. Returns a list a samples from
    the target distribution.
    We work with the log form!.
    """
    def __init__(self, target: Callable[[torch.Tensor], torch.Tensor],
                 config: Train_Config, n_elec: int,
                 device: torch.device | None = None):
        self.target = target
        self.config = config
        self.n_elec = n_elec
        self.device = device or get_device()
```

Using [[PyTorch Stack]], then you can make `.mean()` and becomes more easy. Another important matter is how you propose your first initial configuration. For instance I do it sampling from the **Normal Distribution**.


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

Position of each electron and protons

So we are going to randomize the position of all the electrons and protons, using a normal distribution. So think life follow:

Spin 

And the spin in something that is making me crazy. We always attach the two possible states to a single position? No, for instance for the Helium atom, we consider just two electrons, one labeled with $\uparrow$ and another with down $\downarrow$. This is:

So you look up your sarrus trick:
$$
1s^{2}
$$
Which means that for the Helium you have one spin up and another spin down, $r^{\uparrow}_{1},r^{\downarrow}_{2}$. But in practice how you work with it. Just concatenate, man. 

$$
\begin{align}
h_{1}^{\uparrow} & =\text{concatenate}(r^{\uparrow}_{1}, \lvert r_{1}^{\uparrow} \rvert ) \\
h^{\downarrow}_{2} & =\text{concatenate}(r^{\downarrow}_{2},\lvert r^{\downarrow}_{2} \rvert ) \\
h^{\uparrow\downarrow}_{12} & =\text{concatenate}(r-r, \lvert  \rvert )
\end{align}
$$

Now for the Litium. $1s^{2}2s^{1}$. So two spin up and just one down.

## Hamiltonian 

The inputs from both comes from the samples that **Metropolis Hasting** make. 

Model the Hamiltonian is a lot of fun in the sense that you have to know something about how the [[PyTorch Computational Graph]] works. Because you are going to need how compute the **Laplacian** from a scalar field, and need to know when the computational graph are going to be liberated and how retained it.

On `hamiltonian.py`, we are going to compute $(\nabla \log\psi)^{2}$ and $(\nabla^{2}\log \psi)$.
So it's going to be quite important take the partial derivatives from the Net output. This is formulated with [[PyTorch Grad]].

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
```

So, how we obtain the `grad_log_psi` and `laplacian_log_psi`?  For the first one is:

```python
    def grad_log_psi(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gradient of log psi with graph retained for higher order derivatives.
        """
        x_req = x.clone().detach().requires_grad_(True)
        y = self.log_psi_fn(x_req)
        (g,) = grad(y, x_req, create_graph=True)
        return g
```

And for the **laplacian**.

```python
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

## Potential

Let's begin with the potential for the Hidrogen atom. I didn't think on fixing the movement of the proton. Another important matter is the units, recall that for distances we are working with the Bohr radius. 

Like also the potential $V_{nn},V_{e n},V_{e e}$. 

The proton position vector are fixed to the zero vector. This have a lot of  sense when dealing with a single electron but when you have more than one, you fixed all of them to the zero vector? That is practical? 

The potential energy that comes from the repulsion between electrons in that case would become zero. For the Hidrogen atom we obtain that:
$$
V=-\frac{1}{\lvert r_{e} \rvert }
$$
Pretty simple, now for the Helium atom, we consider the electron - electron and the two electron proton, and what about the proton. Bohr Oppenheimer tell us that we are consider the nucleous like a single point!
$$
V=+\frac{1}{\lvert r_{e_{1}} -r_{e_{2}}\rvert }-\frac{2}{\lvert R_{e}-r \rvert }- \frac{2}{\lvert R_{e}-r \rvert }
$$

But we are going to assume that $R_{e}$.

```python
    def potential(self) -> torch.Tensor:
        eps = 1e-5
        r_i = torch.linalg.norm(self.coords, dim=-1)  # (B, n_elec)
        nuc_term = -self.Z*(1/(r_i+eps)).sum(dim=-1)

        # Electron-electron repulsion: sum over pairwise distances per sample
        n_elec = self.coords.size(1)
        if n_elec < 2:
            return nuc_term

        diff = self.coords[:, :, None, :] - self.coords[:, None, :, :]
        pairwise_dists = torch.linalg.norm(diff, dim=-1) + eps  # (B, n_elec, n_elec)
        i, j = torch.triu_indices(n_elec, n_elec, offset=1,
                                  device=self.coords.device)
        r_ij = pairwise_dists[:, i, j]  # (B, n_pairs)
        e_e_term = (1 / r_ij).sum(dim=-1)

        # (B, )
        return nuc_term + e
```

##  Training Function

With all this pieces we can make create the function `train`.

```python
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
```

Here the tricky relies on the `.detach()` that you have to make. And how is that the derivative is completely necessary. I mean, the big question that you have to make is: Where you want to back propagate and where not. You want to propagate  on $\log \psi$ the argument of the expectation. You are going to threat $\mathbb{E}(E_{L}(X))$, like constant? If that were the case, you would make a `detach()`.  Or a `with torch.no_grad()`. 

Or more precisely you are going to threat $E_{L}(R)$, like a constant. One of the first reason is because if we are not doing that then higher order derivatives are computed. ($E_{L}$ is computed from the laplacian. From psi), then you need three order higher derivatives. Which is actually a lot of error accumulated. 

- We don't want that, so we treat the samples $\mathbf{R}_{k}$ as constants.
- Treat **local energies** as constants.
- We only use the backward guy to get $\nabla_{\theta}\log(\psi)$. Everything else is like just number that our computational graph does not see.


At the end you are going to make:

```python
loss.backward()
```

# Training Metrics

We are going to use [[Wandb]] for supervision, saving [[PyTorch Checkpoints]] and once that the training is complete send the model to [[Hugging Face]].

The metrics that we are going to send to **Weights and biases** are:

```python
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
```



## Batches
We are introducing an additional dimension $B$. To leverage the parallelization that a GPU can offers. So we need to tweak some stuff. 


## Variational Principle 

It says that the energy from our Ansatz should be greater than the ground state, but we don't observe that, why?

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

Now, the question relies on use many `walkers` is parallel. That is the question of using [Batches](#batches)

Complicated,but it just optimization. So it should to converge with more montecarlo steps. To make the plot we are going to make something 

## Numerical Matters

First we are using the **LogSumExp** trick to avoid underflow overflow, when computing the determinants. In the `class Orbital_Head` inside the method.

This is the more complicated part:
[[Psiformer determinant]]

```python
   def slogdet_sum(self, mats: torch.Tensor) -> torch.Tensor:
        """
        mats: (B, n_det, n_spin, n_spin)
        returns stable sum of determinants
        """
        det_logs = []
        for k in range(self.n_det):
            sign, logs_abs = torch.linalg.slogdet(mats[:, k, :, :])
            det_logs.append(logs_abs)
        det_logs = torch.stack(det_logs, dim=-1)
        return torch.logsumexp(det_logs, dim=-1)
```

But naively do this brings some problems!. Because when training explodes!

```python
     logdet_up, logdet_down = self.orbital_head(
            h, self.spin_up_idx, self.spin_down_idx, r_ae_up, r_ae_down
        )

        if not torch.isfinite(logdet_down):
            print("logdet_down not finite")
        if not torch.isfinite(logdet_up):
            print("logdet_up not finite")
```

So how we are going to fix this? First recall that we are working in the log space! Determinants have sign. How they are computed? No idea.

And we are going to use the follow fact:
$$
\det(A)=\text{sign}A\exp(\log|\det A|)
$$
If we ignore the sign you are lossing antisymetry , Why?
Then you simply multiplicate by the signs. 
But that is just a issue compared to the real problem. For the `Nans`. The question is that when we are going to take the second derivative to the determinants you are going to obtain some problems. Because singular matrices. And that is something that they said on the paper but I completely forget.

So we have to implement the backward sensivity, to try to dissapear those error. But is complicated, because you have to ask, how this fit in our code. I don't have slightly idea.

I mean you don't have to go to the low level, just create a function an use it, don't use the ones that **Pytorch** give you.