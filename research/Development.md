---
date: 2025-10-23 19:32
modified: 2025-11-14 07:36
---
# Development of a Transformed based architecture to solve the Time Independent Many Electron Schrodinger Equation

# Abstract

With accurate solutions to the many-electron Schrodinger equation all the chemistry could be derived from first principles, but analytical treatment is intractable due the intrinsic strong electron-electron correlations, anti symmetry and cusp behavior. Recently, due to its high flexibility deep learning approaches had been applied for this problem, neural wave function models such as FermiNet and PauliNet are two good examples, these have advanced accuracy, yet computational cost and error typically grows steeply with system size, limiting applicability to larger molecules. They also lack of strong architectures designed to capture long-range electronic correlations with scalable attention. In this work I develop the Psiformer a transformer-based ansatz that couples scalable attention with physics-aware structure. Training is formulated within Variational Monte Carlo (VMC), evaluation will be do it by comparing against another traditional methods, I also outline design questions for further improvement, including sparsified/global attention and optimizer choices inspired by recent transformer advances.
# Introduction

The electronic structure problem remains challenging: the wave functions  which describes fully the systems lives in a $3N$-dimensional space ($N$ number of electrons, each one lives on the 3 dimensional space, interesting molecules have around 10 electrons), additionally it must satisfy certain properties due to physical laws plus that the solutions lives on the complex space.

Although the governing laws have been known for a century—see Schrödinger’s formulation of wave mechanics [Schrödinger, 1926]—obtaining practical approximations to the quantum many-body wavefunction remains difficult. Established approaches such as density-functional theory, the Born–Oppenheimer separation, and structured variational ansätze trade generality for tractability by imposing specific functional forms or approximations to correlation. These choices are effective within their regimes but can struggle for strongly correlated systems or scale poorly with electron count. This is where modern learning-based methods enter: instead of fixing the functional form, we learn it, while enforcing essential physics (antisymmetry, cusp behavior, permutation symmetry).

Deep learning has reshaped several scientific domains, from protein structure prediction @jumper2021highly to vision @dosovitskiy2021imageworth16x16words and PDE surrogates @RAISSI2019686. Motivated by these successes, the community has explored neural approaches for quantum many-body problems, seeking accurate and scalable approximations to the many-electron wavefunction. 

Thus neural wavefunction models have emerged as a promising alternative. Architectures such as **FermiNet** and **PauliNet** combine flexible function approximators with determinant structures to respect antisymmetry, improving variational expressivity. However, two practical limitations persist. First, error or compute cost often scales unfavorably with the number of electrons, restricting applicability to larger molecules. Second, mechanisms for **long-range electronic correlation**—central to Coulomb and exchange effects—are typically implicit or expensive to capture, leading to optimization difficulty and brittle generalization.

Transformers offer an appealing direction. Self-attention provides direct, many-to-many interactions among tokens in a single layer, is highly parallelizable, and has empirically favorable scaling behavior in other domains (Natural Language Processing) . For electronic structure, where any electron can interact with any other and electron indices are exchangeable, attention aligns naturally with the physics: it enables global coupling without imposing an arbitrary ordering. The challenge is to embed the right **inductive biases** (distance awareness, spin structure, cusp handling) and to maintain **fermionic antisymmetry** while keeping computational cost under control.

I develop **Psiformer**, a transformer-based variational ansatz for many-electron systems. Psiformer uses self-attention to construct rich per-electron features informed by electron–electron and electron–nucleus descriptors, then imposes antisymmetry explicitly via determinant-based heads. Physics-aware priors (e.g., distance/radial encodings and cusp-motivated embeddings) are incorporated to reduce sample complexity and stabilize training. The optimization is formulated within **Variational Monte Carlo (VMC)** by minimizing the variational energy. 

**Contributions.**
1. A transformer-based neural wavefunction (**Psiformer**) that separates correlation modeling (attention) from fermionic symmetry (determinant heads) while embedding Coulomb-aware priors.
2. A VMC training recipe with practical choices for stability (feature design, damping, and optional natural-gradient preconditioning). 

@vonglehn2023selfattentionansatzabinitioquantum  
@Luo_2019 @Qiao_2020.

The objectives are the follow:
# Objectives

- Obtain a model which is able to approximate the ground state state energy of the carbon atom.
- Compare our model with another state of the art methods to solve the many electrons Schrodinger equation respect the ground state energy.
- Look for future improvements when try to tackle larger molecules. 
# Overview

This work is structured as follow: The theoretical framework introduces the foundations of quantum many-body theory, the structure of the Schrodinger equation for many-bodies like also foundational concepts of Deep Learning that are going to be used in the development of this specific problem.

Section [Number] introduce **Psi Former** a transformer based architecture built upon **Fermi Net**. Section [Number] talk about the Methodology which is going to be used to implement this work.
# Theoretical Framework

The concepts presented in this section provideS the physical and mathematical background for our specific problem, introduces the physics of the wave function like also the necessary deep learning concepts.
## The physics law behind the solution

All the physics and methods related necessary.
### The Schrodinger Equation

The Schrodinger equation was presented in a series of publications made it by Erwin Schrodinger in the year 1926. There we search the complex function $\psi$ called **wave function**, for a non relativistic spinless single particle this function depends on the position of the particle $\mathbf{\vec{r}}$ and time $t$ $(\psi(\mathbf{\vec{r}},t))$, the quantity $\lvert \psi (\mathbf{r},t)\rvert^{2}$ is the **probability density** to find the particle near $\mathbf{r}$ at time $t$.

Guided by de Broglie's discover of the wave particle duality and a very smart intuition Schrodinger proposed the time dependent equation (TDSE):
$$ i\hbar \frac{\partial \psi}{\partial t}=\hat{H}\psi $$
Where $i$ is the complex unit, $\hbar$ is the [[Reduced Planck Constant]] approximate to $1.054571817\dots \times 10^{-34} J\cdot s$ and $\hat{H}$ is a Hermitian linear operator called the **Hamiltonian** which represents the total energy of the system, for a single non relativistic (low energy) particle of mass $m$ in a scalar potential $V(\mathbf{r},t)$.
$$
\hat{H}=\frac{\hat{\mathbf{p}}^{2}}{2m}+V(\mathbf{r},t)
$$
Where $\mathbf{\hat{p}}$ is the Momentum Operator and in the **Position representation** it takes the form of:
$$
\mathbf{\hat{p}}=-i\hbar \nabla
$$
Where $\nabla$ is the Laplacian operator, Thus the TDSE is explicitly:
$$
i\hbar\,\frac{\partial \psi}{\partial t}
=
\left[-\frac{\hbar^{2}}{2m}\nabla^{2}+V(\mathbf r,t)\right]\psi
$$
The **time independent form** (TISE) could be derived from equation ,when the wave function $\psi$ could be written like the product of two functions $R$ and $T$, where $R$ depends uniquely on the spatial term $(\mathbf{\vec{r}})$ and $T$ uniquely on the temporal $(t)$, this is:
$$
\psi(\mathbf{\vec{r}},t)=R(\mathbf{\vec{r}})T(t)
$$
Plugin this form on equation **number**, you can derive that:
$$
T(t)=e^{ -iEt/\hbar }
$$
Where $E$,the energy of the system, is constant. You also obtain that the spatial part is conditioned by:
$$
\hat{H}R(\mathbf{\vec{r}})=ER(\mathbf{\vec{r}})
$$
We usually represent $R$ as $\psi$.
### The many electron Schrodinger Equation
When we are considering more than one single particle we consider the spin $(\sigma)$ and the interaction between particles. Thus in its time-independent form the Schrodinger equation can be written as an eigen value problem:
$$ \hat{H}\psi(\mathbf{x}_{0},\dots ,\mathbf{x}_{n})=E\psi(\mathbf{x}_{1},\dots ,\mathbf{x}_{n}) $$
Where $\mathbf{x}_{i}=\{ \mathbf{r}_{i},\sigma \}$,  $\mathbf{r}_{i}\in \mathbb{R}^{3}$ is the position of each particle and $\sigma \in \{ \uparrow.\downarrow \}$ is the so called spin. It's possible model the potential energy of a many body system (e.g atoms, molecules), we first have to consider the repulsion between each electrons:
$$
V_{ij}
= \frac{e^{2}}{4\pi\varepsilon_{0}}
  \frac{1}{\lvert \mathbf{r}_{i}-\mathbf{r}_{j} \rvert}
$$
Here:
- $e = 1.602\,176\,634\times 10^{-19}\ \mathrm{C}$ is the elementary charge,
- $\varepsilon_{0} = 8.854\,187\,8128\times 10^{-12}\ \mathrm{F\,m^{-1}}$ is the **electrical permittivity of vacuum**,
- $\mathbf{r}_i$ is the position vector of electron $i$ in the chosen reference frame.
The attraction between protons and electrons is given by:
$$
V_{iI} = -\frac{1}{4\pi\varepsilon_{0}} \frac{eZ_{I}}{\lvert \mathbf{r}_{i} - \mathbf{R}_{I} \rvert}
$$
Where $Z_I$ is the atomic number of nucleus $I$ (for instance, in a Helium atom $Z = 2$) and $\mathbf{R}_I$ is the position of that nucleus from a chosen reference frame.  
The reference frame is usually taken at the **center of mass** or at the **center of the molecule**. The repulsion between nuclei (protons) is:
$$
V_{IJ} = \frac{1}{4\pi\varepsilon_{0}} \frac{Z_{I}Z_{J}}{\lvert \mathbf{R}_{I} - \mathbf{R}_{J} \rvert}
$$
To avoid writing these constants every time, quantum chemistry commonly uses **atomic units (a.u.)**.  In this system, the unit of length is the **Bohr radius** $a_{0}$,  
and the unit of energy is the **Hartree** $E_{h}$:
$$
E_{h} = \frac{e^{2}}{4\pi\varepsilon_{0} a_{0}}
$$
Under atomic units, $e = 1$, $4\pi\varepsilon_{0} = 1$,$\hbar = 1$, and $m_{e} = 1$.  
Thus, the potential energy of a multi-electron, multi-nucleus system can be written compactly as:
$$
V = -\sum_{i,I} \frac{Z_I}{\lvert \mathbf{r}_i - \mathbf{R}_I \rvert}
+ \sum_{i>j} \frac{1}{\lvert \mathbf{r}_i - \mathbf{r}_j \rvert}
+ \sum_{I>J} \frac{Z_I Z_J}{\lvert \mathbf{R}_I - \mathbf{R}_J \rvert}
$$

where:
- $i, j$ index electrons,
- $I, J$ index nuclei,
- the first term represents **electron–nucleus attraction**,  
- the second term is **electron–electron repulsion**,  
- the third term is **nucleus–nucleus repulsion**.

For the kinetic term we need to consider two expressions: $\nabla_i^2$ acts on the **electron** coordinates $\mathbf r_i$ (fast, light particles) and $\nabla_I^2$ acts on the **nuclear** coordinates $\mathbf R_I$ (slow, heavy particles). 
Let $N$ electrons at $\mathbf r_i$ and $M$ nuclei at $\mathbf R_I$ with charges $Z_I$ and masses $M_I$ (in units of $m_e$). Thus the Hamiltonian can be written as:
$$
\boxed{
\hat H =
-\sum_{i=1}^{N}\frac{1}{2}\nabla_i^{2}
-\sum_{I=1}^{M}\frac{1}{2M_I}\nabla_{I}^{2}
-\sum_{i=1}^{N}\sum_{I=1}^{M}\frac{Z_I}{|\mathbf r_i-\mathbf R_I|}
+\sum_{1\le i<j\le N}\frac{1}{|\mathbf r_i-\mathbf r_j|}
+\sum_{1\le I<J\le M}\frac{Z_I Z_J}{|\mathbf R_I-\mathbf R_J|}
}
$$
### Conditions of the solution

As with any differential equation, where one is searching one solution is important to consider initial conditions (IC) and boundary conditions (BC), here we have to fulfill certain conditions that comes from physical laws.
#### Fermi–Dirac statistics and Pauli Exclusion
Bosons (e.g., photons) follow **Bose–Einstein** statistics, whereas fermions (e.g., electrons, protons) follow **Fermi–Dirac** statistics, because at the microscopic level identical particles are indistinguishable then the wave function should be equal but Pauli Exclusion don't like that , the many-particle wave function must be **anti symmetric** under the exchange of any two fermions. This implies:
$$
\psi(\dots,\mathbf{x}_{i},\dots,\mathbf{x}_{j},\dots)=-\psi(\dots ,\mathbf{x}_{j},\dots ,\mathbf{x}_{i},\dots)
$$
We can enforce **antisymmetry** using a $N\times N$ determinant, which involves one-particle states only (a wave function with a single input) . An interchange of any pair of particles corresponds to an interchange of two columns of the determinant; this interchange introduces a change in the sign of the determinant. For even permutations we have $(-1)^{P}=1$, and for odd permutations we have $(-1)^{P}=-1$. [[Fermi Dirac Statistics]]

$$
\Psi(\mathbf x_1,\ldots,\mathbf x_N)
\propto
\begin{vmatrix}
\phi_1(\mathbf x_1) & \phi_2(\mathbf x_1) & \cdots & \phi_N(\mathbf x_1)\\
\phi_1(\mathbf x_2) & \phi_2(\mathbf x_2) & \cdots & \phi_N(\mathbf x_2)\\
\vdots & \vdots & \ddots & \vdots\\
\phi_1(\mathbf x_N) & \phi_2(\mathbf x_N) & \cdots & \phi_N(\mathbf x_N)
\end{vmatrix},
$$
Where $\phi_k$ are orthonormal (by construction) [[spin-orbital|spin orbitals]]. This is, for instance $(N=2)$:
$$
\Psi(\mathbf x_1,\mathbf x_2)
\propto\Big[\phi_a(\mathbf x_1)\phi_b(\mathbf x_2)-\phi_a(\mathbf x_2)\phi_b(\mathbf x_1)\Big].
$$
#### Kato Cusp Conditions

The potential energy becomes infinite, when particles overlap, which places strict constraints on the form of the wave function at these points, knows as the **Kato Cusp Conditions** [cite]. The cusp conditions states that the wave function must be non-differentiable at these points, and give exact values for the average derivatives at the cusps. This can be obtained if we multiply to the ansatz by multiplying by a Jastrow factor  $\mathcal{J}$which satisfies these conditions analytically. [cite].
$$\lim_{ l \to 0 } \left( \frac{\partial \psi}{\partial r_{iI}} \right)=-Z\psi(r_{iI}=0)
$$
$$
\lim_{ l \to 0 } \left( \frac{\partial \psi}{\partial r_{ij}} \right)=\frac{1}{2}\psi(r_{ij}=0)
$$
Where $r_{iI}(r_{ij})$ is an electron-nuclear (electron-electron) distance, $Z_{I}$ is the nuclear charge of the $I\text{-th}$ nucleous and ave implies a spherical averaging over all directions.

[[Kato Cusp Conditions]].
### Approximations to the problem

Is clear that find analytical solutions is practically impossible, so what people have been doing is first apply good approximations.
[[The Born Oppenheimer Approximation]]
Born Oppenheimer approximation makes it possible to separate the motion of the nuclei and the motion of the electrons, neglects the motion of the atomic nuclei when describing the electrons in a molecule. The physical basis for the Born-Oppenheimer approximation is the fact that the mass of an atomic nucleus in a molecule is much larger than the mass of an electron (more than 1000 times). Because of this difference, the nuclei move much more slowly than the electrons. In addition, due to their opposite charges, there is a mutual attractive force of: acting on an atomic nucleus and an electron. This force causes both particles to be accelerated. Since the magnitude of the acceleration is inversely proportional to the mass, $a = f/m$, the acceleration of the electrons is large and the acceleration of the atomic nuclei is small; the difference is a factor of more than 1000. Consequently, the electrons are moving and responding to forces very quickly, and the nuclei are not:
$$
\boxed{
\hat H_{\mathrm{el}} =
-\sum_{i=1}^{N}\frac{1}{2}\nabla_i^{2}
-\sum_{i=1}^{N}\sum_{I=1}^{M}\frac{Z_I}{|\mathbf r_i-\mathbf R_I|}
+\sum_{1\le i<j\le N}\frac{1}{|\mathbf r_i-\mathbf r_j|}
}
\qquad\text{with}\qquad
E_{\mathrm{nn}}=\sum_{I<J}\frac{Z_I Z_J}{|\mathbf R_I-\mathbf R_J|}.
$$

A second useful approach to the problem is use an **Ansatz**:

guess that solution and using another techniques to improve the solution, to this guess solution we called , guided by intuition. 
Once that you have your Ansatz, which normally depends on depends on certain parameters.

We are going to use a **Ansatz**:
### Optimizing the Ansatz
[[Rayleigh Quotient like Expectation Value]]
How we know that our Ansatz is good? Equivanlently to use a loss function in quantum mechanics we use **rayleight quotient**.
$$
\mathcal{L}(\Psi_{\theta})=\frac{\bra{\Psi_{\theta}} \hat{H}\ket{\Psi_{\theta}} }{\braket{ \Psi_{\theta} | \Psi _{\theta}} }=\frac{\int d\mathbf{r}\Psi ^{*}(\mathbf{r})\hat{H}\Psi(\mathbf{r})}{\int d\mathbf{r}\Psi ^{*}(\mathbf{r})\Psi(\mathbf{r})}
$$
Evaluate that integral is hard, for that first we define:
$$
p_{\theta}(x)=\frac{\Psi^{2}_{\theta}(x)}{\int dx'\Psi^{2}_{\theta}(x')}
$$
Defining the Local Energy:
$$
E_{L}(x)=\Psi ^{-1}_{\theta}(x)\hat{H}\Psi_{\theta}(x)
$$
$$
\mathcal{L}(\Psi_{\theta})=\int \frac{ \hat{H}\Psi_{\theta}(x)}{\Psi_{\theta}(x)}p_{\theta}(x)dx
$$
$$
\mathcal{L}(\Psi_{\theta})=\mathbb{E}_{x\sim p_{\theta}}[E_{L}(x)]
$$
To optimize our wave function we need evaluate that expectation and obtain the derivative.

To that set of techniques for obtain that we call **Quantum Monte Carlo**, there are differents approaches like Diffusion Monte Carlo, Path Integral Monte Carlo but in this case we are going to use Variational Montecarlo.

Samples from that distribution and $E_{L}(x)$. We already have $E_{L}(x)$ 

Whith many samples we can estimate:
$$
\mathbf{R}_{1},\dots,\mathbf{R}_{M}\sim p_{\theta}(\mathbf{R})
$$
Using the Monte Carlo estimates:
$$
\mathcal{L}_{\theta}=\mathbb{E}_{x\sim p_{\theta}}[E_{L}(x)]\approx \frac{1}{M}\sum_{i=1}^{M} E_{L}(\mathbf{R}_{k})
$$
Where we can write:
$$
E_{L}(\mathbf{R}_{k})=\frac{\hat{H}\psi(\mathbf{R}_{k})}{\psi(\mathbf{R}_{k})}=-\frac{1}{2}\frac{\nabla^{2}\psi(\mathbf{R_{k}})}{\psi(\mathbf{R}_{k})}+V(\mathbf{R}_{k})
$$
Calculus tell us:

$$
\frac{\nabla^{2}f}{f}=[\nabla^{2}\log f+(\nabla f)^{2}]
$$

Which is more numerically stable, thus

$$
E_{L}(\mathbf{R}_{k})=-\frac{1}{2}\sum_{i=1}^{N} \sum_{j=1}^{3} \left[ \frac{\partial ^{2} \log \lvert \Psi(x) \rvert }{\partial r_{ij}^{2}}+\left( \frac{\partial \log \lvert \Psi(x) \rvert }{\partial r_{ij}} \right)^{2} \right]+V(\mathbf{R_{k}})
$$

The gradient of the energy respect to the parameters by a parameterized wave functions is:
$$
\nabla _{\theta}\mathbb{E}_{x\sim \Psi}^{2}[E_{L}(x)]=2\mathbb{E}_{x\sim \Psi^{2}}[(E_{L}(x)-\mathbb{E}_{x'\sim\Psi^{2}}[E_{L}(x')])\nabla \log \lvert \Psi(x) \rvert ]
$$
### Metropolis Hastings (MH) Algorithm

Goal: obtain many samples from  [[Metropolis Hasting algorithm]]

MH is a [[Markov Chain Monte Carlo]] (MCMC) method used to obtain a sequence of random samples from a probability distribution. Sota method to sample from high dimensional distributions.

## Deep Learning Fundamentals

This subsection introduces the core concepts of  Deep Learning that are going to be applied in this work.
### Multi Layer Perceptron
A multi layer perceptron (MLP) is a nonlinear function $\mathcal{F}:\mathbb{R}^{\text{in}}\to \mathbb{R}^{\text{out}}$.  @nielsenNeuralNetworksDeep2015 , it actually is the composition of $L$ layers, the first layer is called the input layer, the last output layer and the intermediates hidden layers. In each layer we find an affine map $\mathbf{z}^{(l)},l\in \{ L,L-1,\dots,2 \}$ of the follow form. 
$$
\mathbf{z}^{(l)}=\mathbf{W}^{(l)}\mathbf{a}^{(l-1)}+\mathbf{b}^{(l)}
$$
Where $\mathbf{W}^{(l)}$ called weight matrix and $\mathbf{b}^{(l)}$ the bias vector of the $l$ layer.  We use a non-linear function $\sigma ^{(l)}$ in the $l$ layer (typically Softmax,  ReLu, Tanh), thus the output of each layer is:
$$ f^{(l)}=\sigma ^{(l)}\circ \mathbf{z}^{(l)} $$
Where $\circ$ means composition. A MLP is the composition of all the layers.
$$
\mathcal{F}=f^{(L)}\circ f^{(L-1)}\circ\dots \circ f^{(1)}
$$
We call parameters to the set of all the weights and bias of each layer. And represented it with the symbol $\theta$.
$$\{ \mathbf{W}^{(l)},\mathbf{b}^{(l)}\}_{l=2}^{L}=\theta$$
You typically train a MLP, using a training data set, a loss function and an optimizer. Additionally you can use regularization techniques to improve the performance of the MLP.
### Natural gradient Descent

We need this topic because our optimizer use it. There exist different methods to update our parameters. Like Gradient Descent, Stochastic, [[Adaptive Moment Estimation]] ADAM, but in this work we are going to use gone completely different. 

$$
\Delta \theta _{\text{nat}}=-\eta \mathcal{F}^{-1} \Delta_{\theta}\mathcal{L}
$$
Where $\mathcal{F}$ is the Fisher Information Matrix (FIM) defined as:
$$ \mathcal{F}_{ij}=\mathbb{E}_{p}(\mathbf{x})\left[ \frac{\partial \log p(x)}{\partial \theta_{i} }\frac{\partial \log p(X)}{\partial \theta_{j}} \right] $$

[[Natural Gradient Descent]] 
### Kronecker Factored Approximate Curvature

[[Kroenecker factored Approximate Curvature]]
Find the [[Fisher Information Matrix]] analiticaly becomes very hard for that matter we have two approximations.
1. $\mathcal{F_{ij}}$ are assumed to be zero when $\theta_{i}$ and $\theta_{j}$ are in different layers of the network.
2. The other approximation is the follow:
$$
\mathbb{E}_{p(\mathbf{X})}\left[ \frac{\partial \log p(X)}{\partial \text{vec}(\mathbf{W}_{\ell})}\frac{\partial \log p(X)}{\partial \mathbf{W}_{\ell}}^{\mathsf{T}} \right]=\mathbb{E}_{p(\mathbf{X})}[(\mathbf{a}_{\ell}\otimes \mathbf{e}_{\ell})(\mathbf{a}_{\ell}\otimes \mathbf{e}_{\ell})^{\mathsf{T}}]
$$
Where $\mathbf{a}_{\ell}$ are the forward activation and $\mathbf{e}_{\ell}$ are the backward sensitivities for that layer
Approx:
$$
\mathbb{E}_{p(\mathbf{X})}[(\mathbf{a}_{\ell}\otimes \mathbf{e}_{\ell})(\mathbf{a}_{\ell}\otimes \mathbf{e}_{\ell})^{\mathsf{\top}}]^{-1}\approx \mathbb{E}_{p(\mathbf{X})}[\mathbf{a}_{\ell}\mathbf{a_{\ell}}^{\mathsf{\top}}]^{-1}\otimes \mathbb{E}_{p(\mathbf{X})}[\mathbf{e}_{\ell}\mathbf{e}_{\ell}^{\mathsf{\top}}]^{-1}
$$
We specifically we are going to use: 
$$
\mathbb{E}_{p}(\mathbf{X})\left[ \frac{\partial \log p(\mathbf{X})}{\partial \text{vec}(\mathbf{W}_{\ell})}\frac{\partial \log p(\mathbf{X})^{\mathsf{\top}}}{\partial \text{vec}(\mathbf{W}_{\ell})} \right]\approx \mathbb{E}_{p(\mathbf{X})}[\mathbf{\hat{a}}_{\ell}\mathbf{\hat{a}_{\ell}}^{\mathsf{\top}}]^{-1}\otimes \mathbb{E}_{p(\mathbf{X})}[\mathbf{\hat{e}_{\ell}}\mathbf{\hat{e}}_{\ell}^{\mathsf{\top}}]^{-1}
$$

[[Kroenecker factored Approximate Curvature]]

### Recurrent Neural Networks

To introduce the Transformer architecture, first let's introduce the problem they solve.

MLP doesn't work to well processing sequentially data (e.g text) they tend to forget, thus we need another approach to increase the memory of the Neural Network, for this we can tweak life follow:

$$
\mathbf{a}^{(t)}=\mathbf{b}+\mathbf{W}\mathbf{h}^{(t-1)}+\mathbf{U}x^{(t)}
$$
But do it this ways introduce the well known problem of vanishing gradients.
Another approach to increase the memory of neural networks are LSTMs

### Long Short Term Memory

LSTMs introduce a memory cell $c_{t}$ and gates that regulate information flow.

LSTMs substantially improve sequence modeling across modalities.

This increase the memory of neural networks but still sequencial, we have a better approach, which are another architecture based on a mechanism called attention.  


### Attention Mechanism

The first attention mechanism were introduced by using the follow:

### Self Attention and Multi Head Attention

Here we use the dot product and heads. 
$$
\mathbf{o}_{t,i}=\sum_{j=1}^{t}\text{Softmax}\left( \frac{\mathbf{q}^{T}_{t,i}\mathbf{k}_{j,i}}{\sqrt{ d_{h} }} \right) \mathbf{v}_{j,i}
$$
$$
\mathbf{u}_{t}=W^{O}[\mathbf{o}_{t,1};\mathbf{o}_{t,2};\dots ;\mathbf{o}_{t,n_{h}}]
$$
### Transformers architecture

@Vaswani2017 
There exist several architectures that I can use Recurrent Neural Network, Long Short Term Memory. 

Recurrent Neural Network are: [[Recurrent Neural Network]]
And long short term memory are: [[Long Short Memory]]

Why on earth I would use [[Transformer]]? They are extremely good finding relations between its elements. And the best is that scale well due its [[Transform Architecture]]

Attention mechanism appear with @bahdanau2014neural but it didn't work so:

- [[Attention mechanism]]
- [[Self attention mechanism on one head]]
- [[Multi-head attention]]

# Psi Former

## Fermi Net
A very important work for us is: Fermi Net @Pfau_2020  it uses different MLP to learn the forms of the orbitals. Their ansatz is: [[Fermi Net]]

$$ \psi(\mathbf{x}_{i},\dots,\mathbf{x}_{n})=\sum_{k}\omega_{k}\det[\Phi ^{k}] $$
With:
$$
\begin{vmatrix}
\phi_{1}^{k}(\mathbf{x}_{1})  & \dots  &  \phi_{1}^{k}(\mathbf{x}_{n}) \\
\vdots   &  & \vdots  \\
\phi_{n}^{k}(\mathbf{x}_{1}) & \dots & \phi_{n}^{k}(\mathbf{x}_{n})

\end{vmatrix}=\det[\phi_{i}^{k}(\mathbf{x}_{j})]=\det[\Phi ^{k}]
$$

The elements of the determinant are obtained via [[Obtaining the orbital fermi net flow]]

$\alpha \in \{ \uparrow,\downarrow \}$

$$
\mathbf{h}_{i}^{\ell \alpha} \gets \text{concatenate}(\mathbf{r}^\alpha_i - \mathbf{R}_I, |\mathbf{r}^\alpha_i - \mathbf{R}_I|\ \forall\ I)
$$
$$
\mathbf{h}_{ij}^{\ell \alpha\beta} \gets \text{concatenate}(\mathbf{r}^\alpha_i - \mathbf{r}^\beta_j, |\mathbf{r}^\alpha_i - \mathbf{r}^\beta_j|\ \forall\ j,\beta)
$$

$$
 \begin{align}
    &\left(
    \mathbf{h}^{\ell\alpha}_i,
    \frac{1}{n^\uparrow}\sum_{j=1}^{n^\uparrow} \mathbf{h}^{\ell\uparrow}_j, \frac{1}{n^\downarrow} \sum_{j=1}^{n^\downarrow} \mathbf{h}^{\ell\downarrow}_j,
    \frac{1}{n^\uparrow} \sum_{j=1}^{n^\uparrow} \mathbf{h}^{\ell\alpha\uparrow}_{ij},
    \frac{1}{n^\downarrow} \sum_{j=1}^{n^\downarrow} \mathbf{h}^{\ell\alpha\downarrow}_{ij}\right) \nonumber \\
    &\qquad =
    \left(\mathbf{h}^{\ell\alpha}_i, \mathbf{g}^{\ell\uparrow}, \mathbf{g}^{\ell\downarrow}, \mathbf{g}^{\ell\alpha\uparrow}_i, \mathbf{g}^{\ell\alpha\downarrow}_i\right) = \mathbf{f}^{\ell \alpha}_i,
\end{align}
$$


$$
\begin{align}
    \mathbf{h}^{\ell+1 \alpha}_i &= \mathrm{tanh}\left(\mathbf{V}^\ell \mathbf{f}^{\ell \alpha}_i + \mathbf{b}^\ell\right) + \mathbf{h}^{\ell\alpha}_i \nonumber \\
    \mathbf{h}^{\ell+1 \alpha\beta}_{ij} &= \mathrm{tanh}\left(\mathbf{W}^\ell\mathbf{h}^{\ell \alpha\beta}_{ij} + \mathbf{c}^\ell\right) + \mathbf{h}^{\ell \alpha\beta}_{ij}
\end{align}
$$

$$
\begin{align}
    \phi^{k\alpha}_i(\mathbf{r}^\alpha_j; \{\mathbf{r}^\alpha_{/j}\}; \{\mathbf{r}^{\bar{\alpha}}\}) =
    \left(\mathbf{w}^{k\alpha}_i \cdot \mathbf{h}^{L\alpha}_j + g^{k\alpha}_i\right)\\
	\sum_{m} \pi^{k\alpha}_{im}\mathrm{exp}\left(-|\mathbf{\Sigma}_{im}^{k \alpha}(\mathbf{r}^{\alpha}_j-\mathbf{R}_m)|\right),
\end{align}
$$

$$ \phi ^{k\alpha}_{i}(\mathbf{r}^{\alpha}_{j};\{ \mathbf{r}^{\alpha}_{/j} \};\{ \mathbf{r}^{\bar{\alpha}} \})=(\mathbf{w}^{k\alpha}_{i}\cdot \mathbf{h}^{L\alpha}_{j}+g^{k\alpha}_{i})\sum_{m}\pi_{im}^{k\alpha}\exp\left( -\left\lvert \Sigma _{im}^{k\alpha}(\mathbf{r}^{\alpha}_{j}-\mathbf{R}_{m})\right\rvert  \right)$$.

$$
​￼\begin{align}
	\psi(\mathbf{r}^\uparrow_1,\ldots,\mathbf{r}^\downarrow_{n^\downarrow}) = \sum_{k}\omega_k &\left(\det\left[\phi^{k \uparrow}_i(\mathbf{r}^\uparrow_j; \{\mathbf{r}^\uparrow_{/j}\}; \{\mathbf{r}^\downarrow\})\right]\right.\\&\left.\hphantom{\left(\right.}\det\left[\phi^{k\downarrow}_i(\mathbf{r}^\downarrow_j; \{\mathbf{r}^\downarrow_{/j}\});
	\{\mathbf{r}^\uparrow\};\right]\right).
\end{align}
$$


![[ferminet.png|280x315]]

Until this point we have only use MLPs vanilla. 
[[Psi Former Ansatz]]. @vonglehn2023selfattentionansatzabinitioquantum
$$ \Psi_{\theta}(\mathbf{x})=\exp(\mathcal{J}_{\theta}(\mathbf{x}))\sum_{k=1}^{N_{\det}}\det[\boldsymbol{\Phi}_{\theta}^{k}(x)] $$

Where $\mathcal{J}_{\theta}$ is the [[Jastrow Factor for si Former]] and $\Phi$ are [[Orbital for neural network fermi net|orbitals]]. 


Where $\mathcal{J}_{\theta}:(\mathbb{R}^{3}\times \{ \uparrow,\downarrow \})^{n}\to \mathbb{R}$

- So the question is how you define the outputs of that functions:
- [[Jastrow Factor]]
$$
\mathcal{J}_{\theta}(x)=\sum_{i<j;\sigma_{i}=\sigma_{j}}-\frac{1}{4}\frac{\alpha^{2}_{par}}{\alpha_{par}+\lvert \mathbf{r}_{i}-\mathbf{r}_{j} \rvert }+\sum_{i,j;\sigma_{i}\neq \sigma_{j}}-\frac{1}{2}\frac{\alpha^{2}_{anti}}{\alpha_{anti}+\lvert \mathbf{r}_{i}-\mathbf{r}_{j} \rvert }
$$

Architecture

![[psiformer.png|271x339]]

## Applying Attention to Fermi Net

First compute:
$$ v_{h}=[\text{SelfAttn}(\mathbf{h}^{l}_{1},\dots,\mathbf{h}^{\ell}_{N};\mathbf{W}^{\ell h}_{q},\mathbf{W}^{\ell h}_{k},\mathbf{W}^{\ell h}_{v})] $$

Start with:

$$\mathbf{W}_{o}^{\ell}\text{concat}_{h}[\text{SelfAttn}(\mathbf{h}^{l}_{1},\dots,\mathbf{h}^{\ell}_{N};\mathbf{W}^{\ell h}_{q},\mathbf{W}^{\ell h}_{k},\mathbf{W}^{\ell h}_{v})]$$

With it you can obtain you hidden states, and then how you use it



With them you create the [[Orbital for neural network fermi net]]

And you have it.

# Methodology

To implement the code, the choose of the library is important.

The three options to implement this kind of matter are JAX, Tensor Flow and pytorch, each one with his advantages and disadvantages.

## Environment

For this project we are going to be using Pytorch due his user-friendly and support. Python. with UV

## Training

Due the high computational power needed we are going to using GPUS and of course CUDA.

Is clear that we are going to use virtual GPUS, for that matter we have two option or well use a GPU via SSH or directly using services like Azure , Colab, or anothers matters.

The election of the GPU is not trivial. use TPUS are not a bad idea.

## References

Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate. _arXiv Preprint arXiv:1409.0473_.

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., & Houlsby, N. (2021). _An image is worth 16x16 words: Transformers for image recognition at scale_. [https://arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)

Jumper, J., Evans, R., Pritzel, A., Green, T., Figurnov, M., Ronneberger, O., Tunyasuvunakool, K., Bates, R., Žı́dek, A., Potapenko, A., & others. (2021). Highly accurate protein structure prediction with AlphaFold. _Nature_, _596_(7873), 583–589.

Luo, D., & Clark, B. K. (2019). Backflow transformations via neural networks for quantum many-body wave functions. _Physical Review Letters_, _122_(22). [https://doi.org/10.1103/physrevlett.122.226401](https://doi.org/10.1103/physrevlett.122.226401)

Pfau, D., Spencer, J. S., Matthews, A. G. D. G., & Foulkes, W. M. C. (2020). Ab initio solution of the many-electron Schrödinger equation with deep neural networks. _Physical Review Research_, _2_(3). [https://doi.org/10.1103/physrevresearch.2.033429](https://doi.org/10.1103/physrevresearch.2.033429)

Qiao, Z., Welborn, M., Anandkumar, A., Manby, F. R., & Miller, T. F. (2020). OrbNet: Deep learning for quantum chemistry using symmetry-adapted atomic-orbital features. _The Journal of Chemical Physics_, _153_(12). [https://doi.org/10.1063/5.0021955](https://doi.org/10.1063/5.0021955)

Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. _Journal of Computational Physics_, _378_, 686–707. [https://doi.org/10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045)

Shang, H., Guo, C., Wu, Y., Li, Z., & Yang, J. (2025). Solving the many-electron Schrödinger equation with a transformer-based framework. _Nature Communications_, _16_(1), 8464. [https://doi.org/10.1038/s41467-025-63219-2](https://doi.org/10.1038/s41467-025-63219-2)

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention is all you need. _Advances in Neural Information Processing Systems (NeurIPS)_, _30_.

von Glehn, I., Spencer, J. S., & Pfau, D. (2023). _A self-attention ansatz for ab-initio quantum chemistry_. [https://arxiv.org/abs/2211.13672](https://arxiv.org/abs/2211.13672)

[^1]: Schrodinger Reference.

--- 

# Excerpt

Transformers are monsters finding relations between what you give them. Is tempting use them for emulate the relations between electrons and protons. How you can first encode the information of the electron's positions and second the attraction and repulsion between the particles? 