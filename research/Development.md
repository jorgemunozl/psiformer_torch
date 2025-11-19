---
date: 2025-10-23 19:32
modified: 2025-11-19 17:04
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

The Schrodinger equation was presented in a series of publications made it by Erwin Schrodinger in the year 1926. There we search the complex function $\psi$ which lives on a Hilbert space $\mathcal{H}$ called **wave function**, for a non relativistic spinless single particle this function depends on the position of the particle $\mathbf{\vec{r}}$ and time $t$ $(\psi(\mathbf{\vec{r}},t))$, the quantity $\lvert \psi (\mathbf{r},t)\rvert^{2}$ is the **probability density** to find the particle near $\mathbf{r}$ at time $t$.

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
Where $\nabla$ is the Laplacian operator, thus the TDSE is explicitly:
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
We are going to represent the spatial function  $R$ as $\psi$. And in this work we are going to only focus in the TDSE, when we treat with constant energy, the electron are almost always found near the lowest energy state, known as the ground state. Solutions with higher energy known as excited states are relevant to photochemistry , but in this work we will restrict our attention to ground states.
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
The attraction between the proton $I$ and the electron $i$ is given by:
$$
V_{iI} = -\frac{1}{4\pi\varepsilon_{0}} \frac{eZ_{I}}{\lvert \mathbf{r}_{i} - \mathbf{R}_{I} \rvert}
$$
Where $Z_I$ is the atomic number of nucleus $I$ (for instance, in a Helium atom $Z = 2$) and $\mathbf{R}_I$ is the position of that nucleus from a chosen reference frame.  
The reference frame is usually taken at the **center of mass** or at the **center of the molecule**. The repulsion between the nuclei $I$ and $J$ (protons) is:
$$
V_{IJ} = \frac{1}{4\pi\varepsilon_{0}} \frac{Z_{I}Z_{J}}{\lvert \mathbf{R}_{I} - \mathbf{R}_{J} \rvert}
$$
To avoid writing these constants every time, quantum chemistry commonly uses **atomic units (a.u.)**.  In this system, the unit of length is the **Bohr radius** $a_{0}$,  
and the unit of energy is the **Hartree** $E_{h}$:
$$
E_{h} = \frac{e^{2}}{4\pi\varepsilon_{0} a_{0}}
$$
Under atomic units, $e = 1$, $4\pi\varepsilon_{0} = 1$,$\hbar = 1$, and $m_{e} = 1$.  
Thus, the potential energy of a system with $N_{e}$ electron and $N_{n}$ nucleus can be written compactly as:
$$
V = -\sum_{i=1}^{N_{e}}\sum_{I=1}^{N_{n}}  \frac{Z_I}{\lvert \mathbf{r}_i - \mathbf{R}_I \rvert}
+ \sum_{i>j} \frac{1}{\lvert \mathbf{r}_i - \mathbf{r}_j \rvert}
+ \sum_{I>J} \frac{Z_I Z_J}{\lvert \mathbf{R}_I - \mathbf{R}_J \rvert}
$$

where:
- $i, j$ index electrons.
- $I, J$ index nuclei.
- The first term represents **electron–nucleus attraction**.
- The second term is **electron–electron repulsion**.
- The third term is **nucleus–nucleus repulsion**.
For the kinetic term we need to consider two expressions: $\nabla_i^2$ acts on the **electron** coordinates $\mathbf r_i$ (fast, light particles) and $\nabla_I^2$ acts on the **nuclear** coordinates $\mathbf R_I$ (slow, heavy particles). Let $N$ electrons at $\mathbf r_i$ and $M$ nuclei at $\mathbf R_I$ with charges $Z_I$ and masses $M_I$ (in units of $m_e$) then the **Hamiltonian** can be written as:
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
- Electron-nucleus cusp (electron with charge $-1$, nucleus charge $+Z$, reduced mass $\mu \approx 1$)
$$\lim_{ riI \to 0 } \left( \frac{\partial \psi}{\partial r_{iI}} \right)=-Z\psi(r_{iI}=0)
$$
- Electron-electron cusp, opposite spins (charges $-1$, reduced mass $\mu=\frac{1}{2}$)
$$
\lim_{ r_{ij}\to 0 } \left( \frac{\partial \psi}{\partial r_{ij}} \right)=\frac{1}{2}\psi(r_{ij}=0)
$$
Where $r_{iI}(r_{ij})$ is an electron-nuclear (electron-electron) distance, $Z_{I}$ is the nuclear charge of the $I\text{-th}$ nucleous and ave implies a spherical averaging over all directions.
[[Kato Cusp Conditions]].

So that is the problem we need to find a $\psi$ such that it satisfies all those 
### Approximations to the problem

Is clear that find analytical solutions is practically impossible, so what people have been doing is first apply good approximations.

**Born Oppenheimer** approximation makes it possible to separate the motion of the nuclei and the motion of the electrons, neglects the motion of the atomic nuclei when describing the electrons in a molecule. The physical basis for the Born-Oppenheimer approximation is the fact that the mass of an atomic nucleus in a molecule is much larger than the mass of an electron (more than 1000 times). Because of this difference, the nuclei move much more slowly than the electrons. In addition, due to their opposite charges, there is a mutual attractive force of: acting on an atomic nucleus and an electron. This force causes both particles to be accelerated. Since the magnitude of the acceleration is inversely proportional to the mass, $a = f/m$, the acceleration of the electrons is large and the acceleration of the atomic nuclei is small; the difference is a factor of more than 1000. Consequently, the electrons are moving and responding to forces very quickly, and the nuclei are not, then we fix the nucleus, $\mathbf{R}_{I}$ becomes a constant, thus the kinetic term for nucleus becomes zero. And the potential energy for the repulsion between nucleus becomes a constant. 
$$
\boxed{
\hat H_{\mathrm{el}} =
-\sum_{i=1}^{N}\frac{1}{2}\nabla_i^{2}
-\sum_{i=1}^{N}\sum_{I=1}^{M}\frac{Z_I}{|\mathbf r_i-\mathbf R_I|}
+\sum_{1\le i<j\le N}\frac{1}{|\mathbf r_i-\mathbf r_j|}
+ \sum_{I<J}\frac{Z_I Z_J}{|\mathbf R_I-\mathbf R_J|}
}
$$

This is the form that we are going to work with; a second useful approach to the problem is use an **Ansatz**, which is a guess solution guided by intuition, this normally depend on certain number of parameters, then the problem becomes on optimize this **Ansatz**.

Another important matter are the complex numbers, we should to work with them, no! Why is that?
### Rayleigh Quotient
We need a way to measure how well our Ansatz $\psi_{\theta}$ is doing, if our ansatz is bad then don't reflect the truly form of the system, Deep learning approachs use data sets to train their ansatz (model initialized on randoms parameters) in this case we don't need any external but physical principles.
Lets first introduce the **Rayleigh quotient**. If $A$ is an operator and $x$ is a state, the number:
$$
R_{A}(x)=\frac{\braket{ x |A|x  }}{\braket{ x | x } }
$$
is the **expectation value** of that operator in that state. The important for us is that if $\psi$ is a wave function and $\hat{H}$ the **Hamiltonian**, then the Rayleigh quotient:
$$
R_{\hat{H}}(\psi)=\frac{\braket{ \psi | \hat{H}| \psi}}{\braket{ \psi | \psi } }
$$
is the average (expected) energy of the system when it is in the state $\psi$. 
### Variational Principle 

The [[Variational Principle]]  states that the expectation value for the binding energy obtained using an approximate wave function and the exact Hamiltonian operator will be higher than or equal to the true energy for the system. This idea is really powerful. When implemented it permits us to find the best approximate wavefunction from a given wavefunction that contains one or more adjustable parameters, called a trial wavefunction. A mathematical statement of the variational principle is.
$$
R_{\hat{H}}(\psi_{\text{ansatz}})\geq R_{\hat{H}}(\psi_{\text{true}})
$$
The true ground-state wave function $\psi_{\text{true}}$ is the one that minimizes the rayleigh quotient. 
$$
\psi_{0}=\underset{\psi}{\text{argmin}}( R_{\hat{H}}(\psi))
$$
So if we minimize the rayleigh quotient of our ansatz we are going to be more near of the true wave function.
### Variational Monte Carlo 

To the process that we are going to use for optimize our ansatz is called Variational Monte Carlo (VMC). Now we can see to the rayleigh quotient as a loss function $\mathcal{L}$ with the form of.
$$
\mathcal{L}(\Psi_{\theta})=\frac{\bra{\Psi_{\theta}} \hat{H}\ket{\Psi_{\theta}} }{\braket{ \Psi_{\theta} | \Psi _{\theta}} }=\frac{\int d\mathbf{r}\Psi ^{*}(\mathbf{r})\hat{H}\Psi(\mathbf{r})}{\int d\mathbf{r}\Psi ^{*}(\mathbf{r})\Psi(\mathbf{r})}
$$
Evaluate that integral is hard, another smart approach is the follow, define a probability distribution $p_{\theta}$ with the follow form:
$$
p_{\theta}(x)=\frac{\lvert \Psi_{\theta} (x)\rvert ^{2}}{\int dx'\Psi^{2}_{\theta}(x')}
$$
Realize that for compute $p_{\theta}(x)$ for a specific $x$ is complicated due the integral that appears, this is going to be important later. Defining the local energy $E_{L}$ with:
$$
E_{L}(x)=\Psi ^{-1}_{\theta}(x)\hat{H}\Psi_{\theta}(x)
$$
Then the loss becomes:
$$
\mathcal{L}(\Psi_{\theta})=\int \frac{ \hat{H}\Psi_{\theta}(x)}{\Psi_{\theta}(x)}p_{\theta}(x)dx
$$
Which is the expected value of the local energy:
$$
\mathcal{L}(\Psi_{\theta})=\mathbb{E}_{x\sim p_{\theta}}[E_{L}(x)]
$$
To optimize our wave function we know by deep learning that we need evaluate that expectation and obtain the derivative for back propagation, how we make that?
We are going to use the the Monte Carlo estimatior:
$$
\mathcal{L}_{\theta}=\mathbb{E}_{x\sim p_{\theta}}[E_{L}(x)]\approx \frac{1}{M}\sum_{i=1}^{M} E_{L}(\mathbf{R}_{k})
$$
Whit $\mathbf{R}_{k}$ are samples from the $p_{\theta}$ probability distribution. This is
$$
\mathbf{R}_{1},\dots,\mathbf{R}_{M}\sim p_{\theta}(\mathbf{R})
$$
We obtain $E_{L}$ using:
$$
E_{L}(\mathbf{R}_{k})=\frac{\hat{H}\psi(\mathbf{R}_{k})}{\psi(\mathbf{R}_{k})}=-\frac{1}{2}\frac{\nabla^{2}\psi(\mathbf{R_{k}})}{\psi(\mathbf{R}_{k})}+V(\mathbf{R}_{k})
$$
Calculus tell us that for a any derivable function $f$.
$$
\frac{\nabla^{2}f}{f}=[\nabla^{2}\log f+(\nabla f)^{2}]
$$
In practice is more numerically stable work using that form, thus:
$$
E_{L}(\mathbf{R}_{k})=-\frac{1}{2}\sum_{i=1}^{N} \sum_{j=1}^{3} \left[ \frac{\partial ^{2} \log \lvert \Psi(x) \rvert }{\partial r_{ij}^{2}}+\left( \frac{\partial \log \lvert \Psi(x) \rvert }{\partial r_{ij}} \right)^{2} \right]+V(\mathbf{R_{k}})
$$
The gradient of the energy respect to the parameters by a parameterized wave functions is:
$$
\nabla _{\theta}\mathcal{L}=2\mathbb{E}_{x\sim \Psi^{2}}[(E_{L}(x)-\mathbb{E}_{x'\sim\Psi^{2}}[E_{L}(x')])\nabla \log \lvert \Psi(x) \rvert ]
$$
### Metropolis Hastings (MH) Algorithm
To obtain samples from the probability distribution $p_{\theta}$ we are going to use the [[Metropolis Hasting algorithm]]. (MH)

MH is a [[Markov Chain Monte Carlo]] (MCMC) method used to obtain a sequence of random samples from a probability distribution. The reason to use this method over another well knows methods (e.g. example) is that MH don't suffer of the [[Curse of Dimensionality]] 
this is, it remains strong while increase the dimension of the problem, and since we are going to be working on dimension around ten is efficient use this method. The algorithm works like follow:

1. Take a initial configuration $\mathbf{X}_{0}\in E$ arbitrary:
2. Propose $\mathbf{X}'=\mathbf{X}_{0}+\eta$ ,where $\eta \sim q(\eta)$, $q$ is a probability density on $E$ called **proposal kernel**. In our case we are going to a [[symmetric Gaussian]].
3. Compute the quantity:
$$
A(\mathbf{X_{0}}, \mathbf{X}')=\text{min}\left( 1,\frac{\rho(\mathbf{X}')}{\rho(\mathbf{X}_{0})} \frac{q(\mathbf{X}'-\mathbf{X}_{0})}{q(\mathbf{X}_{0}-\mathbf{X}')}\right)
$$
Where $\rho$ is the target distribution where we want sample, In the case where $q$ is symmetric, this simplifies to:
$$
A(\mathbf{X}_{0},\mathbf{X}')=\text{min}\left( 1,\frac{\rho(\mathbf{X})}{\rho(\mathbf{X}_{0})} \right)
$$
Note that in our case $\rho$ is equal to $p_{\theta}$, we said that compute the integral factor remains a challenge, but in this case we have it.
$$
\frac{p_{\theta}(\mathbf{X}')}{p_{\theta}(\mathbf{X}_{0})}=\frac{\lvert \psi_{\theta}(\mathbf{X}') \rvert ^{2}/\int \lvert \psi_{\theta} \rvert ^{2}dx}{\lvert \psi_{\theta}(\mathbf{X}_{0}) \rvert ^{2}/\int \lvert \psi_{\theta} \rvert ^{2}dx}=\frac{\lvert \psi_{\theta}(\mathbf{X'}) \rvert ^{2}}{\lvert \psi_{\theta}(\mathbf{X}_{0}) \rvert ^{2}}
$$
4. Generate a uniform number $U\in[0,1]$
5. If: $U<A(\mathbf{X}_{0}\to \mathbf{X'}_{l})$ then $\mathbf{X_{1}}=\mathbf{X}'$, otherwise try another $\mathbf{X}'$. Accept or decline.
6. Repeat until obtain $N_{eq}$ accepted sample, the changes stabilizes (we reach a stationary distribution) this phase is called **burn in**.
7. From $\mathbf{X}_{N_{\text{eq}}}$ generate $M$ samples until reach the sample $\mathbf{X}_{N_{\text{eq}}+M+1}$.
In each sample generates $E_{L}(\mathbf{R}_{k})$ then average to obtain $\mathbb{E}(E_{L})$ and with it and the derivative from equation number [number] we are able to begin the back propagation step.
## Deep Learning Fundamentals
This subsection introduces the core concepts of  Deep Learning that are going to be applied in this work.
### Multi Layer Perceptron
A multi layer perceptron (MLP) is a nonlinear function $\mathcal{F}:\mathbb{R}^{\text{in}}\to \mathbb{R}^{\text{out}}$.  @nielsenNeuralNetworksDeep2015 , it actually is the composition of $L$ layers, the first layer is called the input layer, the last output layer and the intermediates hidden layers. In each layer we find a arbitrary number of neurons although is a good practice always choose number which are powers of two and an affine map $\mathbf{z}^{(l)},l\in \{ L,L-1,\dots,2 \}$ ($l=1$ is the input layer) of the follow form.
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
You typically train a MLP, using a training data set, a loss function (e.g Mean Square Error, Mean Absolute Error, Cross entropy) and an optimizer (e.g GD, SGD, ADAM). Additionally you can use regularization techniques such as dropout to improve the generalization of the Net.


### Natural gradient Descent

As we mentioned, there are many ways to update the parameters of a neural network: Gradient Descent, Stochastic Gradient Descent, [[Adaptive Moment Estimation]] (ADAM), etc. All of them implicitly assume that the parameter space $\Theta \subset \mathbb{R}^d$ is equipped with the standard Euclidean metric, so that “length” and “steepest descent” are measured with respect to $\|\Delta\theta\|_2$.

In our case the loss $\mathcal{L}(\theta)$ depends on a probability distribution $p_\theta$, not just on $\theta$ directly. For example, in variational Monte Carlo we take
$$
p_\theta(x)
= \frac{|\psi_\theta(x)|^2}{\displaystyle \int |\psi_\theta(x')|^2\,dx'} ,
$$
so $\theta$ parametrizes an entire family of probability densities over configurations $x$. It is therefore more natural to measure distances between *distributions* $p_\theta$ and $p_{\theta+\Delta\theta}$, rather than between the parameter vectors themselves.

A canonical way to measure the distance between nearby probability distributions is the Kullback–Leibler (KL) divergence
$$
\mathrm{KL}\big(p_\theta \,\|\, p_{\theta+\Delta\theta}\big)
= \mathbb{E}_{x\sim p_\theta}\!\left[\log\frac{p_\theta(x)}{p_{\theta+\Delta\theta}(x)}\right].
$$
For small steps $\Delta\theta$ one can show that a second–order Taylor expansion of the KL gives
$$
\mathrm{KL}\big(p_\theta \,\|\, p_{\theta+\Delta\theta}\big)
= \tfrac12\,\Delta\theta^\top \mathcal{F}(\theta)\,\Delta\theta + \mathcal{O}(\|\Delta\theta\|^3),
$$
where $\mathcal{F}(\theta)$ is the Fisher Information Matrix (FIM). To define it, introduce the **score function**
$$
s_\theta(x) \in \mathbb{R}^d, \qquad
s_\theta(x) = \nabla_\theta \log p(x\mid \theta),
$$
then the FIM is
$$
\mathcal{F}(\theta)
= \mathbb{E}_{x\sim p(\cdot\mid\theta)}\!\big[\,s_\theta(x)\,s_\theta(x)^{\mathsf T}\big].
$$

The set of distributions
$$
\mathcal{M} = \{\, p_\theta(z)\;|\; \theta \in \Theta \subset \mathbb{R}^d \,\}
$$
can be viewed as a differentiable manifold, and $\mathcal{F}(\theta)$ defines a Riemannian metric on its tangent space. Concretely, for tangent vectors $u,v \in \mathbb{R}^d$ at $\theta$ we define the inner product
$$
\langle u,v \rangle_\theta
= u^{\mathsf T}\,\mathcal{F}(\theta)\,v.
$$
This metric says: two parameter directions are “close” if they induce similar infinitesimal changes in the *distribution* $p_\theta$.

Now ask the usual steepest–descent question, but with this non-Euclidean metric:

Find the direction $\Delta\theta$ that decreases $\mathcal{L}(\theta)$ the fastest, among all directions with fixed “length” $\|\Delta\theta\|_\theta^2 = \Delta\theta^\top \mathcal{F}(\theta)\,\Delta\theta$.

Solving this constrained optimization problem (e.g. with Lagrange multipliers) yields the **natural gradient** direction
$$
\Delta\theta_{\text{nat}} \;\propto\; -\,\mathcal{F}(\theta)^{-1}\,\nabla_\theta \mathcal{L}(\theta).
$$
Thus the natural gradient descent update is
$$
\Delta\theta_{\text{nat}}
= -\,\eta\,\mathcal{F}(\theta)^{-1}\,\nabla_\theta \mathcal{L}(\theta),
$$
where $\eta>0$ is a step size. Compared with the usual gradient $\nabla_\theta \mathcal{L}$, the factor $\mathcal{F}^{-1}$ “preconditions” the gradient by the local geometry of the model’s probability distribution: directions that barely change $p_\theta$ are amplified, directions that change it a lot are damped.

Natural gradient descent is therefore meaningful exactly in the situation we care about: when the loss depends on the parameters *through* a probability model $p_\theta$ (e.g. likelihood, cross-entropy, KL, variational objectives, variational Monte Carlo energy, etc.).

### Kronecker Factored Approximate Curvature

Directly computing and inverting the full Fisher matrix $\mathcal{F}(\theta)$ is infeasible for modern neural networks, since $\theta$ can have millions of components. Kronecker Factored Approximate Curvature (KFAC) is an efficient approximation that makes natural gradient updates practical for layered networks.

We sketch the construction for a fully connected layer $\ell$ with weight matrix $W_\ell$ and (for simplicity) no bias. Bias terms can be included by augmenting the activations with a constant $1$; we comment on this below.
#### Forward definition of $\mathbf{a}_\ell$

Consider a standard MLP. For a single input sample $x$, the forward pass at layer $\ell$ is

- **Input (activation) to layer $\ell$**:
$$
\mathbf{a}_\ell \in \mathbb{R}^{n_\ell}
$$
 This is the column vector of activations coming into layer $\ell$. For the first hidden layer, $\mathbf{a}_1$ is just the (possibly preprocessed) input. For deeper layers it is the nonlinearity output from the previous layer.

- **Pre-activation at layer $\ell$**:
  $$
  \mathbf{h}_\ell = W_\ell \,\mathbf{a}_\ell,
  $$
  where $W_\ell \in \mathbb{R}^{m_\ell \times n_\ell}$.
- **Output activation of layer $\ell$**:
  $$
  \tilde{\mathbf{a}}_\ell = \phi(\mathbf{h}_\ell),
  $$
  where $\phi$ is applied element-wise. In many notations $\tilde{\mathbf{a}}_\ell$ would become the input to the next layer, but to keep notation consistent with the Fisher block for $W_\ell$, we explicitly distinguish:
  - $\mathbf{a}_\ell$: input to $W_\ell$,
  - $\mathbf{h}_\ell$: pre-activation,
  - $\tilde{\mathbf{a}}_\ell$: output activation of layer $\ell$.

In KFAC, when we talk about $\mathbf{a}_\ell$ for the Fisher block of $W_\ell$, we always mean “the vector that $W_\ell$ multiplies on the right”.
#### Backward definition of $\mathbf{e}_\ell$
Let the loss for a single sample be $\mathcal{L}(\theta)$ (for example, negative log-likelihood or negative log of the wave-function probability). Define the **backward sensitivity** (or error signal) at layer $\ell$ as
$$
\mathbf{e}_\ell
= \frac{\partial \mathcal{L}}{\partial \mathbf{h}_\ell} \in \mathbb{R}^{m_\ell}.
$$
This is computed via backpropagation:
- At the output layer $L$:
  $$
  \mathbf{e}_L
  = \frac{\partial \mathcal{L}}{\partial \mathbf{h}_L}
  = \left(\frac{\partial \mathcal{L}}{\partial \tilde{\mathbf{a}}_L}\right) \odot \phi'(\mathbf{h}_L),
  $$
  where $\odot$ is the element-wise product or also Hadamard product.
- For hidden layers $\ell < L$:
$$
  \mathbf{e}_\ell
  = \frac{\partial \mathcal{L}}{\partial \mathbf{h}_\ell}
  = \left(W_{\ell+1}^{\mathsf T} \mathbf{e}_{\ell+1}\right) \odot \phi'(\mathbf{h}_\ell).
$$
In the context of natural gradient for probabilistic models, $\mathcal{L}$ is often chosen as $-\log p(X\mid\theta)$, so up to a sign we can also think of $\mathbf{e}_\ell$ as
$$
\mathbf{e}_\ell = \frac{\partial \log p(X\mid\theta)}{\partial \mathbf{h}_\ell}.
$$
#### Gradient w.r.t. $W_\ell$ and the form $\mathbf{a}_\ell \otimes \mathbf{e}_\ell$

For a single sample, using the chain rule,
$$
\frac{\partial \mathcal{L}}{\partial W_\ell}
= \frac{\partial \mathcal{L}}{\partial \mathbf{h}_\ell}
  \frac{\partial \mathbf{h}_\ell}{\partial W_\ell}
= \mathbf{e}_\ell\, \mathbf{a}_\ell^{\mathsf T}.
$$

If instead of $\mathcal{L}$ we use $\log p(X\mid\theta)$ (as in the Fisher definition), we get
$$
\frac{\partial \log p(X\mid\theta)}{\partial W_\ell}
= \mathbf{e}_\ell\, \mathbf{a}_\ell^{\mathsf T},
$$
with $\mathbf{e}_\ell = \partial \log p / \partial \mathbf{h}_\ell$.

Now vectorize the gradient. Using the standard identity
$$
\mathrm{vec}(uv^{\mathsf T}) = v \otimes u,
$$
with $u = \mathbf{e}_\ell$ and $v = \mathbf{a}_\ell$, we obtain
$$
\frac{\partial \log p(X\mid\theta)}{\partial \mathrm{vec}(W_\ell)}
= \mathrm{vec}\!\left(\frac{\partial \log p}{\partial W_\ell}\right)
= \mathrm{vec}(\mathbf{e}_\ell\,\mathbf{a}_\ell^{\mathsf T})
= \mathbf{a}_\ell \otimes \mathbf{e}_\ell.
$$

This gives the key structural form used by KFAC.

#### Fisher block for a single layer

The Fisher block associated with the parameters $W_\ell$ is
$$
\mathcal{F}_\ell
= \mathbb{E}_{p(\mathbf{X})}\!\left[
\frac{\partial \log p(X\mid\theta)}{\partial \mathrm{vec}(W_\ell)}
\frac{\partial \log p(X\mid\theta)}{\partial \mathrm{vec}(W_\ell)}^{\mathsf T}
\right].
$$

Plugging in the expression above,
$$
\mathcal{F}_\ell
= \mathbb{E}_{p(\mathbf{X})}\!\big[
(\mathbf{a}_\ell \otimes \mathbf{e}_\ell)
(\mathbf{a}_\ell \otimes \mathbf{e}_\ell)^{\mathsf T}
\big].
$$

Here $p(\mathbf{X})$ denotes the distribution over inputs and labels (or configurations, in the VMC case). In practice this expectation is approximated by averaging over a mini-batch of samples $X$ and the corresponding forward/backward passes that produce $\mathbf{a}_\ell$ and $\mathbf{e}_\ell$.

Computing and inverting $\mathcal{F}_\ell$ directly is still expensive, because its dimension is
$$
(\text{dim}(\mathbf{a}_\ell)\,\text{dim}(\mathbf{e}_\ell))
\times
(\text{dim}(\mathbf{a}_\ell)\,\text{dim}(\mathbf{e}_\ell)).
$$
KFAC makes two key approximations to make this tractable.

1. **Block–diagonal across layers.**  
   Off–diagonal blocks $\mathcal{F}_{ij}$ are assumed negligible when $\theta_i$ and $\theta_j$ belong to different layers. This makes the Fisher approximately block–diagonal, with one block per layer.

2. **Kronecker factorization within each layer.**  
   Inside a layer, KFAC assumes that the correlation between activations and errors factorizes:
   $$
   \mathcal{F}_\ell
   = \mathbb{E}_{p(\mathbf{X})}\!\big[
   (\mathbf{a}_\ell \otimes \mathbf{e}_\ell)
   (\mathbf{a}_\ell \otimes \mathbf{e}_\ell)^{\mathsf T}
   \big]
   = \mathbb{E}_{p(\mathbf{X})}\!\big[
   (\mathbf{a}_\ell\mathbf{a}_\ell^{\mathsf T}) \otimes
   (\mathbf{e}_\ell\mathbf{e}_\ell^{\mathsf T})
   \big]
   \;\approx\;
   \mathbb{E}_{p(\mathbf{X})}[\mathbf{a}_\ell\mathbf{a}_\ell^{\mathsf T}]
   \;\otimes\;
   \mathbb{E}_{p(\mathbf{X})}[\mathbf{e}_\ell\mathbf{e}_\ell^{\mathsf T}].
   $$

Define the *activation covariance* and *error covariance*:
$$
A_\ell = \mathbb{E}_{p(\mathbf{X})}[\mathbf{a}_\ell\mathbf{a}_\ell^{\mathsf T}],
\qquad
S_\ell = \mathbb{E}_{p(\mathbf{X})}[\mathbf{e}_\ell\mathbf{e}_\ell^{\mathsf T}].
$$
In practice these expectations are updated as running averages over mini-batches:
$$
A_\ell \approx \frac{1}{B}\sum_{b=1}^B \mathbf{a}_\ell^{(b)} \mathbf{a}_\ell^{(b)\mathsf T},
\qquad
S_\ell \approx \frac{1}{B}\sum_{b=1}^B \mathbf{e}_\ell^{(b)} \mathbf{e}_\ell^{(b)\mathsf T},
$$
where $b$ indexes samples in the batch and $\mathbf{a}_\ell^{(b)}, \mathbf{e}_\ell^{(b)}$ are obtained by a standard forward and backward pass for that sample.
With this approximation we have
$$
\mathcal{F}_\ell \approx A_\ell \otimes S_\ell.
$$

The crucial property of the Kronecker product is that
$$
(A_\ell \otimes S_\ell)^{-1}
= A_\ell^{-1} \otimes S_\ell^{-1},
$$
so the inverse of the (huge) layer–Fisher block can be obtained by inverting the much smaller matrices $A_\ell$ and $S_\ell$. Thus the natural gradient update for the weights of layer $\ell$ becomes
$$
\Delta\theta_{\text{nat},\ell}
\approx -\,\eta\,\big(A_\ell^{-1} \otimes S_\ell^{-1}\big)\,
\nabla_{\mathrm{vec}(W_\ell)} \mathcal{L}.
$$

In summary, KFAC replaces the intractable inverse
$$
\mathbb{E}_{p(\mathbf{X})}\big[ (\mathbf{a}_\ell\otimes \mathbf{e}_\ell)
(\mathbf{a}_\ell\otimes \mathbf{e}_\ell)^{\mathsf T} \big]^{-1}
$$
by the efficiently computable approximation
$$
\mathbb{E}_{p(\mathbf{X})}\big[(\mathbf{a}_\ell\otimes \mathbf{e}_\ell)
(\mathbf{a}_\ell\otimes \mathbf{e}_\ell)^{\mathsf T}\big]^{-1}
\;\approx\;
\mathbb{E}_{p(\mathbf{X})}[\mathbf{a}_\ell\mathbf{a}_\ell^{\mathsf T}]^{-1}
\otimes
\mathbb{E}_{p(\mathbf{X})}[\mathbf{e}_\ell\mathbf{e}_\ell^{\mathsf T}]^{-1},
$$
which captures the dominant curvature structure while keeping the cost of natural gradient descent comparable to standard first–order methods.
We have ignored biases above for clarity. In practice one can either (i) augment $\mathbf{a}_\ell$ with a constant $1$ to absorb biases into $W_\ell$, or (ii) maintain separate smaller KFAC factors for biases; both approaches preserve the same Kronecker structure.
### Attention Mechanisms

The idea of an *attention mechanism* was introduced in neural machine translation by Bahdanau et al. @bahdanau2014neural Instead of compressing an entire input sequence into a single fixed-size vector, the model learns to **focus** on different parts of the input when generating each output token.

Given a query vector $\mathbf{q} \in \mathbb{R}^{d_h}$ and a set of key–value pairs
$\{(\mathbf{k}_j, \mathbf{v}_j)\}_{j=1}^T$ with $\mathbf{k}_j, \mathbf{v}_j \in \mathbb{R}^{d_h}$, the (scaled dot–product) attention mechanism computes:

1. **Compatibility scores**
   $$
   e_j \;=\; \frac{\mathbf{q}^\top \mathbf{k}_j}{\sqrt{d_h}}, \qquad j = 1,\dots,T,
   $$

2. **Normalized attention weights**
   $$
   \alpha_j \;=\; \frac{\exp(e_j)}{\sum_{m=1}^{T} \exp(e_m)}
   \;=\; \text{Softmax}_j\!\left( \frac{\mathbf{q}^\top \mathbf{k}_j}{\sqrt{d_h}} \right),
   $$

3. **Weighted sum of values**
$$
\mathbf{o} \;=\; \sum_{j=1}^{T} \alpha_j \mathbf{v}_j.
$$
Intuitively, the query $\mathbf{q}$ asks: *“which elements of the set are relevant to me now?”*  
The keys $\mathbf{k}_j$ encode *what each element offers*, and the values $\mathbf{v}_j$ encode *what we take from each element once we decide to pay attention to it*.

### Self-Attention and Multi-Head Self-Attention

In **self-attention**, the queries, keys, and values are all obtained from the **same** set of input vectors.  
Consider a sequence of input embeddings
$$
\mathbf{x}_1, \dots, \mathbf{x}_T \in \mathbb{R}^{d},
$$
and stack them into a matrix
$$
\mathbf{X} \in \mathbb{R}^{T \times d}, \quad
\mathbf{X} = 
\begin{bmatrix}
\mathbf{x}_1^\top \\
\vdots \\
\mathbf{x}_T^\top
\end{bmatrix}.
$$
To build one attention **head** of dimension $d_h$, we introduce three learnable matrices:
$$
\mathbf{W}^Q \in \mathbb{R}^{d \times d_h}, \quad
\mathbf{W}^K \in \mathbb{R}^{d \times d_h}, \quad
\mathbf{W}^V \in \mathbb{R}^{d \times d_h}.
$$
We then compute queries, keys, and values:
$$
\mathbf{Q} = \mathbf{X} \mathbf{W}^Q \in \mathbb{R}^{T \times d_h}, \qquad
\mathbf{K} = \mathbf{X} \mathbf{W}^K \in \mathbb{R}^{T \times d_h}, \qquad
\mathbf{V} = \mathbf{X} \mathbf{W}^V \in \mathbb{R}^{T \times d_h}.
$$

The **scaled dot-product self-attention** for this head is:
$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V})
= \text{Softmax}\!\left( \frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_h}} \right) \mathbf{V},
$$
where the softmax is applied row-wise.  Element-wise, the output at position $t$ is
$$
\mathbf{o}_t = \sum_{j=1}^{T} \alpha_{tj} \mathbf{v}_j, \quad \text{with} \quad
\alpha_{tj} = \frac{\exp\!\left( \mathbf{q}_t^\top \mathbf{k}_j / \sqrt{d_h} \right)}{\sum_{m=1}^{T} \exp\!\left( \mathbf{q}_t^\top \mathbf{k}_m / \sqrt{d_h} \right)}.
$$
You can think of this as: *each position $t$ in the sequence “looks at” every other position $j$ and decides how much to care about it*.
### Multi-Head Self-Attention

A single head can only look at interactions in one “representation subspace” of dimension $d_h$.  
**Multi-head attention** uses several heads in parallel, each with its own projection matrices, so that different types of relationships can be captured simultaneously.
Let $n_h$ be the number of heads. For head $i = 1, \dots, n_h$ we have
$$
\mathbf{W}_i^Q,\, \mathbf{W}_i^K,\, \mathbf{W}_i^V \in \mathbb{R}^{d \times d_h}.
$$

Head $i$ computes:
$$
\text{head}_i(\mathbf{X})
= \text{Attention}(\mathbf{X}\mathbf{W}_i^Q,\, \mathbf{X}\mathbf{W}_i^K,\, \mathbf{X}\mathbf{W}_i^V)
\in \mathbb{R}^{T \times d_h}.
$$
The outputs of all heads are concatenated along the feature dimension and then linearly mixed:
$$
\mathbf{U} 
= \left[ \text{head}_1(\mathbf{X}) \,;\, \dots \,;\, \text{head}_{n_h}(\mathbf{X}) \right]
\in \mathbb{R}^{T \times (n_h d_h)},
$$
$$
\mathbf{O} = \mathbf{U} \mathbf{W}^O, \qquad
\mathbf{W}^O \in \mathbb{R}^{(n_h d_h) \times d}.
$$
If we focus on one time step $t$ and head $i$, we can write the per-head output as
$$
\mathbf{o}_{t,i} = \sum_{j=1}^{T} 
\text{Softmax}_j\!\left( \frac{\mathbf{q}_{t,i}^\top \mathbf{k}_{j,i}}{\sqrt{d_h}} \right)\mathbf{v}_{j,i},
$$
and the final vector at time $t$ after concatenation and output projection as
$$
\mathbf{u}_t =
\mathbf{W}^{O} 
\begin{bmatrix}
\mathbf{o}_{t,1} \\
\vdots \\
\mathbf{o}_{t,n_h}
\end{bmatrix}.
$$
From a physics point of view, you can read multi-head attention as **several different “channels” of interaction**: one head might focus on short-range relations, another on long-range ones, another on some specific pattern (e.g. symmetry, local structure), and so on.

### Transformer Architecture

The **Transformer** was introduced by Vaswani et al. @Vaswani2017 with the slogan *“Attention Is All You Need.”*  Its core building block is a **layer** that combines:
1. **Multi-head self-attention**  
2. **Position-wise feed-forward network (FFN)**
Both sublayers use **residual connections** and **layer normalization**.
For an input sequence $\mathbf{X} \in \mathbb{R}^{T \times d}$ (already including positional information), one Transformer layer performs:
3. **Multi-head self-attention sublayer**
$$
   \mathbf{H} = \text{MHA}(\mathbf{X}), \qquad
   \mathbf{X}^{(1)} = \text{LayerNorm}\!\left( \mathbf{X} + \mathbf{H} \right).
$$
4. **Feed-forward sublayer** (applied independently at each position)
$$
   \text{FFN}(\mathbf{x}) = \sigma\!\left( \mathbf{x}\mathbf{W}_1 + \mathbf{b}_1 \right)\mathbf{W}_2 + \mathbf{b}_2,
$$
typically with $\sigma$ a nonlinearity such as ReLU or GELU and an intermediate width $d_{\text{ff}} > d$.
At the sequence level:   
$$
   \mathbf{Z} = \text{FFN}(\mathbf{X}^{(1)}), \qquad
   \mathbf{X}^{\text{out}} = \text{LayerNorm}\!\left( \mathbf{X}^{(1)} + \mathbf{Z} \right).
$$
Stacking several such layers yields a deep architecture where, at each layer, every position can interact with every other position through self-attention.

In the original formulation, **positional encodings** (sinusoidal or learned) are added to the embeddings so that the model can distinguish different positions in the sequence:
$$
\mathbf{X}_0 = \mathbf{E} + \mathbf{P},
$$
where $\mathbf{E}$ are token embeddings and $\mathbf{P}$ are positional encodings.

### Why Transformers Instead of RNNs or LSTMs?

Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks process the sequence **sequentially** this is each new state depends on the previous one. This has two important consequences:
1.   Information must flow through many time steps, which can lead to vanishing or exploding gradients and makes it hard to model very long-range interactions.
2. Poor parallelization because each step depends on the previous one, you cannot compute all time steps in parallel. Training and inference are inherently sequential.
Transformers address both issues:
- Global interactions in one step, self-attention allows every position to directly interact with every other position in a *single* layer, which is ideal when we care about *all-to-all* correlations (as in many-electron systems, where each electron “feels” all the others).
- Full parallelism over sequence length. Given $\mathbf{X}$, the matrices $\mathbf{Q}$, $\mathbf{K}$, $\mathbf{V}$ and the attention outputs for all time steps are computed via matrix multiplications. This is extremely efficient on modern accelerators (GPUs/TPUs).
For a many-electron Schrödinger equation, the wave function depends on the joint configuration of all particles.  A Transformer-based ansatz naturally provides a way for each electron’s representation to **look at all other electrons** and the nuclei, capturing complex correlation patterns through attention, while remaining highly parallelizable.

A very important work for us is FermiNet (Pfau et al. 2020). It uses deep neural networks to represent **orbitals** and then combines them into a sum of Slater determinants. At the top level, the ansatz is a linear combination of $K$ determinant products
$$
\psi(\mathbf{x}_1,\dots,\mathbf{x}_n)
= \sum_{k=1}^K \omega_k \,\det[\Phi^{k}],
$$
where $\omega_k$ are learnable coefficients and $\Phi^k$ is a matrix of single-particle orbitals. For a system without explicit spin separation one can write
$$
\det[\Phi^k] =
\begin{vmatrix}
\phi_{1}^{k}(\mathbf{x}_{1})  & \dots  &  \phi_{1}^{k}(\mathbf{x}_{n}) \\
\vdots   &  & \vdots  \\
\phi_{n}^{k}(\mathbf{x}_{1}) & \dots & \phi_{n}^{k}(\mathbf{x}_{n})
\end{vmatrix}
= \det[\phi_i^k(\mathbf{x}_j)].
$$
Here $\phi_i^k$ is the $i$-th orbital in determinant $k$, and we evaluate it on the coordinates of electron $j$.

However, in FermiNet we are dealing with electrons with spin, so things are slightly more structured, and the orbitals depend on **all** electron coordinates, not only on the one being “plugged in”. That is why we write the orbitals as
$$
\phi^{k\alpha}_i\big(\mathbf{r}^\alpha_j;\{\mathbf{r}^\alpha_{/j}\};\{\mathbf{r}^{\bar{\alpha}}\}\big),
$$
where:
- $\alpha \in \{\uparrow,\downarrow\}$ is the spin sector,
- $\mathbf{r}^\alpha_j$ is the position of electron $j$ with spin $\alpha$,
- $\{\mathbf{r}^\alpha_{/j}\}$ denotes the positions of all **other** electrons with spin $\alpha$,
- $\{\mathbf{r}^{\bar{\alpha}}\}$ denotes the positions of electrons with the opposite spin.

So the orbital evaluated on electron $j$ “knows” about all other electrons. The indices:
- $i$ = orbital index (row of the determinant),
- $j$ = electron index (column of the determinant),
- $\alpha,\beta$ = spin labels ($\uparrow$ or $\downarrow$),
- $k$ = determinant index in the sum.

---

### Input coordinates and features

We denote by
- $\mathbf{r}^\uparrow_1,\dots,\mathbf{r}^\uparrow_{n^\uparrow}$ the coordinates of spin-up electrons,
- $\mathbf{r}^\downarrow_1,\dots,\mathbf{r}^\downarrow_{n^\downarrow}$ the coordinates of spin-down electrons,
- $\mathbf{R}_I$ the positions of nuclei, $I=1,\dots,N_\text{nuc}$.

The network builds two types of features:

1. **Electron–nucleus features** for each electron $i$ with spin $\alpha$:
   $$
   \mathbf{h}^{0,\alpha}_i
   = \text{concatenate}\Big(
       \mathbf{r}^\alpha_i - \mathbf{R}_I,\;
       \big|\mathbf{r}^\alpha_i - \mathbf{R}_I\big|
       \ \forall\, I
     \Big).
   $$
   This produces a feature vector that contains, for electron $(i,\alpha)$, all its relative position vectors to each nucleus, plus their distances.

2. **Electron–electron features** for each pair of electrons $(i,\alpha)$ and $(j,\beta)$:
   $$
   \mathbf{h}^{0,\alpha\beta}_{ij}
   = \text{concatenate}\Big(
       \mathbf{r}^\alpha_i - \mathbf{r}^\beta_j,\;
       \big|\mathbf{r}^\alpha_i - \mathbf{r}^\beta_j\big|
       \ \forall\, j,\beta
     \Big).
   $$
   For fixed $(i,\alpha)$, we build such features for all other electrons $(j,\beta)$, capturing their relative positions and distances.

The superscript $0$ indicates that these are the features at layer $\ell=0$ (input to the deep network). At deeper layers we will keep updating
- $\mathbf{h}^{\ell\alpha}_i$ (single-electron features),
- $\mathbf{h}^{\ell\alpha\beta}_{ij}$ (pairwise features),
for $\ell = 0,1,\dots,L-1$.

---

### Mixing and updating features across layers

At each hidden layer $\ell$, we want each electron’s features to depend on *all* other electrons, in a permutation-symmetric way. To do this, we form **averages** over electrons of the same or opposite spin.

First, define global spin-averaged single-electron features
$$
\mathbf{g}^{\ell\uparrow} =
\frac{1}{n^\uparrow}\sum_{j=1}^{n^\uparrow}\mathbf{h}^{\ell\uparrow}_j,
\qquad
\mathbf{g}^{\ell\downarrow} =
\frac{1}{n^\downarrow}\sum_{j=1}^{n^\downarrow}\mathbf{h}^{\ell\downarrow}_j.
$$

Next, for each electron $(i,\alpha)$, define averaged pairwise features:
$$
\mathbf{g}^{\ell\alpha\uparrow}_i
= \frac{1}{n^\uparrow}\sum_{j=1}^{n^\uparrow}\mathbf{h}^{\ell\alpha\uparrow}_{ij},
\qquad
\mathbf{g}^{\ell\alpha\downarrow}_i
= \frac{1}{n^\downarrow}\sum_{j=1}^{n^\downarrow}\mathbf{h}^{\ell\alpha\downarrow}_{ij}.
$$

Now we *concatenate* all this information into a single feature vector for electron $(i,\alpha)$:
$$
\begin{aligned}
\big(
\mathbf{h}^{\ell\alpha}_i,
\frac{1}{n^\uparrow}\sum_{j=1}^{n^\uparrow}\mathbf{h}^{\ell\uparrow}_j,
\frac{1}{n^\downarrow}\sum_{j=1}^{n^\downarrow}\mathbf{h}^{\ell\downarrow}_j,
\frac{1}{n^\uparrow}\sum_{j=1}^{n^\uparrow}\mathbf{h}^{\ell\alpha\uparrow}_{ij},
\frac{1}{n^\downarrow}\sum_{j=1}^{n^\downarrow}\mathbf{h}^{\ell\alpha\downarrow}_{ij}
\big)
&=
\big(\mathbf{h}^{\ell\alpha}_i, \mathbf{g}^{\ell\uparrow}, \mathbf{g}^{\ell\downarrow},
\mathbf{g}^{\ell\alpha\uparrow}_i, \mathbf{g}^{\ell\alpha\downarrow}_i \big) \\
&= \mathbf{f}^{\ell\alpha}_i.
\end{aligned}
$$

This $\mathbf{f}^{\ell\alpha}_i$ is what enters the **single-electron MLP** at layer $\ell$. The update is
$$
\mathbf{h}^{\ell+1,\alpha}_i
= \tanh\big(\mathbf{V}^\ell \mathbf{f}^{\ell\alpha}_i + \mathbf{b}^\ell\big) + \mathbf{h}^{\ell\alpha}_i,
$$
where $\mathbf{V}^\ell$ and $\mathbf{b}^\ell$ are learnable weights and biases, shared between electrons (for the given spin sector). The residual connection $+\mathbf{h}^{\ell\alpha}_i$ stabilizes training.

In parallel, the pairwise features are updated with a **pairwise MLP**:
$$
\mathbf{h}^{\ell+1,\alpha\beta}_{ij}
= \tanh\big(\mathbf{W}^\ell \mathbf{h}^{\ell\alpha\beta}_{ij} + \mathbf{c}^\ell\big)
+ \mathbf{h}^{\ell\alpha\beta}_{ij},
$$
with weights $\mathbf{W}^\ell$ and biases $\mathbf{c}^\ell$, again shared over all pairs $(i,j,\alpha,\beta)$.

By repeating these updates for $\ell = 0,\dots,L-1$, we eventually obtain **final single-electron features**
$$
\mathbf{h}^{L\alpha}_j \quad \text{for each electron } j \text{ of spin } \alpha.
$$
Notice how the indices work:
- $\ell$ runs over layers and disappears at the end,
- $i$ or $j$ always refer to a specific electron within a spin sector,
- $\alpha,\beta$ tell you which spin sector that electron belongs to.

---

### From final features to orbitals

The final orbitals are built as a function of the last-layer features $\mathbf{h}^{L\alpha}_j$, plus some additional “envelope” factors that handle the long-range decay and cusp conditions. For each determinant index $k$, spin $\alpha$, orbital index $i$, and electron $j$ we define
$$
\begin{aligned}
\phi^{k\alpha}_i\big(\mathbf{r}^\alpha_j; \{\mathbf{r}^\alpha_{/j}\}; \{\mathbf{r}^{\bar{\alpha}}\}\big)
&= \left(\mathbf{w}^{k\alpha}_i \cdot \mathbf{h}^{L\alpha}_j + g^{k\alpha}_i\right) \\
&\quad\times \sum_{m} \pi^{k\alpha}_{im}
\exp\Big(
- \big|\mathbf{\Sigma}_{im}^{k\alpha} \big(\mathbf{r}^{\alpha}_j - \mathbf{R}_m\big)\big|
\Big).
\end{aligned}
$$
Here:
- $\mathbf{w}^{k\alpha}_i$ and $g^{k\alpha}_i$ are learnable linear parameters for the “MLP part” of the orbital,
- the sum over $m$ is an “envelope” over nuclei (or centers),
- $\pi^{k\alpha}_{im}$ and $\mathbf{\Sigma}^{k\alpha}_{im}$ are learnable coefficients and matrices controlling the exponential decay around nucleus $m$.

All these parameters depend on the indices:
- $k$ selects which determinant in the sum,
- $i$ selects which orbital (row in the determinant),
- $\alpha$ selects the spin sector,
- $m$ selects which nuclear center in the envelope.

The dependence on all other electrons is hidden inside $\mathbf{h}^{L\alpha}_j$, which was built from the full set of positions $\{\mathbf{r}^\uparrow\},\{\mathbf{r}^\downarrow\}$ through the deep network.

---

### Assembling the spin-separated determinants

For each determinant index $k$ and spin sector $\alpha\in\{\uparrow,\downarrow\}$, we build a matrix
$$
D^{k\alpha}_{ij}
= \phi^{k\alpha}_i\big( \mathbf{r}^\alpha_j; \{\mathbf{r}^\alpha_{/j}\}; \{\mathbf{r}^{\bar{\alpha}}\}\big),
$$
with:
- rows indexed by the orbital label $i = 1,\dots,n^\alpha$,
- columns indexed by the electron label $j = 1,\dots,n^\alpha$ (with that spin).

Taking the determinant gives a properly antisymmetric function of the positions of electrons **with that spin**:
$$
\det\big[D^{k\alpha}\big]
= \det\left[\phi^{k\alpha}_i(\mathbf{r}^\alpha_j; \{\mathbf{r}^\alpha_{/j}\}; \{\mathbf{r}^{\bar{\alpha}}\})\right].
$$

For the full wavefunction, we combine spin-up and spin-down blocks:
$$
\begin{aligned}
\psi(\mathbf{r}^\uparrow_1,\ldots,\mathbf{r}^\uparrow_{n^\uparrow},
     \mathbf{r}^\downarrow_1,\ldots,\mathbf{r}^\downarrow_{n^\downarrow})
= \sum_{k} \omega_k \;&
\det\left[\phi^{k \uparrow}_i(\mathbf{r}^\uparrow_j; \{\mathbf{r}^\uparrow_{/j}\}; \{\mathbf{r}^\downarrow\})\right] \\
&\times
\det\left[\phi^{k \downarrow}_i(\mathbf{r}^\downarrow_j; \{\mathbf{r}^\downarrow_{/j}\}; \{\mathbf{r}^\uparrow\})\right].
\end{aligned}
$$

**Why are there two determinants?**  

In electronic structure, when we separate spin and spatial parts using spin-orbitals, the full Slater determinant over all electrons factorizes into the product of:
- one determinant involving only spin-up electrons,
- another determinant involving only spin-down electrons.

Each of these determinants is antisymmetric under exchange of two electrons **with the same spin**. The overall wavefunction constructed as the product of a spin-up determinant and a spin-down determinant is antisymmetric under exchange of any two electrons (when you take into account the spin labels). FermiNet keeps this structure and lets each block be represented by a powerful neural network ansatz for the orbitals.

Up to this point the building blocks are just MLP layers (with residual connections and special feature mixing), but the careful indexing
- $(i,\alpha)$ for “which electron/spin”,
- $j$ for summation over electrons,
- $\ell$ for layers,
- $k$ for determinant index,
is what guarantees that the final object has the correct permutation symmetry and antisymmetry required for a fermionic wavefunction.


# Psi Former Model

## Fermi Net

A very important work for us is FermiNet (Pfau et al. 2020). It uses deep neural networks to represent **orbitals** and then combines them into a sum of Slater determinants. At the top level, the ansatz is a linear combination of $K$ determinant products
$$
\psi(\mathbf{x}_1,\dots,\mathbf{x}_n)
= \sum_{k=1}^K \omega_k \,\det[\Phi^{k}],
$$
where $\omega_k$ are learnable coefficients and $\Phi^k$ is a matrix of single-particle orbitals. For a system without explicit spin separation one can write
$$
\det[\Phi^k] =
\begin{vmatrix}
\phi_{1}^{k}(\mathbf{x}_{1})  & \dots  &  \phi_{1}^{k}(\mathbf{x}_{n}) \\
\vdots   &  & \vdots  \\
\phi_{n}^{k}(\mathbf{x}_{1}) & \dots & \phi_{n}^{k}(\mathbf{x}_{n})
\end{vmatrix}
= \det[\phi_i^k(\mathbf{x}_j)].
$$
Here $\phi_i^k$ is the $i$-th orbital in determinant $k$, and we evaluate it on the coordinates of electron $j$.

However, in FermiNet we are dealing with electrons with spin, so things are slightly more structured, and the orbitals depend on **all** electron coordinates, not only on the one being “plugged in”. That is why we write the orbitals as
$$
\phi^{k\alpha}_i\big(\mathbf{r}^\alpha_j;\{\mathbf{r}^\alpha_{/j}\};\{\mathbf{r}^{\bar{\alpha}}\}\big),
$$
where:
- $\alpha \in \{\uparrow,\downarrow\}$ is the spin sector,
- $\mathbf{r}^\alpha_j$ is the position of electron $j$ with spin $\alpha$,
- $\{\mathbf{r}^\alpha_{/j}\}$ denotes the positions of all **other** electrons with spin $\alpha$,
- $\{\mathbf{r}^{\bar{\alpha}}\}$ denotes the positions of electrons with the opposite spin.

So the orbital evaluated on electron $j$ “knows” about all other electrons. The indices:
- $i$ = orbital index (row of the determinant),
- $j$ = electron index (column of the determinant),
- $\alpha,\beta$ = spin labels ($\uparrow$ or $\downarrow$),
- $k$ = determinant index in the sum.

---

### Input coordinates and features

We denote by
- $\mathbf{r}^\uparrow_1,\dots,\mathbf{r}^\uparrow_{n^\uparrow}$ the coordinates of spin-up electrons,
- $\mathbf{r}^\downarrow_1,\dots,\mathbf{r}^\downarrow_{n^\downarrow}$ the coordinates of spin-down electrons,
- $\mathbf{R}_I$ the positions of nuclei, $I=1,\dots,N_\text{nuc}$.

The network builds two types of features:

1. **Electron–nucleus features** for each electron $i$ with spin $\alpha$:
   $$
   \mathbf{h}^{0,\alpha}_i
   = \text{concatenate}\Big(
       \mathbf{r}^\alpha_i - \mathbf{R}_I,\;
       \big|\mathbf{r}^\alpha_i - \mathbf{R}_I\big|
       \ \forall\, I
     \Big).
   $$
   This produces a feature vector that contains, for electron $(i,\alpha)$, all its relative position vectors to each nucleus, plus their distances.

2. **Electron–electron features** for each pair of electrons $(i,\alpha)$ and $(j,\beta)$:
   $$
   \mathbf{h}^{0,\alpha\beta}_{ij}
   = \text{concatenate}\Big(
       \mathbf{r}^\alpha_i - \mathbf{r}^\beta_j,\;
       \big|\mathbf{r}^\alpha_i - \mathbf{r}^\beta_j\big|
       \ \forall\, j,\beta
     \Big).
   $$
   For fixed $(i,\alpha)$, we build such features for all other electrons $(j,\beta)$, capturing their relative positions and distances.

The superscript $0$ indicates that these are the features at layer $\ell=0$ (input to the deep network). At deeper layers we will keep updating
- $\mathbf{h}^{\ell\alpha}_i$ (single-electron features),
- $\mathbf{h}^{\ell\alpha\beta}_{ij}$ (pairwise features),
for $\ell = 0,1,\dots,L-1$.

---

### Mixing and updating features across layers

At each hidden layer $\ell$, we want each electron’s features to depend on *all* other electrons, in a permutation-symmetric way. To do this, we form **averages** over electrons of the same or opposite spin.

First, define global spin-averaged single-electron features
$$
\mathbf{g}^{\ell\uparrow} =
\frac{1}{n^\uparrow}\sum_{j=1}^{n^\uparrow}\mathbf{h}^{\ell\uparrow}_j,
\qquad
\mathbf{g}^{\ell\downarrow} =
\frac{1}{n^\downarrow}\sum_{j=1}^{n^\downarrow}\mathbf{h}^{\ell\downarrow}_j.
$$

Next, for each electron $(i,\alpha)$, define averaged pairwise features:
$$
\mathbf{g}^{\ell\alpha\uparrow}_i
= \frac{1}{n^\uparrow}\sum_{j=1}^{n^\uparrow}\mathbf{h}^{\ell\alpha\uparrow}_{ij},
\qquad
\mathbf{g}^{\ell\alpha\downarrow}_i
= \frac{1}{n^\downarrow}\sum_{j=1}^{n^\downarrow}\mathbf{h}^{\ell\alpha\downarrow}_{ij}.
$$

Now we *concatenate* all this information into a single feature vector for electron $(i,\alpha)$:
$$
\begin{aligned}
\big(
\mathbf{h}^{\ell\alpha}_i,
\frac{1}{n^\uparrow}\sum_{j=1}^{n^\uparrow}\mathbf{h}^{\ell\uparrow}_j,
\frac{1}{n^\downarrow}\sum_{j=1}^{n^\downarrow}\mathbf{h}^{\ell\downarrow}_j,
\frac{1}{n^\uparrow}\sum_{j=1}^{n^\uparrow}\mathbf{h}^{\ell\alpha\uparrow}_{ij},
\frac{1}{n^\downarrow}\sum_{j=1}^{n^\downarrow}\mathbf{h}^{\ell\alpha\downarrow}_{ij}
\big)
&=
\big(\mathbf{h}^{\ell\alpha}_i, \mathbf{g}^{\ell\uparrow}, \mathbf{g}^{\ell\downarrow},
\mathbf{g}^{\ell\alpha\uparrow}_i, \mathbf{g}^{\ell\alpha\downarrow}_i \big) \\
&= \mathbf{f}^{\ell\alpha}_i.
\end{aligned}
$$

This $\mathbf{f}^{\ell\alpha}_i$ is what enters the **single-electron MLP** at layer $\ell$. The update is
$$
\mathbf{h}^{\ell+1,\alpha}_i
= \tanh\big(\mathbf{V}^\ell \mathbf{f}^{\ell\alpha}_i + \mathbf{b}^\ell\big) + \mathbf{h}^{\ell\alpha}_i,
$$
where $\mathbf{V}^\ell$ and $\mathbf{b}^\ell$ are learnable weights and biases, shared between electrons (for the given spin sector). The residual connection $+\mathbf{h}^{\ell\alpha}_i$ stabilizes training.

In parallel, the pairwise features are updated with a **pairwise MLP**:
$$
\mathbf{h}^{\ell+1,\alpha\beta}_{ij}
= \tanh\big(\mathbf{W}^\ell \mathbf{h}^{\ell\alpha\beta}_{ij} + \mathbf{c}^\ell\big)
+ \mathbf{h}^{\ell\alpha\beta}_{ij},
$$
with weights $\mathbf{W}^\ell$ and biases $\mathbf{c}^\ell$, again shared over all pairs $(i,j,\alpha,\beta)$.

By repeating these updates for $\ell = 0,\dots,L-1$, we eventually obtain **final single-electron features**
$$
\mathbf{h}^{L\alpha}_j \quad \text{for each electron } j \text{ of spin } \alpha.
$$
Notice how the indices work:
- $\ell$ runs over layers and disappears at the end,
- $i$ or $j$ always refer to a specific electron within a spin sector,
- $\alpha,\beta$ tell you which spin sector that electron belongs to.

---

### From final features to orbitals

The final orbitals are built as a function of the last-layer features $\mathbf{h}^{L\alpha}_j$, plus some additional “envelope” factors that handle the long-range decay and cusp conditions. For each determinant index $k$, spin $\alpha$, orbital index $i$, and electron $j$ we define
$$
\begin{aligned}
\phi^{k\alpha}_i\big(\mathbf{r}^\alpha_j; \{\mathbf{r}^\alpha_{/j}\}; \{\mathbf{r}^{\bar{\alpha}}\}\big)
&= \left(\mathbf{w}^{k\alpha}_i \cdot \mathbf{h}^{L\alpha}_j + g^{k\alpha}_i\right) \\
&\quad\times \sum_{m} \pi^{k\alpha}_{im}
\exp\Big(
- \big|\mathbf{\Sigma}_{im}^{k\alpha} \big(\mathbf{r}^{\alpha}_j - \mathbf{R}_m\big)\big|
\Big).
\end{aligned}
$$
Here:
- $\mathbf{w}^{k\alpha}_i$ and $g^{k\alpha}_i$ are learnable linear parameters for the “MLP part” of the orbital,
- the sum over $m$ is an “envelope” over nuclei (or centers),
- $\pi^{k\alpha}_{im}$ and $\mathbf{\Sigma}^{k\alpha}_{im}$ are learnable coefficients and matrices controlling the exponential decay around nucleus $m$.

All these parameters depend on the indices:
- $k$ selects which determinant in the sum,
- $i$ selects which orbital (row in the determinant),
- $\alpha$ selects the spin sector,
- $m$ selects which nuclear center in the envelope.

The dependence on all other electrons is hidden inside $\mathbf{h}^{L\alpha}_j$, which was built from the full set of positions $\{\mathbf{r}^\uparrow\},\{\mathbf{r}^\downarrow\}$ through the deep network.

---

### Assembling the spin-separated determinants

For each determinant index $k$ and spin sector $\alpha\in\{\uparrow,\downarrow\}$, we build a matrix
$$
D^{k\alpha}_{ij}
= \phi^{k\alpha}_i\big( \mathbf{r}^\alpha_j; \{\mathbf{r}^\alpha_{/j}\}; \{\mathbf{r}^{\bar{\alpha}}\}\big),
$$
with:
- rows indexed by the orbital label $i = 1,\dots,n^\alpha$,
- columns indexed by the electron label $j = 1,\dots,n^\alpha$ (with that spin).

Taking the determinant gives a properly antisymmetric function of the positions of electrons **with that spin**:
$$
\det\big[D^{k\alpha}\big]
= \det\left[\phi^{k\alpha}_i(\mathbf{r}^\alpha_j; \{\mathbf{r}^\alpha_{/j}\}; \{\mathbf{r}^{\bar{\alpha}}\})\right].
$$

For the full wavefunction, we combine spin-up and spin-down blocks:
$$
\begin{aligned}
\psi(\mathbf{r}^\uparrow_1,\ldots,\mathbf{r}^\uparrow_{n^\uparrow},
     \mathbf{r}^\downarrow_1,\ldots,\mathbf{r}^\downarrow_{n^\downarrow})
= \sum_{k} \omega_k \;&
\det\left[\phi^{k \uparrow}_i(\mathbf{r}^\uparrow_j; \{\mathbf{r}^\uparrow_{/j}\}; \{\mathbf{r}^\downarrow\})\right] \\
&\times
\det\left[\phi^{k \downarrow}_i(\mathbf{r}^\downarrow_j; \{\mathbf{r}^\downarrow_{/j}\}; \{\mathbf{r}^\uparrow\})\right].
\end{aligned}
$$
We have don't explained why we can write the determinant as that product.
In electronic structure, when we separate spin and spatial parts using spin-orbitals, the full Slater determinant over all electrons factorizes into the product of: one determinant involving only spin-up electrons, another determinant involving only spin-down electrons.

Each of these determinants is antisymmetric under exchange of two electrons **with the same spin**. The overall wavefunction constructed as the product of a spin-up determinant and a spin-down determinant is antisymmetric under exchange of any two electrons (when you take into account the spin labels). FermiNet keeps this structure and lets each block be represented by a powerful neural network ansatz for the orbitals.

Up to this point the building blocks are just MLP layers (with residual connections and special feature mixing), but the careful indexing
- $(i,\alpha)$ for “which electron/spin”,
- $j$ for summation over electrons,
- $\ell$ for layers,
- $k$ for determinant index,
is what guarantees that the final object has the correct permutation symmetry and antisymmetry required for a fermionic wavefunction.

![[ferminet.png|280x315]]

(Architecture of Fermite Source  )

## Jastrow Factor for Psi Former

[[Psi Former Ansatz]]. @vonglehn2023selfattentionansatzabinitioquantum

The Psiformer wavefunction has the usual Slater–Jastrow structure
$$
\Psi_{\theta}(\mathbf{x})
=
\exp\big(\mathcal{J}_{\theta}(\mathbf{x})\big)\,
\sum_{k=1}^{N_{\det}}\det[\boldsymbol{\Phi}^{k}_{\theta}(\mathbf{x})],
$$
where $\mathbf{x} = (x_1,\dots,x_N)$ is the collection of all $N$ electron states 
$$
x_i = (\mathbf{r}_i,\sigma_i), \qquad \mathbf{r}_i \in \mathbb{R}^3,\;\sigma_i \in \{\uparrow,\downarrow\}.
$$

- $\mathcal{J}_\theta:(\mathbb{R}^{3}\times \{\uparrow,\downarrow\})^{N}\to \mathbb{R}$ is the **Jastrow factor**, encoding (here) only electron–electron cusp information.
- $\boldsymbol{\Phi}^k_\theta$ is the matrix of (spin-)orbitals for determinant $k$.

In Psiformer, the Jastrow factor is *very* simple: it has only two learnable parameters, one for parallel-spin pairs and one for antiparallel-spin pairs:
$$
\mathcal{J}_{\theta}(\mathbf{x})
=
\sum_{i<j;\,\sigma_{i}=\sigma_{j}}
-\frac{1}{4}\frac{\alpha^{2}_{\mathrm{par}}}{\alpha_{\mathrm{par}}+\lvert \mathbf{r}_{i}-\mathbf{r}_{j} \rvert }
\;+\;
\sum_{i,j;\,\sigma_{i}\neq \sigma_{j}}
-\frac{1}{2}\frac{\alpha^{2}_{\mathrm{anti}}}{\alpha_{\mathrm{anti}}+\lvert \mathbf{r}_{i}-\mathbf{r}_{j} \rvert }.
$$

- $\alpha_{\mathrm{par}}$ controls the strength of the Jastrow for **same-spin** electron pairs.
- $\alpha_{\mathrm{anti}}$ does the same for **opposite-spin** pairs.

This Jastrow is responsible for enforcing the electron–electron cusp conditions. The neural network itself (the Psiformer) only sees **electron–nucleus** information in its attention stream; all explicit $|\mathbf{r}_i-\mathbf{r}_j|$ dependence lives in $\mathcal{J}_\theta$.

---

## Applying Attention to Fermi Net (Psiformer-style)

Conceptually, Psiformer is “FermiNet with the two-electron stream replaced by self-attention”, we can see it clearly doing.

- FermiNet: separate one-electron and two-electron feature streams, then mix.
- Psiformer: a **single stream** of self-attention layers on electron–nuclear features only; electron–electron features enter only via the Jastrow.

We now explain the indices and equations carefully.
### Indices

We will use:

- $i,j = 1,\dots,N$: electron indices.
- $I = 1,\dots,N_{\text{nuc}}$: nucleus index.
- $\sigma_i \in \{\uparrow,\downarrow\}$: spin of electron $i$.
- $\ell = 0,\dots,L-1$: layer index in the Psiformer.
- $h = 1,\dots,H$: attention head index.
- $k = 1,\dots,N_{\det}$: determinant index.
- $d$: hidden dimension of the per-electron feature vectors.
So at each layer $\ell$, each electron $i$ carries a feature (hidden state)
$$
\mathbf{h}_i^{\ell} \in \mathbb{R}^{d}.
$$
### Input features and initial hidden states

Psiformer only uses **electron–nuclear** features (plus spin) as input to the attention stack.
For each electron $i$:
1. Let $\mathbf{R}_I$ be nuclear positions.
2. Build raw features by concatenating for all $I$:
   - some function of $\mathbf{r}_i - \mathbf{R}_I$ (relative position),
   - $|\mathbf{r}_i - \mathbf{R}_I|$ (distance),
   - and the spin as a scalar (e.g. $\sigma_i = +1$ for $\uparrow$, $-1$ for $\downarrow$).

In the paper they rescale the electron–nucleus vectors so that large distances grow only logarithmically, but at the level of notation we can just write
$$
\mathbf{f}_i^{0} \in \mathbb{R}^{d_{\text{in}}}
\quad\text{(electron–nucleus features + spin)}.
$$
These are then mapped into the model hidden dimension by a linear layer
$$
\mathbf{h}_{i}^{0} = \mathbf{W}^{0}\,\mathbf{f}_{i}^{0},
$$
where $\mathbf{W}^0 \in \mathbb{R}^{d \times d_{\text{in}}}$ is learned.
So:
- index $i$ is “which electron”,
- superscript $0$ means “before any attention layers.”
### One self-attention layer

At layer $\ell$, we have all electron hidden states
$$
\mathbf{h}_1^{\ell},\dots,\mathbf{h}_N^{\ell}.
$$

For each **head** $h$ and electron $i$, we compute:

- Query:
  $$
  \mathbf{q}^{\ell h}_i = \mathbf{W}^{\ell h}_q \mathbf{h}^{\ell}_i
  $$
- Key:
  $$
  \mathbf{k}^{\ell h}_i = \mathbf{W}^{\ell h}_k \mathbf{h}^{\ell}_i
  $$
- Value:
  $$
  \mathbf{v}^{\ell h}_i = \mathbf{W}^{\ell h}_v \mathbf{h}^{\ell}_i
  $$

Here each $\mathbf{W}^{\ell h}_q,\mathbf{W}^{\ell h}_k,\mathbf{W}^{\ell h}_v$ is a learned matrix, shared across all electrons $i$, but specific to layer $\ell$ and head $h$.

Then the **self-attention output for electron $i$, head $h$** is
$$
\mathbf{A}^{\ell h}_i
=
\sum_{j=1}^{N}
\underbrace{
\frac{\exp\big((\mathbf{q}^{\ell h}_i)^{\mathsf T}\mathbf{k}^{\ell h}_j / \sqrt{d_k}\big)}
     {\sum_{j'=1}^N \exp\big((\mathbf{q}^{\ell h}_i)^{\mathsf T}\mathbf{k}^{\ell h}_{j'} / \sqrt{d_k}\big)}
}_{\text{attention weight from } i \text{ to } j}
\mathbf{v}^{\ell h}_j.
$$

- $j$ runs over “all other electrons,” so electron $i$ “looks” at all others via attention.
- $d_k$ is the key/query dimension (usually $d_k = d/H$ or similar).

This is exactly your
$$
A^{\ell}_{h} = [\text{SelfAttn}(\mathbf{h}_1^\ell,\dots,\mathbf{h}_N^\ell;\mathbf{W}^{\ell h}_q,\mathbf{W}^{\ell h}_k,\mathbf{W}^{\ell h}_v)],
$$
but now written explicitly with indices $i$ and $j$.

Next, we **concatenate over heads** for each electron:
$$
\mathbf{A}^{\ell}_i = \text{concat}_{h=1}^H\big[\mathbf{A}^{\ell h}_i\big]
\in \mathbb{R}^{Hd_v},
$$
where $d_v$ is the value dimension of each head.

This is your
$$
A^{\ell} = \text{concat}_{h}[A_{h}],
$$
but again with the electron index $i$ made explicit.

### Residual projection and MLP

We then map the concatenated attention output back to the hidden dimension and add a residual connection:
$$
\mathbf{f}_{i}^{\ell+1}
=
\mathbf{h}_{i}^{\ell}
+
\mathbf{W}_{o}^{\ell}\,\mathbf{A}^{\ell}_i,
$$
where $\mathbf{W}_{o}^{\ell}$ is a learned matrix.

Then we pass this through a small MLP (just a linear + $\tanh$ here), again with a residual:
$$
\mathbf{h}_{i}^{\ell+1}
=
\mathbf{f}_{i}^{\ell+1}
+
\tanh\big(\mathbf{W}^{\ell+1}\mathbf{f}_{i}^{\ell+1} + \mathbf{b}^{\ell+1}\big).
$$

So a full Psiformer layer $\ell$ is:

1. Self-attention: $\{\mathbf{h}_i^\ell\} \to \{\mathbf{A}^\ell_i\}$.
2. Linear + residual: $\{\mathbf{A}^\ell_i\} \to \{\mathbf{f}_i^{\ell+1}\}$.
3. MLP + residual: $\{\mathbf{f}_i^{\ell+1}\} \to \{\mathbf{h}_i^{\ell+1}\}$.

Repeat this for $\ell=0,\dots,L-1$ and you get **final hidden states**
$$
\mathbf{h}_j^{L} \quad \text{for each electron } j.
$$
Here:
- $L$ = number of layers in the Psiformer.
- For each layer, $i$ indexes the electron the output belongs to, $j$ indexes electrons we attend over.
- $h$ indexes different heads in multi-head attention.

### From hidden states to orbitals and determinants

From the final hidden states $\mathbf{h}_j^L$, we build the spin-orbital matrix for each determinant $k$.

For each determinant index $k$ and orbital index $i$, define a **linear “orbital head”**:
$$
\tilde{\phi}^{k}_i(x_j)
=
\mathbf{w}^{k}_i \cdot \mathbf{h}^{L}_j
+
g^{k}_i,
$$
where $\mathbf{w}^{k}_i$ and $g^{k}_i$ are learned. The dependence on spin $\sigma_j$ and all other electrons is implicit in $\mathbf{h}_j^L$: the self-attention layers have already mixed that information in.

Then we multiply by an **envelope** to enforce the correct asymptotic decay:
$$
\Omega^{k}_{ij}
=
\sum_{m}
\pi^{k}_{im}
\exp\big(
- \big|\mathbf{\Sigma}^{k}_{im}(\mathbf{r}_j - \mathbf{R}_m)\big|
\big),
$$
where
- $m$ indexes nuclei (or “envelope centers”),
- $\pi^{k}_{im}$ and $\mathbf{\Sigma}^{k}_{im}$ are learned parameters.

The final spin-orbital entries are
$$
\Phi^{k}_{ij}
=
\Omega^{k}_{ij}\,
\tilde{\phi}^{k}_i(x_j).
$$

Collecting these into the matrix
$$
\boldsymbol{\Phi}^k(\mathbf{x}) = 
\big[\Phi^{k}_{ij}\big]_{i,j=1}^N,
$$
we form the determinant
$$
\det[\boldsymbol{\Phi}^k(\mathbf{x})]
=
\det\big[\Phi^{k}_{ij}\big]
=
\det\big[\phi^{k}_i(x_j)\big],
$$
and finally the full Psiformer wavefunction
$$
\Psi_{\theta}(\mathbf{x})
=
\exp(\mathcal{J}_{\theta}(\mathbf{x}))
\sum_{k=1}^{N_{\det}}\det[\boldsymbol{\Phi}^{k}_{\theta}(\mathbf{x})].
$$

So the story in terms of indices is:

- $i$ = which **orbital** (row of the determinant).
- $j$ = which **electron** the orbital is evaluated on (column of the determinant).
- $k$ = which **determinant** in the sum.
- $\ell$ = which **layer** of the attention/MLP stack produced the hidden states.
- $h$ = which **attention head** participated in mixing information across electrons.
- $I,m$ = which **nucleus/center** is used for the envelope.

The self-attention layers are what let $\mathbf{h}_j^L$ depend on all other electrons in a flexible, learned way, while the determinant over $i,j$ and the Jastrow over $i,j$ enforce fermionic antisymmetry and cusp conditions.


![[psiformer.png|271x339]]

(Architecture of Psi Former, Source: Pfau et all)

# Methodology

To implement the code, the choose of the library is important.
The three options to implement this kind of matter are JAX, Tensor Flow and pytorch, each one with his advantages and disadvantages.
## Environment

For this project we are going to be using Pytorch due his user-friendly and support. Python, and several libraries as transformers from hugging face, a library who implements KFCA and like guide the implement of Fermi Net by google deepmind which is made it on TensorFlow

Project manager with UV.
## Training

Due the high computational power needed we are going to using GPUS and of course CUDA.
Is clear that we are going to use virtual GPUS, for that matter we have two option or well use a GPU via SSH or directly using services like Azure , Colab, or anothers matters. For simplicity we are going to use Colab services.
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