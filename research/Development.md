---
date: 2025-10-23 19:32
modified: 2025-11-08 10:08
---
# Development of a Transformed based architecture to solve the Time Independent Many Electron Schrodinger Equation

## Table of Contents
1. [Abstract](#Abstract)
2. [Introduction](#Introduction)
3. [Objectives](#Objectives)
4. [Theoretical Framework](#Theoretical%20Framework)
	1. [The problem](#The%20problem)
		1. [The Schrodinger Equation](#The%20Schrodinger%20Equation)
		2. [The many electron Schrodinger Equation](#The%20many%20electron%20Schrodinger%20Equation)
	2. [Approximating a solution](#Approximating%20a%20solution)
		1. [RayLeight Quotient](#RayLeight%20Quotient)
	3. [Using Deep Learning](#Using%20Deep%20Learning)
		1. [Fermi Net](#Fermi%20Net)
		2. [Transformers](#Transformers)
		3. [Psi Former](#Psi%20Former)
		4. [Loss function](#Loss%20function)
		5. [Optimizer](#Optimizer)
		6. [Flow of the architecture](#Flow%20of%20the%20architecture)
5. [Methodology](#Methodology)
	1. [Environment](#Environment)
	2. [Training](#Training)

---
# Abstract

With accurate solutions to the many electron Schrodinger equation all the chemistry could be derived from first principles. Try to find analytical is prohibitively hard due the intrinsic and chaotic relations between each component on a molecule (electrons and protons). Recently, due to its high flexibility deep learning approaches had been already used for this problem, FermiNet and Pauli Net are two good examples, these have advanced accuracy, yet computational cost or error typically grows steeply with system size, limiting applicability to larger molecules. They also lack of strong architectures designed to capture long-range electronic correlations with scalable attention. In this work I develop the Psiformer a transformer-based ansatz that couples scalable attention with physics-aware structure. I  formulate training via Variational MonteCarlo and the evaluation will be do it by comparing against another traditional methods.
# Introduction

The success of deep Learning across different fields like protein folding @jumper2021highly, visual modeling @dosovitskiy2021imageworth16x16words, and PDEs solvers @RAISSI2019686 . Motivated by these successes, the community has explored neural approaches for quantum many-body problems, seeking accurate and scalable approximations to the many-electron wave function.

The electronic structure problem remains challenging: the wave functions lives in $3N$-dimensional space, additionally it must satisfy certain properties due to physical laws (anti symmetry and sharp features). Traditional methods balance accuracy and lost, but often struggle on correlated systems.

specifically finding a good aproximmation for the Quantum Many-Body wave eqaution  the is one of those places where have shown that deep learning could overpass traditional methods @Luo_2019 , @Qiao_2020, but there is still many challenges specifically, the computational power needed for large molecules becomes prohibitively expensive. 
@ad

Tackling that problem the Transformer architecture had demonstrate that scaling laws are not so much complicated for him.

Motivated for that in this work I develop a transformer architecture called Psifomer. @vonglehn2023selfattentionansatzabinitioquantum  
# Objectives
- Obtain a model which is able to approximate the ground state state energy of the carbon atom.
- Compare our model with another state of the art methods to solve the many electrons Schrodinger equation respect the ground state energy.
- Look for improvements when try to tackle larger molecules. 
# Overview

This work is structured as follow: The theoretical framework introduces the foundations of quantum many-body theory, the structure of the Schrodinger equation for many electrons like also foundational concepts of Deep Learning that are going to be used in this specific work, we are going to talk also about the Transformer architecture and Fermi Net a architecture that use Neural Networks to solve the problem and this work is built up on.
The concepts presented in this section provide the physical and mathematical context for the proposed model.

The part where the model itself is introduced.

The methodology section details a brief construction of the **Psiformer** and the environment which is going to be use.
# Theoretical Framework
In order to solve the problem we have to grasp the physics laws that our solution have to follow.
### The Schrodinger Equation
The Schrodinger equation was presented in a series of publication made it by Schrodinger in the year 1916. He derived the time dependent equation:
We search the complex $\psi$ function called **wave function**, $\lvert \psi \rvert^{2}$ is a probability distribution telling the probability of find a particle (electron) is a specific position.
This function is rule by the equation:
$$ i\hbar \frac{\partial \psi}{\partial t}=\hat{H}\psi $$
Where $i$ is the complex unit, $\hbar$ is the [[Reduced Planck Constant]] equals to 
The $\hat{H}$ is a Hermitian linear operator called the Hamiltonian which represents the total energy of the system

$$
\hat{H}=\vec{P}+V(x)
$$
Where $\vec{P}$ is the Linear Momentum Operator $V$ the potential energy of the system.
$\vec{P}$ takes the form of: @Zettili2009
$$
\vec{P}=-\frac{1}{2}\sum \nabla^{2}
$$

And $V$ depends on the specific system.
The time independent form could be derived from the time dependent form.
$$
\hat{H}\psi=E\psi
$$
Where $E$ is the total energy of the system.
### The many electron Schrodinger Equation

In quantum chemistry is regular used atomic units, the unit of distance is the Bohr Radious and the unit of energy is Hartree (Ha).

In its time-independent form the Schrodinger equation can be written as a eigenfunction equation.


$$ \hat{H}\psi(\mathbf{x}_{0},\dots ,\mathbf{x}_{n})=E\psi(\mathbf{x}_{1},\dots ,\mathbf{x}_{n}) $$
Where $\mathbf{x}_{i}=\{ \mathbf{r}_{i},\sigma \}$,  $\mathbf{r}_{i}$ is the position of each electron and protons and $\sigma \in \{ \uparrow.\downarrow \}$ is the spin.
In this case the potential energy of the system we have to consider the repulsion between the electrons
$$
U=\frac{1}{4\pi\varepsilon_{0}}\frac{e^{2}}{\lvert r_{i}-r_{j} \rvert }
$$
The attraction between protons and electrons.
$$
U=-\frac{1}{4\pi\varepsilon_{0}}\frac{eZ_{i}}{\lvert r_{i}-R_{i} \rvert }
$$
Where $Z_{i}$ is the atomic number.
And the repulsion between protons.
$$
U=\frac{1}{4\pi\varepsilon_{0}}\frac{Z_{i}Z_{j}}{\lvert Z_{i}-Z_{j} \rvert }
$$
Thus the potential energy is the sum of those three terms.
To avoid write those constants each time we use atomic [[Quantum Chemistry units|Atomic Units]].

The distances are.

The Hamiltonian using the [[Quantum Chemistry units]] becomes:
$$ \hat{H}=-\frac{1}{2}\sum \nabla^{2}+\sum \frac{1}{\lvert r_{i}-r_{j} \rvert }-\sum \frac{Z_{I}}{\lvert r_{i}-R_{I} \rvert }+\sum \frac{Z_{I}Z_{J}}{\lvert R_{i}-R_{j} \rvert } $$
Now the [[Fermi Dirac Statistics]] tell us that this solution of this equation should be **anti symmetric** this is:
$$
\psi(\dots,\mathbf{x}_{i},\dots,\mathbf{x}_{j},\dots)=-\psi(\dots ,\mathbf{x}_{j},\dots ,\mathbf{x}_{i},\dots)
$$
The potential energy becomes infinite when two electrons overlap , this could be formalized via the [[Kato Cusp Conditions]], a Jastrow factor $\exp(\mathcal{J})$. The explicit form of $\mathcal{J}$ depends on the.
$$
\lim_{ l \to 0 } \left( \frac{\partial \psi}{\partial r_{iI}} \right)=-Z\psi(r_{iI}=0)
$$
$$
\lim_{ l \to 0 } \left( \frac{\partial \psi}{\partial r_{ij}} \right)=\frac{1}{2}\psi(r_{ij}=0)
$$
Where $r_{iI}(r_{ij})$ is an electron-nuclear (electron-electron) distance, $Z_{I}$ is the nuclear charge of the $I\text{-th}$ nucleous and ave implies a spherical averaging over all directions.
## Approximating a solution

Find possible solution in the traditional way is prohibitively hard. So what people have doing and it seem that it becomes a success is guess that solution and using another techniques to improve the solution, to this guess solution we called **Ansatz**.

Once that you have your Ansatz, which normally depends on depends on certain parameters.

### Variational Monte Carlo
Once that you guess an **Ansatz** you optimize using the **rayleight quotient**.

$$
\mathcal{L}=\frac{\bra{\psi} \hat{H}\ket{\psi} }{\braket{ \psi | \psi } }=\frac{\int d\mathbf{r}\psi ^{*}(\mathbf{r})\hat{H}\psi(\mathbf{r})}{\int d\mathbf{r}\psi ^{*}(\mathbf{r})\psi(\mathbf{r})}
$$
So how we optimized this. Here appears [[Variational Quantum Monte Carlo]].
Which can be re-written as:
$$ E_{L}(x)=\Psi ^{-1}_{\theta}(x)\hat{H}\Psi_{\theta}(x) $$
$$ \mathcal{L}_{\theta}=\mathbb{E}_{x\sim \Psi^{2}_{\theta}}[E_{L}(x)] $$
And here we use [[Metropolis algorithm]] to work in real life.

## Using Deep Learning

They are a quite example of it.
examples @shangSolvingManyelectronSchrodinger2025 Related work
### Multi Layer Perceptron
A MLP is a nonlinear function $\mathcal{F}:\mathbb{R}^{\text{in}}\to \mathbb{R}^{\text{out}}$.  @nielsenNeuralNetworksDeep2015

A MLP could be see it like the composition of $L$ layers, the first layer is called the input layers, the last  output layer and the intermediates hidden layers.

Let $\mathbf{z}^{(l)}$ be a affine map of the follow form. $l\in \{ L,L-1,\dots,2 \}$
$$
\mathbf{z}^{(l)}=\mathbf{W}^{(l)}\mathbf{a}^{(l-1)}+\mathbf{b}^{(l)}
$$

Where $\mathbf{W}^{(l)}$ is the weight matrix and $\mathbf{b}^{(l)}$ the bias vector of the $l$ layer.  

Let $\sigma ^{(l)}$ be a nonlinear function of the $l$ layers (typically Softmax,  Relu, Tanh.)

$$ f^{(l)}=\sigma ^{(l)}\circ \mathbf{z}^{(l)} $$
$$
\mathcal{F}=f^{(L)}\circ f^{(L-1)}\circ\dots \circ f^{(1)}
$$

For our paremeters:

$$\{ \mathbf{W}^{(l)},\mathbf{b}^{(l)}\}_{l=2}^{L}=\theta$$

You train the MLP with a training data set using backpropatation a loss function anda optimizer. Additionally you can use regularization techniques to improve the performance of the MLP.
### Fermi Net
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

The elements of the determinant are obtained via

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


### Attention 

The first attention mechanism were introduced by using the follow:


### Self Attention

Here we use the dot product and heads. 
$$
\mathbf{o}_{t,i}=\sum_{j=1}^{t}\text{Softmax}\left( \frac{\mathbf{q}^{T}_{t,i}\mathbf{k}_{j,i}}{\sqrt{ d_{h} }} \right) \mathbf{v}_{j,i}
$$
$$
\mathbf{u}_{t}=W^{O}[\mathbf{o}_{t,1};\mathbf{o}_{t,2};\dots ;\mathbf{o}_{t,n_{h}}]
$$


### Transformers


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

[[Psi Former Ansatz]]. @vonglehn2023selfattentionansatzabinitioquantum
$$ \Psi_{\theta}(\mathbf{x})=\exp(\mathcal{J}_{\theta}(\mathbf{x}))\sum_{k=1}^{N_{\det}}\det[\boldsymbol{\Phi}_{\theta}^{k}(x)] $$

Where $\mathcal{J}_{\theta}$ is the [[Jastrow Factor for si Former]] and $\Phi$ are [[orbital for neural network fermi net|orbitals]]. 


Where $\mathcal{J}_{\theta}:(\mathbb{R}^{3}\times \{ \uparrow,\downarrow \})^{n}\to \mathbb{R}$

- So the question is how you define the outputs of that functions:
- [[Jastrow Factor]]
$$
\mathcal{J}_{\theta}(x)=\sum_{i<j;\sigma_{i}=\sigma_{j}}-\frac{1}{4}\frac{\alpha^{2}_{par}}{\alpha_{par}+\lvert \mathbf{r}_{i}-\mathbf{r}_{j} \rvert }+\sum_{i,j;\sigma_{i}\neq \sigma_{j}}-\frac{1}{2}\frac{\alpha^{2}_{anti}}{\alpha_{anti}+\lvert \mathbf{r}_{i}-\mathbf{r}_{j} \rvert }
$$

Architecture

![[psiformer.png|271x339]]

## Loss function

We are going to take the [[Rayleigh Quotient like Expectation Value]] like loss function.

## Optimizer 

[[Kroenecker factored Approximate Curvature]]

### Flow of the architecture

First compute:
$$ v_{h}=[\text{SelfAttn}(\mathbf{h}^{l}_{1},\dots,\mathbf{h}^{\ell}_{N};\mathbf{W}^{\ell h}_{q},\mathbf{W}^{\ell h}_{k},\mathbf{W}^{\ell h}_{v})] $$

Start with:

$$\mathbf{W}_{o}^{\ell}\text{concat}_{h}[\text{SelfAttn}(\mathbf{h}^{l}_{1},\dots,\mathbf{h}^{\ell}_{N};\mathbf{W}^{\ell h}_{q},\mathbf{W}^{\ell h}_{k},\mathbf{W}^{\ell h}_{v})]$$

With it you can obtain you hidden states, and then how you use it



With them you create the [[orbital for neural network fermi net]]

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

## Excerpt

Transformers are monsters finding relations between what you give them. Is tempting use them for emulate the relations between electrons and protons. How you can first encode the information of the electron's positions and second the attraction and repulsion between the particles? 