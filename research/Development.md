---
tags:
  - baby
date: 2025-10-23 19:32
modified: 2025-11-01 20:53
---
# Developing of a Transformed based architecture to solve the Time Independent Many Electron Schrodinger Equation
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

[[observations 1]]

# Abstract

With accurate solutions to the many electron Schrodinger equation all the chemistry could derived from first principles. Try to find analytical is prohibitively hard due the intrinsic relations between each component on a molecule. In this work I develop the use of a architecture based on the Transformer architecture to tackle this problem.

# Introduction

The success of deep Learning across different fields like protein folding  @jumper2021highly, visual modeling @dosovitskiy2021imageworth16x16words, ODEs solvers @RAISSI2019686 has sparked great interest from the scientific community to apply DL methods to their fields. 

Quantum Chemistry, specifically finding a good aproximmation for the Quantum Many-Body wave eqaution  the is one of those places where have shown that deep learning could overpass traditional methods @Luo_2019 , @Qiao_2020, but there is still many challenges specifically, the computational power needed for large molecules becomes prohibitively expensive. 

Tackling that problem the Transformer architecture had demonstrate that scaling laws are not so much complicated for him. Cite

Motivated for that in this work I develop a transformer architecture called Psifomer. @vonglehn2023selfattentionansatzabinitioquantum  

# Objectives

- Obtain a model which is able to replicate the energy ground states of certain atoms.
- Compare our model with another State of the art methods to solve the Many electrons Schrodinger equation


# Overview

I provide an outline of the model architecture and procedure 
In theoretical frame work we  will and in methodology we will 


# Theoretical Framework

In order to solve the problem we have to grasp the physics laws that our solution have to follow, 

## The problem

We consider the follow:

### The Schrodinger Equation

The schrodinger equation was presented in a series of publication made it by Schrodinger in the year 1916.

It was received pretty well by the scientific comunnity. 

And its relevance is high, in principle it is able to explain all the atomic phenomena and all the facts of chemical bindings.

### The many electron Schrodinger Equation

In quantum chemistry is regular used atomic units, the unit of distance is the Bohr Radious and the unit of energy is Hartree (Ha).

In its time-independent form the Schrodinger equation can be written as a eigenfunction equation.


$$ \hat{H}\psi(\mathbf{x}_{0},\dots ,\mathbf{x}_{n})=E\psi(\mathbf{x}_{1},\dots ,\mathbf{x}_{n}) $$

Where $\hat{H}$ is a Hermitian linear operator called the Hamiltonian and the scalar eigenvalue $E$ corresponds to the energy of a particular solution.

$$
U=\frac{1}{4\pi\varepsilon_{0}}\frac{e^{2}}{\lvert r_{i}-r_{j} \rvert }
$$

Using atomic units we [[Quantum Chemistry units|Atomic Units]]:

The Hamiltonian using the [[Quantum Chemistry units]] becomes:

$$ \hat{H}=-\frac{1}{2}\sum \nabla^{2}+\sum \frac{1}{\lvert r_{i}-r_{j} \rvert }-\sum \frac{Z_{I}}{\lvert r_{i}-R_{I} \rvert }+\sum \frac{Z_{I}Z_{J}}{\lvert R_{i}-R_{j} \rvert } $$

Where $Z_{I}$ are the [[atomic number]] $r_{i}$ is the distance from a reference frame 

Now the [[Fermi Dirac Statistics]] tell us that this solution of this equation should be **anti symmetric** this is:

$$
\psi(\dots,\mathbf{x}_{i},\dots,\mathbf{x}_{j},\dots)=-\psi(\dots ,\mathbf{x}_{j},\dots ,\mathbf{x}_{i},\dots)
$$

The potential energy becomes infinite when two electrons overlap , this could be formalized via the [[Kato Cusp Conditions]], a Jastrow factor $\exp(\mathcal{J})$. The explicit form of $\mathcal{J}$ depends on the author.




## Approximating a solution

Find possible solution in the traditional way is prohibitively hard. So what people have doing and it seem that it becomes a success is guess that solution and using another techniques to improve the solution, to this guess solution we called **Ansatz**.

Once that you have your Ansatz, which normally depends on depends on certain parameters.





### Variational Monte Carlo

Once that you guess an **Ansatz** you optimize using the **rayleight quotient**.

$$ \mathcal{L}=\frac{\bra{\psi} \hat{H}\ket{\psi} }{\braket{ \psi | \psi } }=\frac{\int d\mathbf{r}\psi ^{*}(\mathbf{r})\hat{H}\psi(\mathbf{r})}{\int d\mathbf{r}\psi ^{*}(\mathbf{r})\psi(\mathbf{r})} $$


So how we optimized this. Here appears [[Variational Quantum Monte Carlo]].

Which can be re-written as:
$$ E_{L}(x)=\Psi ^{-1}_{\theta}(x)\hat{H}\Psi_{\theta}(x) $$

$$ \mathcal{L}_{\theta}=\mathbb{E}_{x\sim \Psi^{2}_{\theta}}[E_{L}(x)] $$

And here we use [[Metropolis algorithm]] to work in real life.

## Using Deep Learning

They are a quite example of it.

examples @shangSolvingManyelectronSchrodinger2025 Related work

### Neural Networksee

### RNN


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
\begin{multline}
    \phi^{k\alpha}_i(\mathbf{r}^\alpha_j; \{\mathbf{r}^\alpha_{/j}\}; \{\mathbf{r}^{\bar{\alpha}}\}) =
    \left(\mathbf{w}^{k\alpha}_i \cdot \mathbf{h}^{L\alpha}_j + g^{k\alpha}_i\right)\\
	\sum_{m} \pi^{k\alpha}_{im}\mathrm{exp}\left(-|\mathbf{\Sigma}_{im}^{k \alpha}(\mathbf{r}^{\alpha}_j-\mathbf{R}_m)|\right),
\end{multline}
$$

$$ \phi ^{k\alpha}_{i}(\mathbf{r}^{\alpha}_{j};\{ \mathbf{r}^{\alpha}_{/j} \};\{ \mathbf{r}^{\bar{\alpha}} \})=(\mathbf{w}^{k\alpha}_{i}\cdot \mathbf{h}^{L\alpha}_{j}+g^{k\alpha}_{i})\sum_{m}\pi_{im}^{k\alpha}\exp\left( -\left\lvert \Sigma _{im}^{k\alpha}(\mathbf{r}^{\alpha}_{j}-\mathbf{R}_{m})\right\rvert  \right)$$.

$$
​￼\begin{align}
	\psi(\mathbf{r}^\uparrow_1,\ldots,\mathbf{r}^\downarrow_{n^\downarrow}) = \sum_{k}\omega_k &\left(\det\left[\phi^{k \uparrow}_i(\mathbf{r}^\uparrow_j; \{\mathbf{r}^\uparrow_{/j}\}; \{\mathbf{r}^\downarrow\})\right]\right.\\&\left.\hphantom{\left(\right.}\det\left[\phi^{k\downarrow}_i(\mathbf{r}^\downarrow_j; \{\mathbf{r}^\downarrow_{/j}\});
	\{\mathbf{r}^\uparrow\};\right]\right).
\end{align}
$$

You com

![[ferminet.png|280x315]]

Motivated for the antisymmetry and the Kato cusp conditions our **Ansatz** take the form of: [
### Transformers

There exist several architectures that I can use Recurrent Neural Network, Long Short Term Memory. 

@Vaswani2017 

Recurrent Neural Network are: [[Recurrent Neural Network]]
And long short term memory are: [[Long Short Memory]]

Why on earth I would use [[Transformer]]? They are extremely good finding relations between its elements. And the best is that scale well due its [[Transform Architecture]]

Attention mechanism appear with @bahdanau2014neural but it didn't work so:

- [[Attention mechanism]]
- [[Self attention mechanism on one head]]
- [[Multi-head attention]]
$$
\mathbf{o}_{t,i}=\sum_{j=1}^{t}\text{Softmax}\left( \frac{\mathbf{q}^{T}_{t,i}\mathbf{k}_{j,i}}{\sqrt{ d_{h} }} \right) \mathbf{v}_{j,i}
$$
$$
\mathbf{u}_{t}=W^{O}[\mathbf{o}_{t,1};\mathbf{o}_{t,2};\dots ;\mathbf{o}_{t,n_{h}}]
$$

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

### Loss function

We are going to take the [[Rayleigh Quotient like Expectation Value]] like loss function.

### Optimizer 

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

[^1]: Schrodinger Reference.

--- 

## Excerpt

Transformers are monsters finding relations between its basis part if we use them for emulate the relations between electrons and protons? That sounds a good idea? Let's test it and see what's happen.