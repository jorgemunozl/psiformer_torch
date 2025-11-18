[1mdiff --git a/research/Development.md b/research/Development.md[m
[1mindex fa748eb..14dbcbf 100644[m
[1m--- a/research/Development.md[m
[1m+++ b/research/Development.md[m
[36m@@ -1,6 +1,6 @@[m
 ---[m
 date: 2025-10-23 19:32[m
[31m-modified: 2025-11-16 10:33[m
[32m+[m[32mmodified: 2025-11-17 19:47[m
 ---[m
 # Development of a Transformed based architecture to solve the Time Independent Many Electron Schrodinger Equation[m
 [m
[36m@@ -47,7 +47,7 @@[m [mThe concepts presented in this section provideS the physical and mathematical ba[m
 All the physics and methods related necessary.[m
 ### The Schrodinger Equation[m
 [m
[31m-The Schrodinger equation was presented in a series of publications made it by Erwin Schrodinger in the year 1926. There we search the complex function $\psi$ called **wave function**, for a non relativistic spinless single particle this function depends on the position of the particle $\mathbf{\vec{r}}$ and time $t$ $(\psi(\mathbf{\vec{r}},t))$, the quantity $\lvert \psi (\mathbf{r},t)\rvert^{2}$ is the **probability density** to find the particle near $\mathbf{r}$ at time $t$.[m
[32m+[m[32mThe Schrodinger equation was presented in a series of publications made it by Erwin Schrodinger in the year 1926. There we search the complex function $\psi$ which lives on a Hilbert space $\mathcal{H}$ called **wave function**, for a non relativistic spinless single particle this function depends on the position of the particle $\mathbf{\vec{r}}$ and time $t$ $(\psi(\mathbf{\vec{r}},t))$, the quantity $\lvert \psi (\mathbf{r},t)\rvert^{2}$ is the **probability density** to find the particle near $\mathbf{r}$ at time $t$.[m
 [m
 Guided by de Broglie's discover of the wave particle duality and a very smart intuition Schrodinger proposed the time dependent equation (TDSE):[m
 $$ i\hbar \frac{\partial \psi}{\partial t}=\hat{H}\psi $$[m
[36m@@ -77,7 +77,7 @@[m [mWhere $E$,the energy of the system, is constant. You also obtain that the spatia[m
 $$[m
 \hat{H}R(\mathbf{\vec{r}})=ER(\mathbf{\vec{r}})[m
 $$[m
[31m-We usually represent $R$ as $\psi$.[m
[32m+[m[32mWe usually represent $R$ as $\psi$. And in this work we are going to only focus in the TDSE, when we treat with constant energy, the electron are almost always found near the lowest energy state, known as the ground state. Solutions with higher energy known as excited states are relevant to photochemistry , but in this work we will restrict our attention to ground states.[m
 ### The many electron Schrodinger Equation[m
 When we are considering more than one single particle we consider the spin $(\sigma)$ and the interaction between particles. Thus in its time-independent form the Schrodinger equation can be written as an eigen value problem:[m
 $$ \hat{H}\psi(\mathbf{x}_{0},\dots ,\mathbf{x}_{n})=E\psi(\mathbf{x}_{1},\dots ,\mathbf{x}_{n}) $$[m
[36m@@ -168,7 +168,6 @@[m [m$$[m
 Where $r_{iI}(r_{ij})$ is an electron-nuclear (electron-electron) distance, $Z_{I}$ is the nuclear charge of the $I\text{-th}$ nucleous and ave implies a spherical averaging over all directions.[m
 [[Kato Cusp Conditions]].[m
 [m
[31m-[m
 So that is the problem we need to find a $\psi$ such that it satisfies all those [m
 ### Approximations to the problem[m
 [m
[36m@@ -186,52 +185,62 @@[m [m$$[m
 $$[m
 [m
 This is the form that we are going to work with; a second useful approach to the problem is use an **Ansatz**, which is a guess solution guided by intuition, this normally depend on certain number of parameters, then the problem becomes on optimize this **Ansatz**.[m
[32m+[m
[32m+[m[32mAnother important matter are the complex numbers, we should to work with them, no! Why is that?[m
 ### Rayleigh Quotient[m
[31m-We need a way to measure how well our Ansatz $\psi_{\theta}$. For that matter we use the **Rayleigh quotient**.[m
[31m-If $A$ is an operator and $x$ is a state, the number:[m
[32m+[m[32mWe need a way to measure how well our Ansatz $\psi_{\theta}$ is doing, if our ansatz is bad then don't reflect the truly form of the system, Deep learning approachs use data sets to train their ansatz (model initialized on randoms parameters) in this case we don't need any external but physical principles.[m
[32m+[m[32mLets first introduce the **Rayleigh quotient**. If $A$ is an operator and $x$ is a state, the number:[m
 $$[m
 R_{A}(x)=\frac{\braket{ x |A|x  }}{\braket{ x | x } }[m
 $$[m
[31m-is the **expectation value** of that operator in that state. For us, if $\psi$ is a wave function and $\hat{H}$ the **Hamiltonian**, then the Rayleigh quotient:[m
[32m+[m[32mis the **expectation value** of that operator in that state. The important for us is that if $\psi$ is a wave function and $\hat{H}$ the **Hamiltonian**, then the Rayleigh quotient:[m
 $$[m
 R_{\hat{H}}(\psi)=\frac{\braket{ \psi | \hat{H}| \psi}}{\braket{ \psi | \psi } }[m
 $$[m
 is the average (expected) energy of the system when it is in the state $\psi$. [m
[31m-[m
 ### Variational Principle [m
 [m
[32m+[m[32mThe [[Variational Principle]]  states that the expectation value for the binding energy obtained using an approximate wave function and the exact Hamiltonian operator will be higher than or equal to the true energy for the system. This idea is really powerful. When implemented it permits us to find the best approximate wavefunction from a given wavefunction that contains one or more adjustable parameters, called a trial wavefunction. A mathematical statement of the variational principle is.[m
[32m+[m[32m$$[m
[32m+[m[32mR_{\hat{H}}(\psi_{\text{ansatz}})\geq R_{\hat{H}}(\psi_{\text{true}})[m
[32m+[m[32m$$[m
[32m+[m[32mThe true ground-state wave function $\psi_{\text{true}}$ is the one that minimizes the rayleigh quotient.[m[41m [m
[32m+[m[32m$$[m
[32m+[m[32m\psi_{0}=\underset{\psi}{\text{argmin}}( R_{\hat{H}}(\psi))[m
[32m+[m[32m$$[m
[32m+[m[32mSo if we minimize the rayleigh quotient of our ansatz we are going to be more near of the true wave function.[m
[32m+[m[32m### Variational Monte Carlo[m[41m [m
[32m+[m
[32m+[m[32mTo the process that we are going to use for optimize our ansatz is called Variational Monte Carlo (VMC). Now we can see to the rayleigh quotient as a loss function $\mathcal{L}$ with the form of.[m
 $$[m
 \mathcal{L}(\Psi_{\theta})=\frac{\bra{\Psi_{\theta}} \hat{H}\ket{\Psi_{\theta}} }{\braket{ \Psi_{\theta} | \Psi _{\theta}} }=\frac{\int d\mathbf{r}\Psi ^{*}(\mathbf{r})\hat{H}\Psi(\mathbf{r})}{\int d\mathbf{r}\Psi ^{*}(\mathbf{r})\Psi(\mathbf{r})}[m
 $$[m
[31m-Evaluate that integral is hard, for that first we define:[m
[32m+[m[32mEvaluate that integral is hard, another smart approach is the follow, define a probability distribution $p_{\theta}$ with the follow form:[m
 $$[m
[31m-p_{\theta}(x)=\frac{\Psi^{2}_{\theta}(x)}{\int dx'\Psi^{2}_{\theta}(x')}[m
[32m+[m[32mp_{\theta}(x)=\frac{\lvert \Psi_{\theta} (x)\rvert ^{2}}{\int dx'\Psi^{2}_{\theta}(x')}[m
 $$[m
[31m-Defining the Local Energy:[m
[32m+[m[32mRealize that for compute $p_{\theta}(x)$ for a specific $x$ is complicated due the integral that appears, this is going to be important later. Defining the local energy $E_{L}$ with:[m
 $$[m
 E_{L}(x)=\Psi ^{-1}_{\theta}(x)\hat{H}\Psi_{\theta}(x)[m
 $$[m
[32m+[m[32mThen the loss becomes:[m
 $$[m
 \mathcal{L}(\Psi_{\theta})=\int \frac{ \hat{H}\Psi_{\theta}(x)}{\Psi_{\theta}(x)}p_{\theta}(x)dx[m
 $$[m
[32m+[m[32mWhich is the expected value of the local energy:[m
 $$[m
 \mathcal{L}(\Psi_{\theta})=\mathbb{E}_{x\sim p_{\theta}}[E_{L}(x)][m
 $$[m
[31m-To optimize our wave function we need evaluate that expectation and obtain the derivative.[m
[31m-[m
[31m-To that set of techniques for obtain that we call **Quantum Monte Carlo**, there are differents approaches like Diffusion Monte Carlo, Path Integral Monte Carlo but in this case we are going to use Variational Montecarlo.[m
[31m-[m
[31m-Samples from that distribution and $E_{L}(x)$. We already have $E_{L}(x)$ [m
[31m-[m
[31m-Whith many samples we can estimate:[m
[32m+[m[32mTo optimize our wave function we know by deep learning that we need evaluate that expectation and obtain the derivative for back propagation, how we make that?[m
[32m+[m[32mWe are going to use the the Monte Carlo estimatior:[m
 $$[m
[31m-\mathbf{R}_{1},\dots,\mathbf{R}_{M}\sim p_{\theta}(\mathbf{R})[m
[32m+[m[32m\mathcal{L}_{\theta}=\mathbb{E}_{x\sim p_{\theta}}[E_{L}(x)]\approx \frac{1}{M}\sum_{i=1}^{M} E_{L}(\mathbf{R}_{k})[m
 $$[m
[31m-Using the Monte Carlo estimation:[m
[32m+[m[32mWhit $\mathbf{R}_{k}$ are samples from the $p_{\theta}$ probability distribution. This is[m
 $$[m
[31m-\mathcal{L}_{\theta}=\mathbb{E}_{x\sim p_{\theta}}[E_{L}(x)]\approx \frac{1}{M}\sum_{i=1}^{M} E_{L}(\mathbf{R}_{k})[m
[32m+[m[32m\mathbf{R}_{1},\dots,\mathbf{R}_{M}\sim p_{\theta}(\mathbf{R})[m
 $$[m
[31m-Where we can write:[m
[32m+[m[32mWe obtain $E_{L}$ using:[m
 $$[m
 E_{L}(\mathbf{R}_{k})=\frac{\hat{H}\psi(\mathbf{R}_{k})}{\psi(\mathbf{R}_{k})}=-\frac{1}{2}\frac{\nabla^{2}\psi(\mathbf{R_{k}})}{\psi(\mathbf{R}_{k})}+V(\mathbf{R}_{k})[m
 $$[m
[36m@@ -248,10 +257,30 @@[m [m$$[m
 \nabla _{\theta}\mathcal{L}=2\mathbb{E}_{x\sim \Psi^{2}}[(E_{L}(x)-\mathbb{E}_{x'\sim\Psi^{2}}[E_{L}(x')])\nabla \log \lvert \Psi(x) \rvert ][m
 $$[m
 ### Metropolis Hastings (MH) Algorithm[m
[32m+[m[32mTo obtain samples from the probability distribution $p_{\theta}$ we are going to use the [[Metropolis Hasting algorithm]]. (MH)[m
 [m
[31m-Goal: obtain many samples from  [[Metropolis Hasting algorithm]][m
[32m+[m[32mMH is a [[Markov Chain Monte Carlo]] (MCMC) method used to obtain a sequence of random samples from a probability distribution. The reason to use this method over another well knows methods (e.g. example) is that MH don't suffer of the [[Curse of Dimensionality]][m[41m [m
[32m+[m[32mthis is, it remains strong while increase the dimension of the problem, and since we are going to be working on dimension around ten is efficient use this method. The algorithm works like follow:[m
 [m
[31m-MH is a [[Markov Chain Monte Carlo]] (MCMC) method used to obtain a sequence of random samples from a probability distribution. Sota method to sample from high dimensional distributions.[m
[32m+[m[32m1. Take a initial configuration $\mathbf{X}_{0}\in E$ arbitrary:[m
[32m+[m[32m2. Propose $\mathbf{X}'=\mathbf{X}_{0}+\eta$ ,where $\eta \sim q(\eta)$, $q$ is a probability density on $E$ called **proposal kernel**. In our case we are going to a [[symmetric Gaussian]].[m
[32m+[m[32m3. Compute the quantity:[m
[32m+[m[32m$$[m
[32m+[m[32mA(\mathbf{X_{0}}, \mathbf{X}')=\text{min}\left( 1,\frac{\rho(\mathbf{X}')}{\rho(\mathbf{X}_{0})} \frac{q(\mathbf{X}'-\mathbf{X}_{0})}{q(\mathbf{X}_{0}-\mathbf{X}')}\right)[m
[32m+[m[32m$$[m
[32m+[m[32mWhere $\rho$ is the target distribution where we want sample, In the case where $q$ is symmetric, this simplifies to:[m
[32m+[m[32m$$[m
[32m+[m[32mA(\mathbf{X}_{0},\mathbf{X}')=\text{min}\left( 1,\frac{\rho(\mathbf{X})}{\rho(\mathbf{X}_{0})} \right)[m
[32m+[m[32m$$[m
[32m+[m[32mNote that in our case $\rho$ is equal to $p_{\theta}$, we said that compute the integral factor remains a challenge, but in this case we have it.[m
[32m+[m[32m$$[m
[32m+[m[32m\frac{p_{\theta}(\mathbf{X}')}{p_{\theta}(\mathbf{X}_{0})}=\frac{\lvert \psi_{\theta}(\mathbf{X}') \rvert ^{2}/\int \lvert \psi_{\theta} \rvert ^{2}dx}{\lvert \psi_{\theta}(\mathbf{X}_{0}) \rvert ^{2}/\int \lvert \psi_{\theta} \rvert ^{2}dx}=\frac{\lvert \psi_{\theta}(\mathbf{X'}) \rvert ^{2}}{\lvert \psi_{\theta}(\mathbf{X}_{0}) \rvert ^{2}}[m
[32m+[m[32m$$[m
[32m+[m[32m4. Generate a uniform number $U\in[0,1]$[m
[32m+[m[32m5. If: $U<A(\mathbf{X}_{0}\to \mathbf{X'}_{l})$ then $\mathbf{X_{1}}=\mathbf{X}'$, otherwise try another $\mathbf{X}'$. Accept or decline.[m
[32m+[m[32m6. Repeat until obtain $N_{eq}$ accepted sample, the changes stabilizes (we reach a stationary distribution) this phase is called **burn in**.[m
[32m+[m[32m7. From $\mathbf{X}_{N_{\text{eq}}}$ generate $M$ samples until reach the sample $\mathbf{X}_{N_{\text{eq}}+M+1}$.[m
[32m+[m[32mIn each sample generates $E_{L}(\mathbf{R}_{k})$ then average to obtain $\mathbb{E}(E_{L})$ and with it and the derivative from equation number [number] we are able to begin the back propagation step.[m
 ## Deep Learning Fundamentals[m
 This subsection introduces the core concepts of  Deep Learning that are going to be applied in this work.[m
 ### Multi Layer Perceptron[m
[36m@@ -303,7 +332,6 @@[m [m$$[m
 ### Attention Mechanism[m
 [m
 The first attention mechanism were introduced by using the follow:[m
[31m-[m
 ### Self Attention and Multi Head Attention[m
 [m
 From the embedding space $d$ we can generate sub spaces where the heads are going to live and its dimension is $d_{h}$.[m
[36m@@ -330,7 +358,7 @@[m [mAttention mechanism appear with @bahdanau2014neural but it didn't work so:[m
 - [[Attention mechanism]][m
 - [[Self attention mechanism on one head]][m
 - [[Multi-head attention]][m
[31m-# Psi Former[m
[32m+[m[32m# Psi Former Model[m
 [m
 This sections describes the architectures of the model that we are going to use.[m
 First introduce Fermine which generates the hidden states using merely MLP whereas PsiFormer Generate the Hidden states using the Multi Head Attention (MHA) and a MLP.[m
[36m@@ -437,7 +465,6 @@[m [mThe three options to implement this kind of matter are JAX, Tensor Flow and pyto[m
 ## Environment[m
 [m
 For this project we are going to be using Pytorch due his user-friendly and support. Python. with UV[m
[31m-[m
 ## Training[m
 [m
 Due the high computational power needed we are going to using GPUS and of course CUDA.[m
