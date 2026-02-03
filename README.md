<div align="center">
  <h1>Solving the Schrödinger equation with Transformers</h1>
  <p>
    <a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ready-EE4C2C?logo=pytorch&logoColor=white"></a>
    <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white"></a>
    <a href="https://wandb.ai/"><img alt="Weights & Biases" src="https://img.shields.io/badge/Weights_&_Biases-Track_runs-FFBE00?logo=weightsandbiases&logoColor=white"></a>
    <a href="https://deepmind.google/"><img alt="DeepMind" src="https://img.shields.io/badge/DeepMind-inspired-4285F4?logo=google&logoColor=white"></a>
    <a href="#"><img alt="Tests" src="https://img.shields.io/badge/CI-tests_pending-lightgrey?logo=githubactions&logoColor=white"></a>
    <a href="#"><img alt="PRs Welcome" src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg"></a>
    <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-repo-default-lightgrey.svg"></a>
  </p>
</div>

This repository demonstrates an experimental approach to approximating solutions to the many-electron Schrödinger equation using transformer architectures. It contains reference code and utilities to build and run scale experiments, for more details on how this repository works and was created look [PsiFormer](src/PsiFormer_Torch%20Documentation.md) for the full documentation.  Watch the [paper](assets/research/main.pdf) for a detailed theoretical framework result,the [beamer](assets/research/beamer.pdf) for a more friendly explanation or the [poster](assets/research/poster.pdf) for a resume of this work.

> My sincerest thanks to Angel A. Flores and Alejandro D. Paredes for their feedback and helpful technical discussion. I would also like to thank the Faculty of Computer Science at UNI for the GPU used to make this project.

<div style="text-align:center;"><img src="assets/research/poster.png" width="400"></div>

## Table of contents

- [What this does](#what-this-does)
- [Key ideas (short)](#key-ideas-short)
- [Quick install (CPU-only)](#quick-install-cpu-only)
- [Usage](#usage)
- [Files and layout](#files-and-layout)
- [Example workflow](#example-workflow)
- [Tips and caveats](#tips-and-caveats)
- [Reproducibility](#reproducibility)
- [References and further reading](#references-and-further-reading)
- [License](#license)


## What this does

- Uses transformer-style attention to model electronic wavefunction structure.
- Provides utilities and example scripts to run  problems and evaluate model performance against simple quantum chemistry references.

This project is intended for research .It is not production-ready for large-scale quantum chemistry simulations but can be used to prototype ideas and compare modeling choices.

## Key ideas (short)

- Represent electronic configurations or basis expansions as sequences and
	learn interactions using attention layers.
- Use permutation-equivariant input encodings or ordered basis sequences to
	capture antisymmetry constraints through learned modules and loss terms.
- Train with physics-informed losses (energy expectation, cusp conditions,
	or density overlaps) and supervised or self-supervised pretraining.

## Quick install (CPU-only)

If you use `uv` (fast installer), you can install a CPU-only PyTorch wheel and
other dependencies like this:

```bash
# install uv if you don't have it
pip install -U uv

# Sync dependencies
uv sync
```

Then, install PyTorch CPU-only version:
```bash
uv pip install --index-url https://download.pytorch.org/whl/cpu torch
```

## Usage

1. Prepare data / basis encodings for the target system.
2. Edit `config.py` to set model size, number of heads, block (sequence)
	 length, and training hyperparameters.
3. Run the training/experiment script (example):

Replace the example command above with your own runner or flags as needed.

## Files and layout

- `src/` — main source code
	- `main.py` — training / experiment entrypoint
	- `utils.py` — helper utilities, device selection, etc.
	- `config.py` — config and hyperparameters
- `pyproject.toml` — Python project metadata
- `template.tex` — report template (optional)

## Example workflow

1. Choose a small system (2–10 electrons) and a compact basis set.
2. Build sequence encodings for orbitals/electron coordinates.
3. Train the transformer to predict minimal-energy coefficients or approximate the wavefunction amplitude for sampled configurations.
4. Evaluate energy and compare against reference (Hartree–Fock, small CI).

## Tips and caveats

- Enforcing antisymmetry exactly (Slater determinants) is nontrivial when
	using sequence models; consider hybrid approaches (learn corrections to a
	Slater determinant) or include antisymmetry in the loss.
- Transformers scale quadratically with sequence length; start with small
	basis sizes and toy molecules.
- Use physics-informed losses wherever possible to improve sample efficiency.


## References and further reading

- Vaswani et al., "Attention Is All You Need" (transformers).
- A Self Attention Ansatz for Quantum Chemistry
- Ab-Initio Solution of the Many-Electron Schrodinger Equation With Deep Neural Networks
- QMC Torch Molecular Wave functions with Neural Components for energy and force calculations.