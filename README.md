# Solving the many-electron Schrödinger equation with transformers

> Acknolwedge to 

Implementation of the paper A self attention mechanism for quantum chemistry

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

This repository demonstrates an experimental approach to approximating solutions to the many-electron Schrödinger equation using transformer architectures.
It contains reference code and utilities to build and run small-scale
experiments on CPU (no CUDA required).

## What this does

- Uses transformer-style attention to model electronic wavefunction structure.
- Provides utilities and example scripts to run toy problems and evaluate
	model performance against simple quantum chemistry references.

This project is intended for research and learning. It is not production-ready
for large-scale quantum chemistry simulations but can be used to prototype
ideas and compare modeling choices.

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

# install CPU-only pytorch (from official PyTorch CPU index)
uv install torch --index-url https://download.pytorch.org/whl/cpu

# install other lightweight deps (edit as needed)
uv install -r requirements.txt || pip install -r requirements.txt
```

If you prefer plain pip for CPU-only PyTorch:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r requirements.txt
```

Note: The code in this repo sets PyTorch's default device to CPU; no CUDA
setup is required.

## Usage

1. Prepare data / basis encodings for the target system.
2. Edit `config.py` to set model size, number of heads, block (sequence)
	 length, and training hyperparameters.
3. Run the training/experiment script (example):

```bash
python src/main.py --config configs/toy_system.yaml
```

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
3. Train the transformer to predict minimal-energy coefficients or
	 approximate the wavefunction amplitude for sampled configurations.
4. Evaluate energy and compare against reference (Hartree–Fock, small CI).

## Tips and caveats

- Enforcing antisymmetry exactly (Slater determinants) is nontrivial when
	using sequence models; consider hybrid approaches (learn corrections to a
	Slater determinant) or include antisymmetry in the loss.
- Transformers scale quadratically with sequence length; start with small
	basis sizes and toy molecules.
- Use physics-informed losses wherever possible to improve sample efficiency.

## Reproducibility

- Pin dependencies in `requirements.txt` if you need strict reproducibility.
- Seed RNGs at the start of experiments (`torch.manual_seed`, `random.seed`).

## References and further reading

- Vaswani et al., "Attention Is All You Need" (transformers)
- Recent literature on machine learning for quantum chemistry and wavefunction
	modeling (e.g., Neural quantum states, FermiNet, PauliNet).

## License

This project is provided under the repository license.