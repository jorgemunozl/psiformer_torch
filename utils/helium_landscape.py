#!/usr/bin/env python3
"""Generate |psi|^2 and local-energy
landscapes for a 2e Helium checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from psiformer_torch.config import Model_Config
from psiformer_torch.hamiltonian import Hamiltonian
from psiformer_torch.psiformer import PsiFormer, get_device


def softplus_inverse(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Inverse softplus with a small numerical guard."""
    return torch.log(torch.expm1(x) + eps)


def maybe_disable_layernorm(model: PsiFormer, state_dict: dict) -> None:
    """If the checkpoint lacks layernorm weights, replace LN with identity."""
    has_ln = any(k.startswith("layers.0.ln_1") for k in state_dict)
    if has_ln:
        return
    for layer in model.layers:
        layer.ln_1 = torch.nn.Identity()
        layer.ln_2 = torch.nn.Identity()


def build_config(
    state_dict: dict,
    n_spin_up: int | None,
    n_spin_down: int | None,
    n_determinants: int | None,
    n_head: int | None,
    nuclear_charge: int,
) -> Model_Config:
    """
    Infer a model config from checkpoint tensor shapes, with optional overrides.
    """
    n_embd, in_features = state_dict["l_0.weight"].shape
    n_features = in_features - 1  # forward appends radius
    n_layer = max(int(k.split(".")[1]) for k in state_dict if k.startswith("layers.")) + 1

    out_up = state_dict["orbital_head.orb_up.weight"].shape[0]
    out_down = state_dict["orbital_head.orb_down.weight"].shape[0]

    spin_up = n_spin_up if n_spin_up is not None else 1
    spin_down = n_spin_down if n_spin_down is not None else 1

    det_from_up = out_up // max(spin_up, 1)
    det_from_down = out_down // max(spin_down, 1)
    det_guess = min(det_from_up, det_from_down)
    n_det = n_determinants if n_determinants is not None else det_guess

    head = n_head if n_head is not None else 32

    return Model_Config(
        n_layer=n_layer,
        n_head=head,
        n_embd=n_embd,
        n_features=n_features,
        n_determinants=n_det,
        n_electron_num=spin_up + spin_down,
        n_spin_up=spin_up,
        n_spin_down=spin_down,
        nuclear_charge=nuclear_charge,
    )


def load_model(
    checkpoint_path: Path,
    device: torch.device,
    n_spin_up: int | None,
    n_spin_down: int | None,
    n_determinants: int | None,
    n_head: int | None,
    nuclear_charge: int,
) -> PsiFormer:
    """Instantiate the model to match the checkpoint and load weights."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )

    config = build_config(
        state_dict,
        n_spin_up=n_spin_up,
        n_spin_down=n_spin_down,
        n_determinants=n_determinants,
        n_head=n_head,
        nuclear_charge=nuclear_charge,
    )
    model = PsiFormer(config).to(device)

    maybe_disable_layernorm(model, state_dict)

    # Backwards compat for envelope parameter names and missing det_logits.
    for spin in ("up", "down"):
        sigma_key = f"orbital_head.envelope_{spin}.sigma"
        raw_key = f"orbital_head.envelope_{spin}.raw_sigma"
        if raw_key not in state_dict and sigma_key in state_dict:
            state_dict[raw_key] = softplus_inverse(state_dict[sigma_key])
    if "orbital_head.det_logits" not in state_dict:
        state_dict["orbital_head.det_logits"] = torch.zeros(
            config.n_determinants, device=device
        )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print("load_state_dict warnings:")
        if missing:
            print("  missing:", missing)
        if unexpected:
            print("  unexpected:", unexpected)
    model.eval()
    return model


def get_logabs(model: PsiFormer, R: torch.Tensor, output_kind: str) -> torch.Tensor:
    """
    Return log|psi| for a batch of positions.
    output_kind: "logpsi" when the model already returns log|psi|,
                 "psi" when it returns psi directly.
    """
    out = model(R)
    if isinstance(out, tuple) and len(out) == 2:
        logabs, _ = out
        return logabs
    if output_kind == "psi":
        psi = out
        return torch.log(torch.clamp(torch.abs(psi), min=1e-12))
    return out


def make_grid(xmin: float, xmax: float, n: int, device: torch.device) -> tuple[torch.Tensor, np.ndarray]:
    xs = np.linspace(xmin, xmax, n, dtype=np.float32)
    x1_grid, x2_grid = np.meshgrid(xs, xs, indexing="xy")

    zeros = np.zeros_like(x1_grid, dtype=np.float32)
    r1 = np.stack([x1_grid.ravel(), zeros.ravel(), zeros.ravel()], axis=1)
    r2 = np.stack([x2_grid.ravel(), zeros.ravel(), zeros.ravel()], axis=1)
    R = np.stack([r1, r2], axis=1)  # (B, 2, 3)
    R_tensor = torch.from_numpy(R).to(device)
    return R_tensor, xs


def compute_density(model: PsiFormer, R: torch.Tensor, batch_size: int, output_kind: str) -> torch.Tensor:
    density_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for chunk in R.split(batch_size):
            logabs = get_logabs(model, chunk, output_kind)
            density_chunks.append(torch.exp(2.0 * logabs).cpu())
    return torch.cat(density_chunks, dim=0)


def compute_local_energy(
    model: PsiFormer,
    R: torch.Tensor,
    batch_size: int,
    output_kind: str,
    nuclear_charge: int = 2,
) -> torch.Tensor:
    logpsi_fn = lambda coords: get_logabs(model, coords, output_kind)
    hamil = Hamiltonian(logpsi_fn, n_elec=R.size(1), Z=nuclear_charge)

    energy_chunks: list[torch.Tensor] = []
    for chunk in R.split(batch_size):
        energy_chunks.append(hamil.local_energy(chunk).detach().cpu())
    return torch.cat(energy_chunks, dim=0)


def plot_landscape(
    xs: np.ndarray,
    density_grid: np.ndarray,
    energy_grid: np.ndarray,
    x2_slices: tuple[float, float],
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=(11, 5))
    gs = fig.add_gridspec(
        2, 2, height_ratios=[1, 5], width_ratios=[1, 1], hspace=0.05, wspace=0.2
    )

    ax_slice = fig.add_subplot(gs[0, 0])
    ax_density = fig.add_subplot(gs[1, 0], sharex=ax_slice)
    ax_energy = fig.add_subplot(gs[:, 1])

    im_density = ax_density.imshow(
        density_grid,
        extent=[xs[0], xs[-1], xs[0], xs[-1]],
        origin="lower",
        aspect="auto",
        cmap="viridis",
    )
    ax_density.set_xlabel("x1 (Bohr)")
    ax_density.set_ylabel("x2 (Bohr)")
    ax_density.set_title(r"$|\psi(r_1, r_2)|^2$")
    fig.colorbar(im_density, ax=ax_density, fraction=0.046, pad=0.02, label="Density")

    colors = ("#d62728", "#1f77b4")
    for x2_val, color in zip(x2_slices, colors):
        idx = int(np.argmin(np.abs(xs - x2_val)))
        ax_slice.plot(xs, density_grid[idx, :], label=f"x2 = {xs[idx]:.2f}", color=color)
        ax_density.axhline(xs[idx], linestyle="--", linewidth=1.0, color=color, alpha=0.8)

    ax_slice.set_ylabel(r"$|\psi|^2$ slice")
    ax_slice.legend(frameon=False, fontsize=9)
    plt.setp(ax_slice.get_xticklabels(), visible=False)

    im_energy = ax_energy.imshow(
        energy_grid,
        extent=[xs[0], xs[-1], xs[0], xs[-1]],
        origin="lower",
        aspect="auto",
        cmap="coolwarm",
    )
    ax_energy.set_xlabel("x1 (Bohr)")
    ax_energy.set_ylabel("x2 (Bohr)")
    ax_energy.set_title(r"Local energy $E_L(r_1, r_2)$ (Ha)")
    fig.colorbar(im_energy, ax=ax_energy, fraction=0.046, pad=0.02, label="Hartree")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    print(f"Saved figure to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Helium wavefunction and local energy landscapes.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/checkpoints/CUDA_BATCHED_HELIUM.pth"),
        help="Path to the trained Helium checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("helium_landscape.png"),
        help="Output image path.",
    )
    parser.add_argument(
        "--grid-min", type=float, default=-2.0, help="Lower bound for x1/x2 grid (Bohr)."
    )
    parser.add_argument(
        "--grid-max", type=float, default=2.0, help="Upper bound for x1/x2 grid (Bohr)."
    )
    parser.add_argument("--points", type=int, default=200, help="Number of grid points along each axis.")
    parser.add_argument("--batch", type=int, default=2048, help="Chunk size for forward/local-energy eval.")
    parser.add_argument("--n-spin-up", type=int, default=1, help="Number of spin-up electrons (default helium).")
    parser.add_argument("--n-spin-down", type=int, default=1, help="Number of spin-down electrons (default helium).")
    parser.add_argument("--n-determinants", type=int, default=None, help="Override determinant count if desired.")
    parser.add_argument("--n-head", type=int, default=None, help="Attention heads (defaults to 32).")
    parser.add_argument("--nuclear-charge", type=int, default=2, help="Atomic number Z (default helium).")
    parser.add_argument(
        "--output-kind",
        choices=["logpsi", "psi"],
        default="logpsi",
        help="Set to 'psi' if the model forward returns psi instead of log|psi|.",
    )
    parser.add_argument(
        "--x2-slices",
        type=float,
        nargs=2,
        default=(-1.0, 1.0),
        metavar=("X2A", "X2B"),
        help="Two x2 values to draw density slices at.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint}")

    model = load_model(
        args.checkpoint,
        device,
        n_spin_up=args.n_spin_up,
        n_spin_down=args.n_spin_down,
        n_determinants=args.n_determinants,
        n_head=args.n_head,
        nuclear_charge=args.nuclear_charge,
    )

    R, xs = make_grid(args.grid_min, args.grid_max, args.points, device)

    density_flat = compute_density(model, R, args.batch, args.output_kind)
    density_grid = density_flat.view(args.points, args.points).numpy()

    energy_flat = compute_local_energy(
        model, R, args.batch, args.output_kind, nuclear_charge=args.nuclear_charge
    )
    energy_grid = energy_flat.view(args.points, args.points).numpy()

    plot_landscape(xs, density_grid, energy_grid, tuple(args.x2_slices), args.output)


if __name__ == "__main__":
    main()
