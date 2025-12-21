#!/usr/bin/env python3
"""Plot fast vs. large convergence curves for atomic numbers 4–7."""

from __future__ import annotations

from pathlib import Path
import csv
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def read_energy_curve(csv_path: Path) -> Tuple[List[float], List[float]]:
    """Return step and energy arrays from a W&B style CSV file."""
    with csv_path.open("r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames:
            raise ValueError(f"No header detected in {csv_path}")

        energy_column = next(
            (name for name in reader.fieldnames if "Energy" in name and "__" not in name),
            None,
        )
        if energy_column is None:
            raise ValueError(f"Could not find an energy column in {csv_path}")

        steps: List[float] = []
        energies: List[float] = []
        for row in reader:
            steps.append(float(row["Step"]))
            energies.append(float(row[energy_column]))

    return steps, energies


def align_curve(target_steps: np.ndarray, source_steps: np.ndarray, source_values: np.ndarray) -> np.ndarray:
    """Interpolate source values onto the target step grid, clamping endpoints."""
    if source_steps.size == 0:
        return np.full_like(target_steps, np.nan, dtype=float)
    # np.interp extrapolates with end values automatically.
    return np.interp(target_steps, source_steps, source_values)


def percent_error(values: np.ndarray, reference_energy: float) -> np.ndarray:
    """Compute percent error vs. a provided reference energy."""
    denom = abs(reference_energy) if abs(reference_energy) > 1e-12 else 1.0
    errors = (values - reference_energy) / denom * 100.0
    # Avoid zeros for log scale
    errors = np.maximum(errors, 1e-8)
    return errors


def plot_element(
    ax: plt.Axes,
    csv_root: Path,
    fast_name: str,
    large_name: str,
    title: str,
    reference_energy: float,
) -> None:
    fast_path = csv_root / fast_name
    large_path = csv_root / large_name

    if not fast_path.exists():
        ax.text(
            0.5,
            0.5,
            f"Missing fast ({fast_path.name})",
            ha="center",
            va="center",
            fontsize=9,
            transform=ax.transAxes,
        )
        return

    fast_steps, fast_energy = read_energy_curve(fast_path)
    fast_steps_arr = np.array(fast_steps, dtype=float)
    fast_energy_arr = np.array(fast_energy, dtype=float)

    large_steps_arr: np.ndarray
    large_energy_arr: np.ndarray
    if large_path.exists():
        large_steps, large_energy = read_energy_curve(large_path)
        large_steps_arr = np.array(large_steps, dtype=float)
        large_energy_arr = np.array(large_energy, dtype=float)
    else:
        large_steps_arr = np.array([], dtype=float)
        large_energy_arr = np.array([], dtype=float)
        ax.text(
            0.5,
            0.45,
            f"Missing large ({large_path.name})",
            ha="center",
            va="center",
            fontsize=9,
            transform=ax.transAxes,
        )

    if large_steps_arr.size > 0:
        large_energy_aligned = align_curve(fast_steps_arr, large_steps_arr, large_energy_arr)
    else:
        large_energy_aligned = np.full_like(fast_energy_arr, np.nan)

    # Restrict to the shared iteration span to avoid extrapolated tails.
    if large_steps_arr.size > 0:
        overlap_max = min(fast_steps_arr.max(), large_steps_arr.max())
        mask = fast_steps_arr <= overlap_max
    else:
        mask = np.ones_like(fast_steps_arr, dtype=bool)

    fast_steps_arr = fast_steps_arr[mask]
    fast_energy_arr = fast_energy_arr[mask]
    large_energy_aligned = large_energy_aligned[mask]

    fast_err = percent_error(fast_energy_arr, reference_energy)
    large_err = percent_error(large_energy_aligned, reference_energy) if large_steps_arr.size > 0 else None

    ax.plot(fast_steps_arr, fast_err, label="fast", linewidth=2.0, color="#1f77b4")
    if large_err is not None:
        ax.plot(fast_steps_arr, large_err, label="large", linewidth=2.0, color="#ff7f0e")

    ax.set_title(title, fontsize=12, fontweight="bold")

    # Reference dashed lines
    for level, color in [(100.0, "#555555"), (10.0, "#555555"), (1.0, "#555555")]:
        ax.axhline(level, color=color, linestyle="--", linewidth=0.8, alpha=0.8)
    ax.axhline(0.1, color="#d62728", linestyle="--", linewidth=1.2, alpha=0.9)


def render_grid(
    csv_root: Path,
    plot_root: Path,
    xscale: str,
    yscale: str,
    ylim: tuple[float, float] | None,
    output_name: str,
    title: str,
) -> None:
    elements = [
        {"fast": "4_fast.csv", "large": "4_large.csv", "title": "Be", "ref": -14.667},
        {"fast": "5_fast.csv", "large": "5_large.csv", "title": "B", "ref": -24.653},
        {"fast": "6_fast.csv", "large": "6_large.csv", "title": "C", "ref": -37.845},
        {"fast": "7_fast.csv", "large": "7_large.csv", "title": "N", "ref": -54.589},
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharex=False, sharey=True)
    for idx, (ax, element) in enumerate(zip(axes.flat, elements)):
        plot_element(ax, csv_root, element["fast"], element["large"], element["title"], element["ref"])
        ax.set_yscale(yscale)
        if ylim is not None:
            ax.set_ylim(*ylim)
        elif yscale == "linear":
            ax.set_ylim(bottom=0.0)
        ax.set_xscale(xscale)
        ax.grid(alpha=0.3, linestyle="--", linewidth=0.5, which="both")
        if idx // 2 == 1:
            ax.set_xlabel("Iterations")
        else:
            ax.tick_params(labelbottom=False)
        if idx % 2 == 0:
            ax.set_ylabel("Correlation energy error (%)")

    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.legend(
        handles=[
            plt.Line2D([], [], color="#1f77b4", linewidth=2.0, label="fast"),
            plt.Line2D([], [], color="#ff7f0e", linewidth=2.0, label="large"),
            plt.Line2D([], [], color="#d62728", linewidth=1.2, linestyle="--", label="Chemical Accuracy (0.1%)"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        frameon=False,
    )

    output_path = plot_root / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved plot grid to {output_path}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    csv_root = project_root / "assets" / "train_wandb_csv"
    plot_root = project_root / "assets" / "plots"

    # Linear iterations view (original)
    render_grid(
        csv_root,
        plot_root,
        xscale="linear",
        yscale="log",
        ylim=(1.0, 1e2),
        output_name="convergence_fast_vs_large.pdf",
        title="Fast vs. Large correlation energy error",
    )

    # Exponential (log) iterations view
    render_grid(
        csv_root,
        plot_root,
        xscale="log",
        yscale="log",
        ylim=(1.0, 1e2),
        output_name="convergence_fast_vs_large_log_iter.pdf",
        title="Fast vs. Large correlation energy error (log iterations)",
    )

    # Linear iterations and linear error view
    render_grid(
        csv_root,
        plot_root,
        xscale="linear",
        yscale="linear",
        ylim=(0.0, 60.0),
        output_name="convergence_fast_vs_large_linear_axes.pdf",
        title="Fast vs. Large correlation energy error (linear axes)",
    )


if __name__ == "__main__":
    main()
