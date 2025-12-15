#!/usr/bin/env python3
"""Generate a composite figure for the paper that blends CSV curves and PNG renders."""

from __future__ import annotations

from pathlib import Path
import csv
from typing import Iterable, Tuple, List

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def read_energy_curve(csv_path: Path) -> Tuple[List[float], List[float]]:
    """Parse the first energy series from a CSV exported by Weights & Biases."""
    with csv_path.open("r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames:
            raise ValueError(f"No header detected in {csv_path}")

        energy_column = next(
            (
                name
                for name in reader.fieldnames
                if "Energy" in name and "__" not in name
            ),
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


def plot_energy(ax: plt.Axes, steps: Iterable[float], energies: Iterable[float], title: str) -> None:
    ax.plot(steps, energies, linewidth=2.0, color="#1f77b4")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Step")
    ax.set_ylabel("Energy (Ha)")
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)


def plot_image(ax: plt.Axes, image_path: Path, title: str) -> None:
    img = mpimg.imread(image_path)
    ax.imshow(img)
    ax.set_title(title, fontsize=12)
    ax.axis("off")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    assets_root = project_root / "assets"

    csv_sources = [
        (
            assets_root / "train_wandb_csv" / "berilium.csv",
            "Beryllium (Energy vs Step)",
        ),
        (
            assets_root / "train_wandb_csv" / "hidrogen_cuda.csv",
            "Hydrogen CUDA (Energy vs Step)",
        ),
    ]

    image_sources = [
        (assets_root / "plots" / "helium.png", "Helium Render"),
        (assets_root / "plots" / "lithium.png", "Lithium Render"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (csv_path, title) in zip(axes[0], csv_sources):
        steps, energies = read_energy_curve(csv_path)
        plot_energy(ax, steps, energies, title)

    for ax, (img_path, title) in zip(axes[1], image_sources):
        plot_image(ax, img_path, title)

    fig.suptitle("PsiFormer Element Benchmarks", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    output_path = assets_root / "plots" / "paper_composite.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    print(f"Saved combined plot to {output_path}")


if __name__ == "__main__":
    main()
