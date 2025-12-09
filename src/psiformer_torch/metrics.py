import argparse
import csv
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = PROJECT_ROOT / "assets" / "train_wandb" / "helium.csv"


def _detect_metric_columns(fieldnames: List[str],
                           requested: str | None = None
                           ) -> Tuple[str, str | None, str | None]:
    """
    Infer metric column names. The csv headers look like:
      Step, <metric>, <metric>__MIN, <metric>__MAX
    Returns the base metric key and optional min/max keys.
    """
    normalized = {name.lower(): name for name in fieldnames}
    step_key = normalized.get("step")
    metric_candidates: List[str] = []

    for name in fieldnames:
        if name == step_key:
            continue
        if "__" in name:
            base = name.split("__")[0]
            metric_candidates.append(base)
        else:
            metric_candidates.append(name)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for n in metric_candidates:
        if n not in seen:
            unique.append(n)
            seen.add(n)

    if requested:
        for n in unique:
            if n.lower() == requested.lower():
                base_metric = n
                break
        else:
            raise ValueError(
                f"Metric '{requested}' not found."
                )
    else:
        base_metric = unique[0]

    min_key = f"{base_metric}__MIN"
    max_key = f"{base_metric}__MAX"
    return base_metric, min_key, max_key


def read_energy_csv(csv_path: Path,
                    metric: str | None = None
                    ) -> Tuple[List[int], List[float],
                               List[float], List[float], str]:
    """
    Read the training csv exported from wandb.

    Returns (steps, energy, energy_min, energy_max, metric_name).
    """
    steps: List[int] = []
    energy: List[float] = []
    energy_min: List[float] = []
    energy_max: List[float] = []

    with csv_path.open() as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row.")

        base_metric, min_key, max_key = _detect_metric_columns(
            reader.fieldnames, requested=metric
        )
        step_key = "Step"

        for row in reader:
            steps.append(int(row[step_key]))
            energy.append(float(row[base_metric]))
            if min_key:
                energy_min.append(float(row[min_key]))
            if max_key:
                energy_max.append(float(row[max_key]))

    # Fallback if min/max are absent
    if not energy_min:
        energy_min = list(energy)
    if not energy_max:
        energy_max = list(energy)

    return steps, energy, energy_min, energy_max, base_metric


def _safe_name(csv_path: Path) -> str:
    # Title for the plot from file name: helium.csv -> Helium
    name = csv_path.stem.replace("_", " ").strip()
    return name.capitalize()


def plot_energy(
    csv_path: Path,
    output_path: Path | None = None,
    show_range: bool = True,
    metric: str | None = None,
    target: float | None = None,
) -> Path:
    """
    Create a plot from a wandb-exported csv.

    Saves a png next to the csv (unless output_path is provided).
    """
    steps, energy, energy_min, energy_max, metric_name = read_energy_csv(
        csv_path, metric=metric)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, energy, label=metric_name, color="tab:blue", linewidth=2)

    if show_range and not _all_equal(energy, energy_min, energy_max):
        ax.fill_between(
            steps,
            energy_min,
            energy_max,
            color="tab:blue",
            alpha=0.15,
            label=f"{metric_name} range",
        )

    ax.set_title(f"{_safe_name(csv_path)} {metric_name} over steps")
    ax.set_xlabel("Step")
    ax.set_ylabel(metric_name)
    if target is not None:
        ax.axhline(target, color="tab:red",
                   linestyle="--", linewidth=1.4, label=f"Target {target}")

    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend()
    fig.tight_layout()

    if output_path is None:
        output_path = csv_path.with_suffix(".png")
    else:
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def _all_equal(*sequences: Iterable[float]) -> bool:
    """
    Return True if every sequence has identical values in the same order.
    """
    first = None
    for seq in sequences:
        if first is None:
            first = list(seq)
        else:
            if list(seq) != first:
                return False
    return True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot wandb-exported training csv (energy vs steps)."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"Path to csv file (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output png path (default: alongside csv)",
    )
    parser.add_argument(
        "--no-range",
        action="store_true",
        help="Disable min/max shading band.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default=None,
        help="Metric column to plot (defaults to the first found).",
    )
    parser.add_argument(
        "target",
        nargs="?",
        type=float,
        default=None,
        help="Optional target/reference value to draw as a horizontal line.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    png_path = plot_energy(
        args.csv,
        args.out,
        show_range=not args.no_range,
        metric=args.metric,
        target=args.target,
    )
    print(f"Saved plot to {png_path}")
