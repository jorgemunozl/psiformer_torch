import sys
import csv
import math
import matplotlib.pyplot as plt

def load_series(path):
    steps, energies = [], []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if not row or row[0].strip().startswith("#"):
                continue
            try:
                step = float(row[0])
                energy = float(row[1])
            except (ValueError, IndexError):
                continue  # skip headers/bad rows
            steps.append(step)
            energies.append(energy)
    return steps, energies

def main():
    if len(sys.argv) < 3:
        print("Usage: python compare_energy_grid.py model1.csv model2.csv [out.png]")
        sys.exit(1)

    steps1, energy1 = load_series(sys.argv[1])
    steps2, energy2 = load_series(sys.argv[2])
    n = min(len(steps1), len(steps2))
    if n == 0:
        print("No overlapping data to plot.")
        sys.exit(1)

    steps1, energy1 = steps1[:n], energy1[:n]
    steps2, energy2 = steps2[:n], energy2[:n]

    out_path = sys.argv[3] if len(sys.argv) > 3 else "energy_grid.png"

    fig, axes = plt.subplots(3, 2, figsize=(10, 8), sharex=False, sharey=False)
    axes = axes.flatten()
    chunk = max(1, math.ceil(n / 6))

    for i in range(6):
        start = i * chunk
        end = min(n, start + chunk)
        ax = axes[i]
        ax.plot(steps1[start:end], energy1[start:end],
                label="Model 1", color="tab:blue", linewidth=1)
        ax.plot(steps2[start:end], energy2[start:end],
                label="Model 2", color="tab:orange", linewidth=1)
        ax.set_title(f"Segment {i+1}")
        ax.set_xlabel("Step")
        ax.set_ylabel("Energy")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
