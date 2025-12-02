from psiformer import MH
import torch
import matplotlib.pyplot as plt


def target(x: torch.Tensor) -> torch.Tensor:
    """
    A sample from a known distribution. First one dimension
    """
    a = (- 0.5 * (x**2).sum())
    return a


def main():
    mh = MH(target, 30, 4000, 1)
    samples = mh.sampler()
    data = samples.squeeze().numpy()
    print(data.shape)
    return data


def ploter(data):
    plt.figure(figsize=(8, 4))
    plt.hist(data, bins=30, color="#4c72b0", edgecolor="white", alpha=0.85) # Here is the guy that I was looking for, what does bins means? the __ I guess.
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Histogram example")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    ploter(main())
