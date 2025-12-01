import matplotlib.pyplot as plt
import numpy as np

# Sample data
data = np.random.normal(loc=0.0, scale=1.0, size=100000) # mu and sigma I guess

plt.figure(figsize=(8, 4))
plt.hist(data, bins=30, color="#4c72b0", edgecolor="white", alpha=0.85) # Here is the guy that I was looking for, what does bins means? the __ I guess.
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Histogram example")
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()
