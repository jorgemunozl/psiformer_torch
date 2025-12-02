import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Sample data
data = np.random.normal(loc=0.0, scale=1.0, size=10)  # mu and sigma I guess
x = torch.tensor([3.4])

data2 = []

for i in range(100):
    y = torch.randn_like(x)
    data2.append(float(y))

print(data2)
print(data)

plt.figure(figsize=(8, 4))
plt.hist(data2, bins=30, edgecolor="white", alpha=0.85) # Here is the guy that I was looking for, what does bins means? the __ I guess.
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Histogram example")
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()


class Model(nn.Module):
    def __init__(self, n_in: int, n_out:int):
        super().__init__()
        self.fc = nn.Linear(n_in, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


model = Model(2, 3)
#for name, p in model.named_parameters():
    #print(name, p.shape)
