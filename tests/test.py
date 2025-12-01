import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch


x = np.linspace(0,1,100)
y = np.linspace(0,1,100)
X, Y = np.meshgrid(x,y)


lap_analytic = (4*X**2 + 4*Y**2 - 4) * np.exp(-(X**2 + Y**2))


def prove(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.exp(-torch.pow(x, 2)-torch.pow(y, 2))


x_t = torch.linspace(0, 1, 100)
y_t = torch.linspace(0, 1, 100)
X_t, Y_t = torch.meshgrid(x_t, y_t)

Z = prove(X_t, Y_t)

# Compute the divergence of this guy.


Z.backward()

# Detach the gradients.
X_t = X_t.detach().numpy()
Y_t = Y_t.detach().numpy()
Z = Z.detach().numpy()


plt.figure()
# plt.subplot(1,2,2)
plt.imshow(Z)
# plt.imshow(lap_analytic)
plt.title("Analytic Laplacian")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()