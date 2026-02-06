import torch.nn as nn
import torch

def function_to_learn(x: torch.Tensor) -> torch.Tensor:
    return x**2

class Model(nn.Module):
    def __init__(self, din, dmid, dout):
        super(Model, self).__init__()
        self.linear = nn.Linear(din, dmid)
        self.linear_out = nn.Linear(dmid, dout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = torch.nn.functional.sigmoid(x)
        x = self.linear_out(x)
        return x

# Generate training data
x_data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).reshape(-1, 1)
y_data = function_to_learn(x_data)

# Initialize model
model = Model(din=1, dmid=10, dout=1)

# Initialize optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(5000):
    # Forward pass
    y_pred = model(x_data)
    loss = torch.nn.functional.mse_loss(y_pred, y_data)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Test prediction
test_x = torch.tensor([2.5, 3.5]).reshape(-1, 1)
test_y = model(test_x)
print(f"Predictions for x={test_x.flatten()}: {test_y.flatten()}")
print(f"True values: {function_to_learn(test_x).flatten()}")