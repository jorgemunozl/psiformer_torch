from psiformer_torch.psiformer import Psiformer
from test_config import PsiformerConfig




model = Psiformer()  # Initialize your model with appropriate config
output = model(input_data)  # Forward pass with input_data

print(output)
