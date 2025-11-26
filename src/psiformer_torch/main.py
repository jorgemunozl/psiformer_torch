import torch


class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.layer(x)    

def main():
    model = NeuralNetwork()
    sample_input = torch.randn(5, 10)
    output = model(sample_input)
    print(output)


if __name__ == "__main__":
    main()