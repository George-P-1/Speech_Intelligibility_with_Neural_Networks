import torch.nn as nn
from torchsummary import summary

# NOTE - MLP model
class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Regression output - Single value output
        )

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    # Instantiate the model
    model = MLP(277)
    # Print the model summary
    summary(model.cuda(), (277,))
