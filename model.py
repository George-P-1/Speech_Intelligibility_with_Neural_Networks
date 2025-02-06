import torch.nn as nn
from torchsummary import summary

# NOTE - MLP model
class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 1)    # Regression output - Single value output
        )

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    # Instantiate the model
    model = MLP(124650)
    # Print the model summary
    summary(model.cuda(), (124650,))
