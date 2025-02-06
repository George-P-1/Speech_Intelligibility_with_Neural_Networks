import torch.nn as nn
from torchsummary import summary

# NOTE - MLP model
class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, 1),      # Regression output - Single value output
            nn.Sigmoid()            # Ensures output is between 0 and 1
        )

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    # Instantiate the model
    model = MLP(35456)
    # Print the model summary
    summary(model.cuda(), (35456,))
