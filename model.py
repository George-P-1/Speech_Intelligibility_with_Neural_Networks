import torch.nn as nn
# from torchsummary import summary
from torchinfo import summary

# NOTE - MLP model
class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 256),
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            # nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 1),      # Regression output - Single value output

            nn.Sigmoid()            # Ensures output is between 0 and 1
            # maybe use torch.clamp() instead of sigmoid # torch.clamp(self.model(x), 0, 1)  # Keeps outputs in range [0,1]
        )

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    # Input Size
    input_size = 4115  # 277 * 15
    # Instantiate the model
    model = MLP(input_size)
    model.eval() # Set model to evaluation mode
    # Print the model summary
    summary(model, input_size=(1, input_size,), mode="eval", device="cuda", 
            col_names=["input_size", "output_size","num_params","params_percent","kernel_size"], 
            col_width=16,
            verbose=1)
