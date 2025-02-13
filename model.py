import torch
import torch.nn as nn
# from torchsummary import summary
from torchinfo import summary

# NOTE - GRU model
class CNN1d(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        # CNN Layers
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=20, kernel_size=1, stride=1)  # Input: (batch_size, 15, 277) -> Output: (batch_size, 32, 277)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=20, out_channels=1, kernel_size=1, stride=1)  # Input: (batch_size, 32, 277) -> Output: (batch_size, 64, 277)
        self.relu2 = nn.ReLU()
        
    def forward(self, x):
        # x shape: (batch_size, frames=277, feature_dim=15)

        # Transpose
        x = x.permute(0, 2, 1)  # (batch_size, 15, 277)

        # CNN Layers
        x = self.conv1(x)  # (batch_size, 20, 277)
        x = self.relu1(x)

        x = self.conv2(x)  # (batch_size, 1, 277)
        x = self.relu2(x)

        # Take the mean over the time dimension
        x = torch.mean(x, dim=2)  # (batch_size, 1)

        return x

if __name__ == "__main__":
    # Input Size
    # Input Settings
    frames = 277  # Number of frames in the input
    octave_bands = 15  # Number of octave bands in the input
    # Instantiate the model
    model = CNN1d(input_size=octave_bands)
    model.eval() # Set model to evaluation mode
    # Print the model summary
    summary(model, input_size=(1, frames, octave_bands), mode="eval", device="cuda", 
            col_names=["input_size", "output_size","num_params","params_percent","kernel_size"], 
            col_width=16,
            verbose=1)
