import torch
import torch.nn as nn
# from torchsummary import summary
from torchinfo import summary

# NOTE - GRU model
class CNN1d(nn.Module):
    def __init__(self, input_size, kernel_size):
        super().__init__()

        # CNN Layers
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=20, kernel_size=kernel_size, stride=1)  # Input: (batch_size, 15, 277) -> Output: (batch_size, 20, 277)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=20, out_channels=1, kernel_size=kernel_size, stride=1)  # Input: (batch_size, 20, 277) -> Output: (batch_size, 25, 277)
        self.relu2 = nn.ReLU()

        # self.sigmoid = nn.Sigmoid()

        
    def forward(self, x, masks):
        # x shape: (batch_size, frames=277, feature_dim=15)

        # Transpose
        x = x.permute(0, 2, 1)  # (batch_size, 15, 277)

        # CNN Layers
        x = self.conv1(x)  # (batch_size, 20, 277)
        x = self.relu1(x)

        x = self.conv2(x)  # (batch_size, 25, 277)
        x = self.relu2(x)

        # Remove the last channel
        x = x.squeeze(1)  # Shape: (batch_size, 277)

        # Apply mask
        x = x * masks  # Element-wise multiplication    

        # average over all time frames
        x = torch.mean(x, dim=1)  # Shape: (batch_size,)

        # Sigmoid activation
        # x = self.sigmoid(x)

        return x

if __name__ == "__main__":
    # Input Size
    # Input Settings
    frames = 277  # Number of frames in the input
    octave_bands = 15  # Number of octave bands in the input
    kernel_size = 5  # Kernel size for the convolutional layers
    # Instantiate the model
    model = CNN1d(octave_bands, kernel_size)
    model.eval() # Set model to evaluation mode
    # Print the model summary
    summary(model, input_size=(1, frames, octave_bands), mode="eval", device="cuda", 
            col_names=["input_size", "output_size","num_params","params_percent","kernel_size"], 
            col_width=16,
            verbose=1)
