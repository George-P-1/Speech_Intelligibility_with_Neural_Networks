from torch import nn
from torchsummary import summary

class CNNNetwork(nn.Module):
    
    def __init__(self):
        super().__init__()
        # 4 conv blocks / flatten / linear / sigmoid for output
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 24, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(24, 48, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(48, 128, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.flatten = nn.Flatten()  # Flatten the output of conv layers
        self.linear = nn.Linear(128 * 5 * 6 * 3, 1)  # Linear layer for final prediction
        self.sigmoid = nn.Sigmoid()  # Sigmoid for output in range [0, 1]

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        logits = self.linear(x)
        predictions = self.sigmoid(logits)  # Apply Sigmoid to logits for regression
        return predictions

if __name__ == "__main__":
    cnn = CNNNetwork()
    summary(cnn.cuda(), (1, 64, 5))  # Check model summary to confirm layer shapes
