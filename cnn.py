from torch import nn
from torchsummary import summary
import torch.nn.functional as F

class CNNNetwork(nn.Module):
    
    # def __init__(self):
    #     super().__init__()
    #     self.conv1 = nn.Conv2d(1,16,2,1) # 1 input channel, 16 output channels, 3x3 kernel size, 1 stride
    #     self.conv2 = nn.Conv2d(16,24,2,1)
    #     # Fully Connected Layer
    #     self.fc1 = nn.Linear(23400, 120)
    #     self.fc2 = nn.Linear(120, 84)
    #     self.fc3 = nn.Linear(84, 1)

    # def forward(self, X):
    #     X = F.relu(self.conv1(X))
    #     X = F.max_pool2d(X,2,2) # 2x2 kernal and stride 2
    #     # Second Pass
    #     X = F.relu(self.conv2(X))
    #     X = F.max_pool2d(X,2,2) # 2x2 kernal and stride 2

    #     # Re-View to flatten it out
    #     X = X.view(X.size(0), -1)  # Dynamically flatten


    #     # Fully Connected Layers
    #     X = F.relu(self.fc1(X))
    #     X = F.relu(self.fc2(X))
    #     X = self.fc3(X)
    #     return F.log_softmax(X, dim=1)
    
    def __init__(self):
        super().__init__()
        # 4 conv blocks / flatten / linear / sigmoid for output
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=2), # 1 input channel, 16 output channels, 3x3 kernel size, 1 stride
            nn.ReLU(), # Activation function for non-linearity 
            nn.MaxPool2d(kernel_size=2) # downsampling the image representation by 2x2 
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
        
            # Fully Connected Layer
        self.fc1 = nn.Linear(5*5*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
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
    summary(cnn.cuda(), (2, 64, 263))  # Check model summary to confirm layer shapes and parameters
