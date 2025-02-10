import torch
import torch.nn as nn
# from torchsummary import summary
from torchinfo import summary

# NOTE - GRU model
class GRU_Model(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, bidirectional=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # GRU Layer
        self.gru = nn.GRU(input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=self.bidirectional)

        # Fully Connected Layer
        if self.bidirectional:
            self.fc = nn.Linear(hidden_size * 2, 1)  # *2 because bidirectional GRU
        else:
            self.fc = nn.Linear(hidden_size, 1)

        self.sigmoid = nn.Sigmoid()  # Ensures output is between 0 and 1

    def forward(self, x):
        # x shape: (batch_size, sequence_length=277, feature_dim=15)
        packed_output, hidden = self.gru(x)  # Process sequence
        
        # If bidirectional, combine forward & backward hidden states
        if self.gru.bidirectional:
            final_hidden_state = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            final_hidden_state = hidden[-1]

        output = self.fc(final_hidden_state)  # Fully connected layer
        output = self.sigmoid(output)  # Sigmoid activation
        # output = torch.clamp(output, min=0, max=1)
        return output

if __name__ == "__main__":
    # Input Size
    # Input Settings
    batch_size = 1  # Used only for model summary
    sequence_length = 277
    feature_dim = 15  # Each time step has 15 features
    # input_size = (277, 15)  # 277 * 15
    # Instantiate the model
    model = GRU_Model(input_size=feature_dim)
    model.eval() # Set model to evaluation mode
    # Print the model summary
    summary(model, input_size=(batch_size, sequence_length, feature_dim), mode="eval", device="cuda", 
            col_names=["input_size", "output_size","num_params","params_percent","kernel_size"], 
            col_width=16,
            verbose=1)
