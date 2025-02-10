import torch
import torch.nn as nn
# from torchsummary import summary
from torchinfo import summary
import torch.nn.utils.rnn as rnn_utils

# NOTE - GRU model
class GRU_Model(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, bidirectional=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # GRU Layer
        self.gru = nn.GRU(input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=self.bidirectional) 
        # Input: (batch_size, sequence_length=277, feature_dim=15)
        # Output: (batch_size, sequence_length=277, hidden_size*2 if bidirectional else hidden_size)

        # Fully Connected Layer
        if self.bidirectional:
            self.fc = nn.Linear(hidden_size * 2, 1)  # *2 because bidirectional GRU     # Input: (batch_size, hidden_size=128 * 2) -> Output: (batch_size, 1)
        else:
            self.fc = nn.Linear(hidden_size, 1)                                         # Input: (batch_size, hidden_size=128) -> Output: (batch_size, 1)

        self.sigmoid = nn.Sigmoid()  # Ensures output is between 0 and 1

    def forward(self, x, lengths):
        """
        x: (batch_size, max_sequence_length, feature_dim)
        lengths: (batch_size,) - Actual lengths of each sequence (without padding)
        """

        print("Input Shape before packing:", x.shape)
        print("Sequence Lengths before packing:", lengths)

        # Sort sequences by length (required for pack_padded_sequence)
        sorted_lengths, sorted_idx = torch.sort(lengths, descending=True)
        x = x[sorted_idx]

        # Pack padded sequences (removes padding internally)
        packed_x = rnn_utils.pack_padded_sequence(x, sorted_lengths.cpu(), batch_first=True, enforce_sorted=True)
        
        print("Packed output shape before unpacking:", packed_output.data.shape)
        print("Lengths before unpacking:", lengths)


        # Forward pass through GRU
        packed_output, hidden = self.gru(packed_x)  # Process sequence
        # packed_output shape: (batch_size, sequence_length=277, hidden_size=128 * 2 if bidirectional else hidden_size)
        # hidden shape: (num_layers * num_directions, batch_size, hidden_size=128)
        # num_directions = 2 if bidirectional else 1

        # Unpack sequence correctly
        # output, _ = rnn_utils.pad_packed_sequence(packed_output, batch_first=True, total_length=x.shape[1])
        output, _ = rnn_utils.pad_packed_sequence(packed_output, batch_first=True)

        # Restore original order
        _, original_idx = torch.sort(sorted_idx)
        output = output[original_idx]

        # If bidirectional, combine forward & backward hidden states
        if self.gru.bidirectional:
            final_hidden_state = torch.cat((hidden[-2], hidden[-1]), dim=1)
            # final_hidden_state shape: (batch_size, hidden_size=128 * 2)
        else:
            final_hidden_state = hidden[-1]
            # final_hidden_state shape: (batch_size, hidden_size=128)

        output = self.fc(final_hidden_state)  # Fully connected layer   # (batch_size, 128) -> (batch_size, 1)
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
    # Create a fake lengths tensor (batch_size, ) filled with max sequence length
    lengths = torch.tensor([sequence_length] * batch_size)
    summary(model, input_data=[torch.randn(batch_size, sequence_length, feature_dim), lengths], mode="eval", device="cuda", 
            col_names=["input_size", "output_size","num_params","params_percent","kernel_size"], 
            col_width=16,
            verbose=1)
