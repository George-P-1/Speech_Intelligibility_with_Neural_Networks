import torch
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F  # For pooling operations

class SpeechIntelligibilityDataset(Dataset):
    def __init__(self, file_path):
        # Load the preprocessed data
        data = np.load(file_path)
        
        # Convert d_matrices to a PyTorch tensor and reshape
        d_matrices = torch.tensor(data["d_matrices"], dtype=torch.float32)  # Shape: (batch, 277, 15)
        masks = torch.tensor(data["masks"], dtype=torch.float32)  # Shape: same as d_matrices

        # Flatten `d_matrices` to match new input size
        self.d_matrices = d_matrices.view(len(self.d_matrices), -1)  # Shape: (batch, 277*15 = 4155)
        self.masks = masks.view(len(self.masks), -1)  # Shape: (batch, 4155)

        self.correctness = torch.tensor(data["correctness"], dtype=torch.float32)
        self.correctness = self.correctness / 100.0  # Normalize correctness values to [0, 1]

    def __len__(self):
        return len(self.correctness)

    def __getitem__(self, idx):
        d_matrix = self.d_matrices[idx].clone().detach()
        mask = self.masks[idx].clone().detach()
        correctness = self.correctness[idx].clone().detach()
        return d_matrix, mask, correctness

