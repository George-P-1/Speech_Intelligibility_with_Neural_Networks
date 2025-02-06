import torch
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F  # For pooling operations

class SpeechIntelligibilityDataset(Dataset):
    def __init__(self, file_path, pool_size=(8, 16)):   # Pooling size for (15,30) → (8,16)
        # Load the preprocessed data
        data = np.load(file_path)
        
        # Convert d_matrices to a PyTorch tensor and reshape to (batch, 277, 15, 30)
        d_matrices = torch.tensor(data["d_matrices"], dtype=torch.float32)  # Shape: (N, 277, 15, 30)

        # Apply 2D average pooling (Adaptive)
        d_matrices_pooled = F.adaptive_avg_pool2d(d_matrices, pool_size)  # Shape: (N, 277, 8, 16)

        # Flatten after pooling
        self.d_matrices = d_matrices_pooled.view(len(d_matrices_pooled), -1)  # Shape: (N, 277*8*16)

        self.correctness = torch.tensor(data["correctness"], dtype=torch.float32)
        self.correctness = self.correctness / 100.0  # Normalize correctness values to [0, 1]

    def __len__(self):
        return len(self.correctness)

    def __getitem__(self, idx):
        d_matrix = self.d_matrices[idx].clone().detach()
        correctness = self.correctness[idx].clone().detach()
        return d_matrix, correctness

