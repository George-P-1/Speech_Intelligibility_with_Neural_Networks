import torch
from torch.utils.data import Dataset
import numpy as np

class SpeechIntelligibilityDataset(Dataset):
    def __init__(self, file_path):
        # Load the preprocessed data
        data = np.load(file_path)
        self.d_matrices = torch.tensor(data["d_matrices"].reshape(len(data["d_matrices"]), -1), dtype=torch.float32)
        self.correctness = torch.tensor(data["correctness"], dtype=torch.float32)
        self.correctness = self.correctness / 100.0  # Normalize correctness values to [0, 1]

    def __len__(self):
        return len(self.correctness)

    def __getitem__(self, idx):
        d_matrix = torch.tensor(self.d_matrices[idx], dtype=torch.float32).clone().detach()
        correctness = torch.tensor(self.correctness[idx], dtype=torch.float32).clone().detach()
        return d_matrix, correctness

