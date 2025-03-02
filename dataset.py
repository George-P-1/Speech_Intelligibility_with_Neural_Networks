import torch
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F  # For pooling operations

class SpeechIntelligibilityDataset(Dataset):
    def __init__(self, file_path):
        # Load the preprocessed data
        data = np.load(file_path)
        
        # Convert d_matrices to a PyTorch tensor (retain 3D shape for GRU)
        d_matrices = torch.tensor(data["d_matrices"], dtype=torch.float32)  # Shape: (batch, 277, 15)

        # Apply normalization and log transform
        d_matrices = d_matrices/30.0            # maybe do this in preprocessing
        d_matrices = np.log1p(d_matrices)       # maybe do this in preprocessing

       # Masks (same shape as d_matrices)
        masks = torch.tensor(data["masks"], dtype=torch.float32)  # Shape: (batch, 277, 15)

        # Store tensors (KEEP AS 3D, DO NOT FLATTEN)
        self.d_matrices = d_matrices  # Shape: (batch, 277, 15)
        self.masks = masks  # Shape: (batch, 277, 15)

        # Correctness
        self.correctness = torch.tensor(data["correctness"], dtype=torch.float32)
        self.correctness = self.correctness / 100.0  # Normalize correctness values to [0, 1]

    def __len__(self):
        return len(self.correctness)

    def __getitem__(self, idx):
        d_matrix = self.d_matrices[idx].clone().detach()
        mask = self.masks[idx].clone().detach()
        correctness = self.correctness[idx].clone().detach()
        return d_matrix, mask, correctness


if __name__ == "__main__":
    # Test the dataset
    dataset = SpeechIntelligibilityDataset(r"preprocessed_datasets\npz_d_matrices_2d_masks_correctness\d_matrices_2d_masks_correctness_audiograms_Train_2025-02-08_18-28-50.npz")
    print(f"Dataset Length: {len(dataset)}")
    print(f"Dataset d_matrices shape: {dataset.d_matrices.shape}")
    print(f"Dataset masks shape: {dataset.masks.shape}")
    print(f"Dataset correctness shape: {dataset.correctness.shape}")
    idx = 1
    d_matrix, mask, correctness = dataset[idx]
    print(f"Sample d_matrix shape: {d_matrix.shape}")
    print(f"Sample mask shape: {mask.shape}")
    print(f"Sample correctness: {correctness}\n")

    # Print stuff from getitem
    print(f"Sample d_matrix: {dataset.__getitem__([idx])[0]}")
    print(f"Sample mask: {dataset.__getitem__([idx])[1]}")
    print(f"Sample correctness: {dataset.__getitem__([idx])[2]}")
