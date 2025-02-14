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

        # Store tensors
        self.d_matrices = d_matrices  # Shape: (batch, 277, 15)
        self.masks = masks  # Shape: (batch, 277, 15)

        # Correctness
        self.correctness = torch.tensor(data["correctness"], dtype=torch.float32)
        # Normalize correctness values to [0, 1]
        self.correctness = self.correctness / 100.0 

        # Convert correctness from scalar value to 10-bin vector where each bin represents a correctness range
        '''
        # for example  if SI >= 0.00 and SI < 0.11, then the correctness vector is [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]   place=1     idx =0
        #           or if SI >= 0.11 and SI < 0.22, then the correctness vector is [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]   place=2     idx =1
        #           or if SI >= 0.22 and SI < 0.33, then the correctness vector is [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]   place=3     idx =2
        #           or if SI >= 0.33 and SI < 0.44, then the correctness vector is [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]   place=4     idx =3
        #           or if SI >= 0.44 and SI < 0.55, then the correctness vector is [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]   place=5     idx =4
        #           or if SI >= 0.55 and SI < 0.66, then the correctness vector is [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]   place=6     idx =5
        #           or if SI >= 0.66 and SI < 0.77, then the correctness vector is [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]   place=7     idx =6
        #           or if SI >= 0.77 and SI < 0.88, then the correctness vector is [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]   place=8     idx =7
        #           or if SI >= 0.88 and SI < 0.99, then the correctness vector is [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]   place=9     idx =8
        #           or if SI >= 0.99 and SI <= 1.00, then the correctness vector is [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  place=10    idx =9
        '''
        correctness_bins = torch.linspace(0, 1, steps=10)  # 10 bins for correctness values
        # Correctness bins: tensor([0.0000, 0.1111, 0.2222, 0.3333, 0.4444, 0.5556, 0.6667, 0.7778, 0.8889, 1.0000])
        
        # For each scalar correctness value, find the bin it belongs to
        self.correctness_vec = torch.bucketize(self.correctness, correctness_bins, right=True)
        # Correctness after bucketize:  tensor(10)

        # Subtract 1 to make it 0-indexed
        self.correctness_vec -= 1
        
        # Make the bin numbers into one-hot vectors
        # self.correctness = torch.clamp(self.correctness_vec, min=0, max=9)  # Ensure values are within [0, 10]. Not necessary anymore because i used right=True in bucketize
        self.correctness_vec = F.one_hot(self.correctness_vec.to(torch.int64), num_classes=10).to(torch.float32)   

        # Shape of d_matrices:  torch.Size([4863, 277, 15])
        # Shape of masks:  torch.Size([4863, 277, 15])
        # Shape of correctness_vec:  torch.Size([4863, 10])

    def __len__(self):
        return len(self.correctness)

    def __getitem__(self, idx):
        d_matrix = self.d_matrices[idx].clone().detach()
        mask = self.masks[idx].clone().detach()
        correctness = self.correctness[idx].clone().detach()
        correctness_vec = self.correctness_vec[idx].clone().detach()
        return d_matrix, mask, correctness, correctness_vec


if __name__ == "__main__":
    # Test the dataset
    dataset = SpeechIntelligibilityDataset(r"preprocessed_datasets\npz_d_matrices_2d_masks_correctness\d_matrices_2d_masks_correctness_audiograms_Train_2025-02-08_18-28-50.npz")
    print(f"Dataset Length: {len(dataset)}")
    print(f"Dataset d_matrices shape: {dataset.d_matrices.shape}")
    print(f"Dataset masks shape: {dataset.masks.shape}")
    print(f"Dataset correctness shape: {dataset.correctness.shape}")
    idx = 1
    d_matrix, mask, correctness, correctness_vec = dataset[idx]
    print(f"Sample d_matrix shape: {d_matrix.shape}")
    print(f"Sample mask shape: {mask.shape}")
    print(f"Sample correctness: {correctness}\n")

    # Print stuff from getitem
    print(f"Sample d_matrix: {dataset.__getitem__([idx])[0]}")
    print(f"Sample mask: {dataset.__getitem__([idx])[1]}")
    print(f"Sample correctness: {dataset.__getitem__([idx])[2]}")
    print(f"Sample correctness_vec: {dataset.__getitem__([idx])[3]}")
