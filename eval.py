import torch
import torch.nn as nn
import numpy as np
import wandb
from dataset import SpeechIntelligibilityDataset
from model import GRU_Model
from torch.utils.data import DataLoader
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import kendalltau, pearsonr
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# -----------------------------------------------------------
# Set this to True if you want to update the same WandB run
WANDDB_UPDATE = False # TODO - Doesn't work yet
# -----------------------------------------------------------
# -----------------------------------------------------------

# Set paths
MODEL_PATH = r"C:\Users\George\Desktop\Automatic Control and Robotics\Semester 7\Thesis\Neural Networks\Workspace\saved_models\mlp-dmatrix-correctness\speech_intelligibility_mlp_2025-02-06_05-38-08_wandbd_i5pfp37j.pth"
# Set this to the correct test dataset path
DATASET_PATH = r"C:\Users\George\Desktop\Automatic Control and Robotics\Semester 7\Thesis\Neural Networks\Workspace\preprocessed_datasets\npz_d_matrices_correctness_audiograms\d_matrices_correctness_audiograms_Test_2025-02-05_22-46-44.npz"

# Set this to the correct training run ID
if WANDDB_UPDATE:
    WANDB_RUN_ID = "i5pfp37j"
    # Reattach to the same WandB run
    wandb.init(project="speech_intelligibility", id=WANDB_RUN_ID, resume="allow")

# Load dataset
dataset = SpeechIntelligibilityDataset(DATASET_PATH)
test_loader = DataLoader(dataset, batch_size=16, shuffle=False)

# Load trained model
# Load trained model
sequence_length = 277  # Time steps
feature_dim = dataset.d_matrices.shape[1]  # Should be 15
device = "cuda" if torch.cuda.is_available() else "cpu"
model = GRU_Model(input_size=feature_dim).to(device)
model.load_state_dict(torch.load(MODEL_PATH))

# Evaluate model
all_preds, all_targets = [], []

model.eval()    # Set model to evaluation mode
with torch.no_grad():
    for inputs, masks, targets in test_loader:
        inputs, masks, targets = inputs.to(device), masks.to(device), targets.to(device)

        outputs = model(inputs).squeeze()
        all_preds.extend(outputs.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

# Compute Metrics
rmse = root_mean_squared_error(all_targets, all_preds)
mae = mean_absolute_error(all_targets, all_preds)
r2 = r2_score(all_targets, all_preds)
kendall_tau, _ = kendalltau(all_targets, all_preds)
pearson_corr, _ = pearsonr(all_targets, all_preds)  # Same as Correlation Coefficient (CC)

print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R² Score: {r2:.4f}, Kendall's Tau: {kendall_tau:.4f}, Pearson Correlation (or CC): {pearson_corr:.4f}")

if WANDDB_UPDATE:
    # Log metrics to the same WandB run
    wandb.summary['evaluation_rmse'] = rmse
    wandb.summary['evaluation_mae'] = mae
    wandb.summary['evaluation_r2_score'] = r2
    wandb.summary['evaluation_kendall_tau'] = kendall_tau
    wandb.summary['evaluation_pearson_corr'] = pearson_corr
    wandb.finish()

# Plot predictions vs. targets
plt.figure(figsize=(10, 6))
plt.scatter(all_targets, all_preds, alpha=0.5)
plt.xlabel("True Correctness")
plt.ylabel("Predicted Correctness")
plt.title("Predictions vs. Targets")
plt.grid(True)
plt.show()
