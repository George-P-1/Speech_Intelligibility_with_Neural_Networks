import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from dataset import SpeechIntelligibilityDataset
from model import GRU_Model
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import kendalltau, pearsonr
import wandb

# Function to evaluate model after training
def evaluate_model(model, test_dataset_path, R_VEC, batch_size=128):
    print("\nRunning evaluation on test dataset...")

    # Load dataset
    dataset = SpeechIntelligibilityDataset(test_dataset_path)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)  # Use larger batch size for efficiency

    # Set model to evaluation mode
    model.eval()

    all_preds, all_targets = [], []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()    # Set model to evaluation mode
    with torch.no_grad():
        for inputs, masks, targets, targets_vec in test_loader:
            inputs, masks, targets, targets_vec = inputs.to(device), masks.to(device), targets.to(device), targets_vec.to(device)

            # outputs = model(inputs).squeeze()
            # Forward pass
            outputs = model(inputs)                     # Get 10-dimensional softmax output
            final_outputs = F.softmax(outputs, dim=1)  # Softmax activation without nn module
            final_outputs = torch.sum(final_outputs * R_VEC, dim=1)  # Compute final intelligibility score

            all_preds.extend(final_outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # Compute Metrics
    rmse = root_mean_squared_error(all_targets, all_preds)
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)
    kendall_tau, _ = kendalltau(all_targets, all_preds)
    pearson_corr, _ = pearsonr(all_targets, all_preds)  # Same as Correlation Coefficient (CC)

    print(f"Evaluation Results - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R² Score: {r2:.4f}, Kendall's Tau: {kendall_tau:.4f}, Pearson Correlation (or CC): {pearson_corr:.4f}")

    # Log metrics to WandB (same training run)
    wandb.log({
        "predictions_vs_targets": wandb.plot.scatter(
            wandb.Table(data=list(zip(all_targets, all_preds)), columns=["Ground Truth", "Predictions"]),
            "Ground Truth",
            "Predictions",
            title="Predictions vs Targets"
        )
    })
    # Log histogram of predicted scores
    wandb.log({"pred_histogram": wandb.plot.histogram(
        wandb.Table(data=[[p] for p in all_preds], columns=["Predicted Scores"]),
        "Predicted Scores", title="Distribution of Predicted Intelligibility"
    )})
    # Log histogram of true scores
    wandb.log({"true_histogram": wandb.plot.histogram(
        wandb.Table(data=[[t] for t in all_targets], columns=["True Scores"]),
        "True Scores", title="Distribution of True Intelligibility"
    )})

    wandb.summary['evaluation_rmse'] = rmse
    wandb.summary['evaluation_mae'] = mae
    wandb.summary['evaluation_r2_score'] = r2
    wandb.summary['evaluation_kendall_tau'] = kendall_tau
    wandb.summary['evaluation_pearson_corr'] = pearson_corr