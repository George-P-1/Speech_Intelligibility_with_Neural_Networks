# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import SpeechIntelligibilityDataset
from model import CNN1d
import time
from datetime import datetime
import wandb
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from torchinfo import summary
import os
from eval_function import evaluate_model

# Constants and Parameters -----------------------------------
WANDB_PROJECT_NAME = "speech-intelligibility-prediction"

WANDB_GROUP_NAME = "cnn-dmatrix-masks-correctness"
PREPROCESSED_DATASET_NAME = "d_matrices_2d_masks_correctness_audiograms"

DATASET_PART = "Train" # -----------------------------------
DATASET_FILE_PATH = r"preprocessed_datasets\npz_d_matrices_2d_masks_correctness\d_matrices_2d_masks_correctness_audiograms_Train_2025-02-08_18-28-50.npz"
TEST_DATASET_PATH = r"preprocessed_datasets\npz_d_matrices_2d_masks_correctness\d_matrices_2d_masks_correctness_audiograms_Test_2025-02-08_18-47-23.npz"
# DATASET_PART = "Train_indep" # -----------------------------------
# DATASET_FILE_PATH = r"preprocessed_datasets\npz_d_matrices_2d_masks_correctness\d_matrices_2d_masks_correctness_audiograms_Train_Independent_2025-02-09_16-26-28.npz"
# TEST_DATASET_PATH = r"preprocessed_datasets\npz_d_matrices_2d_masks_correctness\d_matrices_2d_masks_correctness_audiograms_Test_Independent_2025-02-09_16-23-47.npz"

BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.001
DROPOUT = 'variable' # Options: 'none', 'fixed', 'variable'
KERNEL_SIZE = 3

TAGS = [
    # "adaptive_pooling",
    # "modified-masking-logic",
    DATASET_PART,
    PREPROCESSED_DATASET_NAME,
    "d-matrix-2d",
    # "d-matrix-3d-reduced",
    "divided-dmatrices-by-30",
    # "sigmoid-last-layer",
    # "ReLU-last-layer",
    # "torch-clamp",
    # "no-last-layer-activation",
    # "variable-dropout",
    "normalized-dmatrix-log1p",
    # "batch-normalization",
    ]

frames = 277  # Number of frames in the input
octave_bands = 15  # Number of octave bands in the input

# MODEL_ARCHITECTURE = "MLP (input(4155)->4096->2048->1024->512->256->128->1)"
# DROPOUT_ARCHITECTURE = "(input->0.3->0.3->0.2->0.1->0.0->0.0->output)"
MODEL_ARCHITECTURE = f"CNN (input(15,frames)->CNN(20,frames)->frames->average->1)"
DROPOUT_ARCHITECTURE = "(input->0.0->0.0->0.0->output)"

CRITERION = "MSELoss"   # Other options: nn.L1Loss(), nn.HuberLoss()
OPTIMIZER = "Adam"      # Other options: optim.AdamW()
MASKING_LOGIC = "regular"

# -----------------------------------------------------------

# WandB configuration
CONFIG = dict(
    dataset=DATASET_PART, 
    batch_size=BATCH_SIZE, 
    epochs=EPOCHS, 
    learning_rate=LEARNING_RATE, 
    model_architecture=MODEL_ARCHITECTURE,
    dropout_architecture=DROPOUT_ARCHITECTURE,
    criterion=CRITERION,
    optimizer=OPTIMIZER,
    dropout=DROPOUT,
    masking_logic=MASKING_LOGIC,
    kernel_size=KERNEL_SIZE
    # adaptive_pool_size=ADAPTIVE_POOL_SIZE
    )


# Functions ---------------------------------------------------
def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Format: YYYY-MM-DD_HH-MM-SS

# Get the timestamp for the current run
timestamp = get_timestamp()

# SECTION - Main code here -----------------------------------
def main() -> None:
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize WandB
    wandb.init(project=WANDB_PROJECT_NAME, group=WANDB_GROUP_NAME, tags=TAGS, config=CONFIG, name=f"run_{timestamp}")

    # Load dataset
    dataset = SpeechIntelligibilityDataset(DATASET_FILE_PATH) # Instantiate the dataset

    # Split into training and validation sets
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Instantiate model
    input_size = dataset.d_matrices.shape[2]  # d-matrix octave bands dimension
    print("Input Size:", input_size) # REMOVE_LATER - to see if input size is correct
    model = CNN1d(input_size, KERNEL_SIZE).to(device)

    # Log model architecture to WandB
    wandb.watch(model)

    # NOTE - Define loss function and optimizer
    criterion = nn.MSELoss(reduction='none')  # Default reduction is 'mean'. Using 'none' to compute loss for each sample to apply mask
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    start_time = time.time()

    for epoch in range(EPOCHS): # Iterate over dataset multiple times
        model.train() # Set model to training mode
        total_loss = 0 # Total loss for the epoch
        total_rmse_train_loss = 0 # Total RMSE loss for the epoch

        for batch_idx, (inputs, masks, targets) in enumerate(train_loader): # Iterate over batches of size BATCH_SIZE and go through the entire dataset
            inputs, masks, targets = inputs.to(device), masks.to(device), targets.to(device).unsqueeze(1) 
            # unsqueeze ensures targets have shape (batch_size, 1) which is required for MLP model

            optimizer.zero_grad()                       # Reset gradients
            outputs = model(inputs)                     # Forward pass. Outputs are predictions
            loss = criterion(outputs, targets)          # Compute loss
            rmse_loss = torch.sqrt(loss.mean())   # Compute RMSE loss for the epoch
            
            # SECTION - Apply mask

            # loss = loss.expand(-1, 277)  # Expands loss to (batch_size, 277)
            # # Check values and shapes
            # if batch_idx % 100 == 0:
            #     print("Loss Shape:", loss.shape) # Loss Shape: torch.Size([16, 1])
            #     # print("Loss:", loss)
            #     print("Masks Shape:", masks.shape)  # Masks Shape: torch.Size([16, 277, 15])
            #     print("Masks mean:", masks.mean(dim=2).shape)  # Masks mean: torch.Size([16, 277])
            #     print("Masks sum:", masks.sum())    # Masks sum: tensor(33075., device='cuda:0')    values not valid for all batches and samples
            #     print("Loss *Masks mean sum:", (loss * masks.mean(dim=2)).sum())    # Loss *Masks mean sum: tensor(53.1478, device='cuda:0', grad_fn=<SumBackward0>)

            loss = (loss * masks.mean(dim=2)).sum() / masks.sum()

            #!SECTION - Apply mask

            loss.backward()                             # Backpropagation
            # grad_norm = torch.sqrt(sum(p.grad.norm()**2 for p in model.parameters() if p.grad is not None)) # Compute gradient norms for wandb logging
            optimizer.step()                    # Update model weights
            total_loss += loss.item()           # Accumulate loss
            total_rmse_train_loss += rmse_loss.item() # Accumulate RMSE loss
            # # Log batch loss & gradient norm to WandB
            # wandb.log({
            #     "batch": batch_idx + 1,
            #     "batch_loss": loss.item(),
            #     "gradient_norm": grad_norm.item() if torch.isfinite(grad_norm) else 0.0  # Handle NaN cases
            # })

        # Validation loop
        model.eval() # Set model to evaluation mode
        val_loss = 0 # Validation loss
        total_rmse_val_loss = 0 # Total RMSE loss for the epoch
        with torch.no_grad():
            for inputs, masks, targets in val_loader:
                inputs, masks, targets = inputs.to(device), masks.to(device), targets.to(device).unsqueeze(1)
                outputs = model(inputs)                     # Forward pass
                loss = criterion(outputs, targets)          # Compute validation loss
                rmse_loss = torch.sqrt(loss.mean())   # Compute RMSE loss for the epoch
                
                # SECTION - Apply mask
                loss = (loss * masks.mean(dim=2)).sum() / masks.sum()
                #!SECTION - Apply mask

                val_loss += loss.item()                     # Accumulate validation loss
                total_rmse_val_loss += rmse_loss.item()     # Accumulate RMSE loss

        # Compute average training and validation loss over all batches
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_rmse_train_loss = total_rmse_train_loss / len(train_loader)
        avg_rmse_val_loss = total_rmse_val_loss / len(val_loader)

        # Log metrics to WandB
        wandb.log({
            "epoch": epoch + 1, 
            "epoch_train_loss": avg_train_loss,     # average training loss over all batches
            "epoch_val_loss": avg_val_loss,        # average validation loss over all batches
            "epoch_rmse_train_loss": avg_rmse_train_loss,     # RMSE loss for the epoch
            "epoch_rmse_val_loss": avg_rmse_val_loss})  # RMSE loss for the validation set

        print(f"Epoch [{epoch+1}], Epoch Train Loss: {avg_train_loss:.8f}, Epoch Val Loss: {avg_val_loss:.8f}")

    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")

    # Save trained model
    # Create folder if it doesn't exist
    if not os.path.exists(f"saved_models/{WANDB_GROUP_NAME}"):
        os.makedirs(f"saved_models/{WANDB_GROUP_NAME}")
    model_save_path = f"saved_models/{WANDB_GROUP_NAME}/{WANDB_GROUP_NAME}_{DATASET_PART}_{timestamp}.pth"
    print(f"WandB Run ID: {wandb.run.id}")
    torch.save(model.state_dict(), model_save_path)
    print("Model saved successfully.")
    print(f"Model saved to: {model_save_path}")

    # Log final model checkpoint
    wandb.log({"model_path": model_save_path})

    # Evaluate model on test dataset
    evaluate_model(model, TEST_DATASET_PATH)

    # Print model summary
    summary(model, input_size=(1, frames, octave_bands), mode="eval", device="cuda", 
            col_names=["input_size", "output_size","num_params","params_percent","kernel_size"], 
            col_width=16,
            verbose=1)
    
    # Print model details
    print("Model Architecture:", MODEL_ARCHITECTURE)
    if DROPOUT != 'none':
        print("Dropout Architecture:", DROPOUT_ARCHITECTURE)

    # Finish WandB run
    wandb.finish()

if __name__ == '__main__':
    main()
