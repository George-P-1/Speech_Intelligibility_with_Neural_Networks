# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import SpeechIntelligibilityDataset
from model import MLP
import time
from datetime import datetime
import wandb

# Constants and Parameters -----------------------------------
WANDB_PROJECT_NAME = "speech-intelligibility-prediction"
WANDB_GROUP_NAME = "mlp-dmatrix-correctness"

DATASET_FILE_PATH = r"preprocessed_datasets\npz_d_matrices_correctness_audiograms\d_matrices_correctness_audiograms_Train_2025-02-05_22-07-04.npz"
DATASET_PART = "Train"
BATCH_SIZE = 32
EPOCHS = 40
LEARNING_RATE = 0.0005
DROPOUT = 0.2

MODEL_ARCHITECTURE = "MLP (input(124650)->4096->2048->1024->512->512->256->1)"
CRITERION = "MSELoss"   # Other options: nn.L1Loss(), nn.HuberLoss()
OPTIMIZER = "Adam"      # Other options: optim.AdamW()

# -----------------------------------------------------------

# WandB configuration
CONFIG = dict(
    dataset=DATASET_PART, 
    batch_size=BATCH_SIZE, 
    epochs=EPOCHS, 
    learning_rate=LEARNING_RATE, 
    model_architecture=MODEL_ARCHITECTURE,
    criterion=CRITERION,
    optimizer=OPTIMIZER,
    dropout=DROPOUT)


# Functions ---------------------------------------------------
def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Format: YYYY-MM-DD_HH-MM-SS

# Get the timestamp for the current run
timestamp = get_timestamp()

# SECTION - Main code here
def main() -> None:
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize WandB
    wandb.init(project=WANDB_PROJECT_NAME, group=WANDB_GROUP_NAME, config=CONFIG, name=f"run_{timestamp}")

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
    input_size = dataset.d_matrices.shape[1]  # Flattened d-matrix size
    model = MLP(input_size).to(device)

    # Log model architecture to WandB
    wandb.watch(model)

    # NOTE - Define loss function and optimizer
    criterion = nn.MSELoss()  # Regression task
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    start_time = time.time()

    for epoch in range(EPOCHS): # Iterate over dataset multiple times
        model.train() # Set model to training mode
        total_loss = 0 # Total loss for the epoch

        for batch_idx, (inputs, targets) in enumerate(train_loader): # Iterate over batches of size BATCH_SIZE and go through the entire dataset
            inputs, targets = inputs.to(device), targets.to(device).unsqueeze(1) 
            # unsqueeze ensures targets have shape (batch_size, 1) which is required for MLP model

            optimizer.zero_grad()               # Reset gradients
            outputs = model(inputs)             # Forward pass
            loss = criterion(outputs, targets)  # Compute loss
            loss.backward()                     # Backpropagation
            # grad_norm = torch.sqrt(sum(p.grad.norm()**2 for p in model.parameters() if p.grad is not None)) # Compute gradient norms for wandb logging
            optimizer.step()                    # Update model weights
            total_loss += loss.item()           # Accumulate loss
            # # Log batch loss & gradient norm to WandB
            # wandb.log({
            #     "batch": batch_idx + 1,
            #     "batch_loss": loss.item(),
            #     "gradient_norm": grad_norm.item() if torch.isfinite(grad_norm) else 0.0  # Handle NaN cases
            # })

        # Validation loop
        model.eval() # Set model to evaluation mode
        val_loss = 0 # Validation loss
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device).unsqueeze(1)
                outputs = model(inputs)             # Forward pass
                loss = criterion(outputs, targets)  # Compute validation loss
                val_loss += loss.item()             # Accumulate validation loss

        # Compute average training and validation loss over all batches
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        # Log metrics to WandB
        wandb.log({
            "epoch": epoch + 1, 
            "epoch_train_loss": avg_train_loss,     # average training loss over all batches
            "epoch_val_loss": avg_val_loss})        # average validation loss over all batches

        print(f"Epoch [{epoch+1}], Epoch Train Loss: {avg_train_loss}, Epoch Val Loss: {avg_val_loss}")

    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")

    # Save trained model
    model_save_path = f"saved_models/speech_intelligibility_mlp_{timestamp}.pth"
    torch.save(model.state_dict(), model_save_path)
    print("Model saved successfully.")

    # Log final model checkpoint
    wandb.log({"model_path": model_save_path})

    # Finish WandB run
    wandb.finish()

if __name__ == '__main__':
    main()
