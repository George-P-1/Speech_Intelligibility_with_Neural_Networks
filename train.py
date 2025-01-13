import torch
from torch import nn
from torch.utils.data import DataLoader
from CNNNNNNN.CPC1_data_loader import CPC1
import torchaudio
import wandb  # Import WandB
import numpy as np
from CNNNNNNN.cnn import CNNNetwork
import random


BATCH_SIZE = 200
EPOCHS = 10
LEARNING_RATE = 0.001

# Specify paths directly here
ANNOTATIONS_FILE = "C:/Users/Codeexia/FinalSemester/CPC1 Data/clarity_CPC1_data.test.v1/clarity_CPC1_data/metadata/CPC1.test.json"
SPIN_FOLDER = "C:/Users/Codeexia/FinalSemester/CPC1 Data/clarity_CPC1_data.test.v1/clarity_CPC1_data/clarity_data/HA_outputs/test"
SCENES_FOLDER = "C:/Users/Codeexia/FinalSemester/CPC1 Data/clarity_CPC1_data.test.v1/clarity_CPC1_data/clarity_data/scenes"

SAMPLE_RATE = 16000
NUM_SAMPLES = SAMPLE_RATE * 6

# TODO: Add random seed for reproducibility after finishing the CNN

seed = 42
torch.manual_seed(seed)  # For PyTorch operations
np.random.seed(seed)  # For NumPy operations
random.seed(seed)  
def create_data_loader(train_data, batch_size, shuffle=True):
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
    return train_dataloader


def train_single_epoch(model, data_loader, loss_fn, optimiser, device, epoch):
    total_loss = 0
    for batch in data_loader:
        input = batch['spin'].to(device).float()  # Input to the model
        print(f"Spin name {batch['spin']}")
        target = batch['correctness'].to(device).float() / 100.0  # Normalize target
        target = target.unsqueeze(1)

        # Calculate loss
        prediction = model(input).float()  # Prediction should be between 0 and 1
        loss = loss_fn(prediction, target)  # MSE loss for regression

        # Backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        total_loss += loss.item()

    # Log loss to WandB after every epoch
    wandb.log({"epoch": epoch, "train_loss": total_loss / len(data_loader)})
    print(f"Loss: {total_loss / len(data_loader)}")


def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_single_epoch(model, data_loader, loss_fn, optimiser, device, i+1)
        print("---------------------------")
    print("Finished training")


if __name__ == "__main__":
    # Initialize WandB run
    wandb.init(project="speech-clarity-prediction", entity="codeexia0")
    
    # TODO: Add different scenarios for training by giving different values to the hyperparameters
    # Log hyperparameters
    wandb.config.batch_size = BATCH_SIZE
    wandb.config.epochs = EPOCHS
    wandb.config.learning_rate = LEARNING_RATE
    wandb.config.sample_rate = NUM_SAMPLES

    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using Device: {device}")

    # Create mel spectrogram transform
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64,  # Number of mel filterbanks
    )

    # Initialize dataset
    dataset = CPC1(
        ANNOTATIONS_FILE,
        SPIN_FOLDER,
        SCENES_FOLDER,
        mel_spectrogram,
        SAMPLE_RATE,
        NUM_SAMPLES,
        device,
        max_length=263
    )

    train_dataloader = create_data_loader(dataset, BATCH_SIZE, shuffle=True)

    # Initialize the model
    cnn_model = CNNNetwork().to(device)
    print(cnn_model)

    # Initialize loss function and optimizer
    loss_fn = nn.MSELoss()
    optimiser = torch.optim.Adam(cnn_model.parameters(), lr=LEARNING_RATE)

    # Log the model architecture to WandB
    wandb.watch(cnn_model, log="all", log_freq=100)

    # Train the model
    train(cnn_model, train_dataloader, loss_fn, optimiser, device, EPOCHS)

    # Save the model
    torch.save(cnn_model.state_dict(), "feedforwardnet_fl.pth")
    print("Trained feedforward net saved at feedforwardnet.pth")

    # End the WandB run
    wandb.finish()
