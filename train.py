import torch
from torch import nn
from torch.utils.data import DataLoader
from CPC1_data_loader import CPC1
import torchaudio

from cnn import CNNNetwork

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001

# Specify paths directly here
ANNOTATIONS_FILE = "C:/Users/Codeexia/FinalSemester/CPC1 Data/clarity_CPC1_data.test.v1/clarity_CPC1_data/metadata/CPC1.test.json"
SPIN_FOLDER = "C:/Users/Codeexia/FinalSemester/CPC1 Data/clarity_CPC1_data.test.v1/clarity_CPC1_data/clarity_data/HA_outputs/test"
SCENES_FOLDER = "C:/Users/Codeexia/FinalSemester/CPC1 Data/clarity_CPC1_data.test.v1/clarity_CPC1_data/clarity_data/scenes"

SAMPLE_RATE = 16000
NUM_SAMPLES = 2421


def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader


def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
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
        # for name, param in cnn_model.named_parameters():
        #     if param.grad is not None:
        #         print(f"{name} - Gradient Mean: {param.grad.mean()}")

        optimiser.step()

    print(f"Loss: {loss.item()}")
    



def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_single_epoch(model, data_loader, loss_fn, optimiser, device)
        print("---------------------------")
    print("Finished training")


if __name__ == "__main__":
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

    train_dataloader = create_data_loader(dataset, BATCH_SIZE)

    # Initialize the model
    cnn_model = CNNNetwork().to(device)
    print(cnn_model)

    # Initialize loss function and optimizer
    loss_fn = nn.MSELoss()
    optimiser = torch.optim.Adam(cnn_model.parameters(), lr=LEARNING_RATE)

    # Train the model
    train(cnn_model, train_dataloader, loss_fn, optimiser, device, EPOCHS)

    # Save the model
    torch.save(cnn_model.state_dict(), "feedforwardnet.pth")
    print("Trained feedforward net saved at feedforwardnet.pth")
