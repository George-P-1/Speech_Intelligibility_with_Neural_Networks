import torch
from torch import nn
from torch.utils.data import DataLoader
from CPC1_data_loader import CPC1
import torchaudio
import wandb
import numpy as np
from cnn import CNNNetwork
import random
import datetime
import os
import time

BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.0001

# Specify paths directly here
ANNOTATIONS_FILE = "C:/Users/Codeexia/FinalSemester/CPC1 Data/clarity_CPC1_data.v1_1/clarity_CPC1_data/metadata/CPC1.train.json"
SPIN_FOLDER = "C:/Users/Codeexia/FinalSemester/CPC1 Data/clarity_CPC1_data.v1_1/clarity_CPC1_data/clarity_data/HA_outputs/train"
SCENES_FOLDER = "C:/Users/Codeexia/FinalSemester/CPC1 Data/clarity_CPC1_data.v1_1/clarity_CPC1_data/clarity_data/scenes"

SAMPLE_RATE = 16000
NUM_SAMPLES = SAMPLE_RATE * 6

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

def create_data_loader(train_data, batch_size, shuffle=True):
    return DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)

def train_single_epoch(model, data_loader, loss_fn, optimiser, device, epoch):
    total_loss = 0
    for batch in data_loader:
        input = batch['spin'].to(device).float()
        mask = batch['mask'].to(device).unsqueeze(1)  # Ensure mask matches prediction dimensions
        target = batch['correctness'].to(device).float().unsqueeze(1)
        prediction = model(input).float()
        loss = (loss_fn(prediction, target) * mask).sum() / mask.sum()  # Apply mask to loss

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        total_loss += loss.item()

    wandb.log({"epoch": epoch, "train_loss": total_loss / len(data_loader)})
    print(f"Epoch {epoch}: Loss: {total_loss / len(data_loader):.4f}")

def train(model, data_loader, loss_fn, optimiser, device, epochs, model_filename):
    start_time = time.time()
    try:
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            train_single_epoch(model, data_loader, loss_fn, optimiser, device, epoch + 1)
            print("---------------------------")
        print("Finished training")
    except KeyboardInterrupt:
        print("\nTraining interrupted! Saving model...")
        torch.save(model.state_dict(), model_filename)
        print(f"Model saved at {model_filename}")
        raise
    finally:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Total training time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"cnn_model_{timestamp}.pth"

    tags = [
        f"batch_size_{BATCH_SIZE}",
        f"epochs_{EPOCHS}",
        f"learning_rate_{LEARNING_RATE}",
        f"sample_rate_{SAMPLE_RATE}",
        f"model_filename_{model_filename}",
    ]
    comments = "Training the model without masking and with updated data processing." 

    wandb.init(project="speech-clarity-prediction", entity="codeexia0", name=model_filename, tags=tags, notes=comments)

    wandb.config.update({
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "sample_rate": SAMPLE_RATE,
    })

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using Device: {device}")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64,
    )

    dataset = CPC1(
        ANNOTATIONS_FILE,
        SPIN_FOLDER,
        SCENES_FOLDER,
        mel_spectrogram,
        SAMPLE_RATE,
        NUM_SAMPLES,
        device,
        max_length=263,
    )

    train_dataloader = create_data_loader(dataset, BATCH_SIZE, shuffle=True)

    cnn_model = CNNNetwork().to(device)
    print(cnn_model)

    # loss_fn = nn.MSELoss()
    loss_fn = nn.MSELoss(reduction='none')  # Ensure loss is not reduced for masking
    optimiser = torch.optim.Adam(cnn_model.parameters(), lr=LEARNING_RATE)

    wandb.watch(cnn_model, log="all", log_freq=100)

    print(f"Model will be saved to {model_filename}")

    try:
        train(cnn_model, train_dataloader, loss_fn, optimiser, device, EPOCHS, model_filename)
    except RuntimeError as e:
        print(f"Runtime Error: {e}")
        print("Ensure the padding length is consistent for all tensors.")
    finally:
        if not os.path.exists(model_filename):
            print("Saving model due to final exit.")
            torch.save(cnn_model.state_dict(), model_filename)
            print(f"Final model saved at {model_filename}")

    wandb.finish()
