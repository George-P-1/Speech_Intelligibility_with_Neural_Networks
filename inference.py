import torch
from cnn import CNNNetwork
import torchaudio
from CPC1_data_loader import CPC1
from train import SAMPLE_RATE, NUM_SAMPLES, ANNOTATIONS_FILE, SPIN_FOLDER, SCENES_FOLDER


def predict(model, input, target):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        predicted = predictions[0].item()  # Get the predicted value (single float)
        expected = target  # target is already a float
    return predicted, expected


if __name__ == "__main__":
    # Load the model
    CNNNetwork = CNNNetwork()
    state_dict = torch.load("feedforwardnet.pth")
    CNNNetwork.load_state_dict(state_dict)

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
        'cpu',
        max_length=263  # Specify max length for padding
    )
    # Variables for tracking correctness
    correct_count = 0
    total_samples = 0
    total_correctness = 0

    # Loop through first 10 samples
    for idx in range(1000):
        sample = dataset[idx]  # Get the current sample
        input, target = sample["spin"], sample["correctness"]  # Access 'spin' and 'correctness'
        # normally torch expects 3D input, but we have 4D input so we need to unsqueeze it to add a batch dimension
        # 3D input as (num_channels, fr, time)
        input.unsqueeze_(0)  # Unsqueeze to add batch dimension

        # Make an inference
        predicted, expected = predict(CNNNetwork, input, target)

        # Calculate correctness
        print(f"Sample {idx+1}: Predicted: {predicted}, Expected: {expected/100}")

        # Check if the prediction is close enough to the expected correctness
        if abs(predicted - (expected/100)) < 0.05:  # Assuming a tolerance of 5%
            correct_count += 1

        total_samples += 1
        total_correctness += expected  # Sum the expected correctness values

    # Calculate percentage correctness
    correctness_percentage = (correct_count / total_samples) * 100
    avg_correctness = total_correctness / total_samples

    print(f"\nTotal samples: {total_samples}")
    print(f"Correct predictions: {correct_count}")
    print(f"Correctness percentage: {correctness_percentage:.2f}%")
    print(f"Average expected correctness: {avg_correctness:.2f}")