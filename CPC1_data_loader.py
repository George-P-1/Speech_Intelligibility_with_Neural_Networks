import json
from torch.utils.data import Dataset
import torchaudio
from pathlib import Path
import torch

# Clarity Prediction Challenge Dataset
class CPC1(Dataset):
    def __init__(self,
                 annotations_file,
                 spin_folder,
                 scenes_folder,
                 transformation,
                 target_sample_rate,
                 num_samples,
                 device,
                 max_length): # max_length is the maximum length of the spectrogram
        # Load JSON annotations
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        self.spin_folder = Path(spin_folder)
        self.scenes_folder = Path(scenes_folder)
        self.device = device
        self.transformation = transformation.to(self.device) # Move the transformation to the device
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.max_length = max_length  # Max length for padding

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        entry = self.annotations[index]
        spin_path, target_path = self._get_audio_sample_paths(entry)
        correctness = entry['correctness']  # Extract correctness value
    
        # # Print out the details
        print(f"Processing Sample Index: {index}")
        print(f"Scene: {entry['scene']}, Correctness: {correctness}")
        print(f"Spin Signal Path: {spin_path}")
        print(f"Target Signal Path: {target_path}")
        
        # Load audio files
        spin_signal, spin_sr = torchaudio.load(spin_path)
        target_signal, target_sr = torchaudio.load(target_path)
        
        # Print signal details after loading
        # print(f"Loaded Spin Signal: {spin_signal.shape}, Sample Rate: {spin_sr}")
        # print(f"Loaded Target Signal: {target_signal.shape}, Sample Rate: {target_sr}")
        spin_signal = spin_signal.to(self.device)
        target_signal = target_signal.to(self.device)
        
        
        # Resample if necessary
        spin_signal = self._resample_if_necessary(spin_signal, spin_sr)
        target_signal = self._resample_if_necessary(target_signal, target_sr)
        
        # print(f"Resampled Spin Signal: {spin_signal.shape} Sample Rate: {self.target_sample_rate}")
        # print(f"Resampled Target Signal: {target_signal.shape} Sample Rate: {self.target_sample_rate}")
        
        # Mix down if necessary
        spin_signal = self._mix_down_if_necessary(spin_signal)
        target_signal = self._mix_down_if_necessary(target_signal)
        
        # print(f"Mix Down Spin Signal: {spin_signal.shape}")
        # print(f"Mix Down Target Signal: {target_signal.shape}")
        
        # Cut if necessary
        # spin_signal = self._cut_if_necessary(spin_signal)
        # target_signal = self._cut_if_necessary(target_signal)
        
        
        # Right pad if necessary
        # spin_signal = self._right_pad_if_necessary(spin_signal)
        # target_signal = self._right_pad_if_necessary(target_signal)
        
        # Adjust length of spin signal to match target signal
        spin_signal = self._adjust_length_to_target(spin_signal, target_signal)
        # print(f"Adjusted Spin Signal: {spin_signal.shape}")
        
        # Apply transformation whihc is mel spectrogram
        spin_signal = self.transformation(spin_signal)
        target_signal = self.transformation(target_signal)
        
        # print(f"Transformed Spin Signal: {spin_signal.shape}")
        # print(f"Transformed Target Signal: {target_signal.shape}")
        
        # Pad the spectrograms to max length
        spin_signal = self._pad_spectrogram(spin_signal, self.max_length)
        target_signal = self._pad_spectrogram(target_signal, self.max_length)
        
        # print(f"Transformed and Padded Spin Signal: {spin_signal.shape}")
        # print(f"Transformed and Padded Target Signal: {target_signal.shape}")

        return {
            "spin": spin_signal,
            # "target": target_signal,
            "correctness": correctness  # Return correctness
        }
        
    def _adjust_length_to_target(self, signal, target_signal):
        
        target_length = target_signal.shape[1]
        signal_length = signal.shape[1]
        
        if signal_length > target_length:
            # Cut the signal to match the target length
            signal = signal[:, :target_length]
        elif signal_length < target_length:
            # Pad the signal to match the target length
            num_missing_samples = target_length - signal_length
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
    
        return signal

        
    def _cut_if_necessary(self, signal):
        # Cut the signal to the desired length
        # signal -> Tensor -> (1, num_samples) 
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal
            
    def _right_pad_if_necessary(self, signal):
        # signal -> Tensor -> (1, 1100) -> (1, 2421) we want to pad the signal to the target number of samples  
        length = signal.shape[1]
        if length < self.num_samples:
            # [1, 1, 1] -> [1, 1, 1, 0, 0]
            num_missing_samples = self.num_samples - length
            last_dim_padding = (0, num_missing_samples) # (1, 2) -> 1 padding before and 2 padding after
            # [1, 1, 1] -> [0,  1, 1, 1, 0, 0]
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal
            
    def _resample_if_necessary(self, signal, sample_rate): 
        # we want to resample only when the sample rate is different from the target sample rate
        if sample_rate != self.target_sample_rate: 
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate).to(self.device)
            signal = resampler(signal)
        return signal
    
    def _mix_down_if_necessary(self, signal): # mix down from 2 channels to 1 channel
        # signal -> (num_channels, num_samples) -> (2, 16000) -> (1, 16000) we want to aggregate the channels into one
        # we want to mix down only when the number of channels is greater than 1
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _get_audio_sample_paths(self, entry):
        # Construct file paths from the entry
        spin_path = self.spin_folder / f"{entry['signal']}.wav"
        target_path = self.scenes_folder / f"{entry['scene']}_target_anechoic.wav"
        return spin_path, target_path
    
    def find_max_spectrogram_length(dataset):
        max_length = 0
        for i in range(len(dataset)):
            sample = dataset[i]  # Retrieve the sample
            spin_signal = sample["spin"]
            time_dim = spin_signal.shape[-1]  # Get the time dimension
            if time_dim > max_length:
                max_length = time_dim
            print(f"Processed sample {i+1}/{len(dataset)}, Current Max Length: {max_length}")
        return max_length
    
    def _pad_spectrogram(self, spectrogram, max_length):
        time_dim = spectrogram.shape[-1]
        if time_dim < max_length:
            padding = (0, max_length - time_dim)
            spectrogram = torch.nn.functional.pad(spectrogram, padding)
        return spectrogram


if __name__ == "__main__":
    # Paths from the Hydra config
# Specify paths directly here
    annotations_file = "C:/Users/Codeexia/FinalSemester/CPC1 Data/clarity_CPC1_data.test.v1/clarity_CPC1_data/metadata/CPC1.test.json"
    spin_folder = "C:/Users/Codeexia/FinalSemester/CPC1 Data/clarity_CPC1_data.test.v1/clarity_CPC1_data/clarity_data/HA_outputs/test"
    scenes_folder = "C:/Users/Codeexia/FinalSemester/CPC1 Data/clarity_CPC1_data.test.v1/clarity_CPC1_data/clarity_data/scenes"
    SAMPLE_RATE = 16000
    NUM_SAMPLES = 2421
    MAX_LENGTH = 263
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using Device: {device}")
    
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64  # Number of mel filterbanks
    )

    # Initialize dataset
    dataset = CPC1(
        annotations_file,
        spin_folder,
        scenes_folder,
        mel_spectrogram,
        SAMPLE_RATE,
        NUM_SAMPLES,
        device,
        MAX_LENGTH
    )
    print(f"There are {len(dataset)} samples in the dataset.")

    # Find the maximum length of the spectrogram
    # max_length = dataset.find_max_spectrogram_length()
    # print(f"Maximum Spectrogram Time Dimension Across Dataset: {max_length}")

    # # Test a few samples
    for i in range(10):
        sample = dataset[i]
        print(f"Sample {i+1}: Spin Spectrogram Shape: {sample['spin'].shape}, Correctness: {sample['correctness']}")

    
    
    
    # print(f"Scene: {sample['spin']}, Correctness: {sample['correctness']}")
    # print(f"Spin Signal Shape: {sample['spin'].shape}, Target Signal Shape: {sample['target'].shape}")