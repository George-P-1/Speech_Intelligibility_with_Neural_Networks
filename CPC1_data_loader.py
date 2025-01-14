import json
from torch.utils.data import Dataset
import torchaudio
from pathlib import Path
import torch
import matplotlib.pyplot as plt
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
        print(f"Loaded Spin Signal: {spin_signal.shape}, Sample Rate: {spin_sr}")
        print(f"Loaded Target Signal: {target_signal.shape}, Sample Rate: {target_sr}")
        spin_signal = spin_signal.to(self.device)
        target_signal = target_signal.to(self.device)
        
        
        # Resample if necessary
        spin_signal = self._resample_if_necessary(spin_signal, spin_sr)
        target_signal = self._resample_if_necessary(target_signal, target_sr)
        
        print(f"Resampled Spin Signal: {spin_signal.shape} Sample Rate: {self.target_sample_rate}")
        print(f"Resampled Target Signal: {target_signal.shape} Sample Rate: {self.target_sample_rate}")
        
        # Mix down if necessary
        spin_signal = self._mix_down_if_necessary(spin_signal)
        target_signal = self._mix_down_if_necessary(target_signal)
        
        print(f"Mix Down Spin Signal: {spin_signal.shape}")
        print(f"Mix Down Target Signal: {target_signal.shape}")
        
        # Plot the signals before cuttin
        self.plot_signals(spin_signal, target_signal, self.target_sample_rate)
        
        # Cut the signals based on the timings (2 seconds from start, 1 second from end)
        spin_signal, target_signal = self._cut_timings(spin_signal, target_signal, self.target_sample_rate)
        # Plot the signals after cutting
        self.plot_signals(spin_signal, target_signal, self.target_sample_rate)
        # Cut if necessary
        # spin_signal = self._cut_if_necessary(spin_signal)
        # target_signal = self._cut_if_necessary(target_signal)
        
        # Right pad if necessary
        # spin_signal = self._right_pad_if_necessary(spin_signal)
        # target_signal = self._right_pad_if_necessary(target_signal)
        
        # Adjust length of spin signal to match target signal
        spin_signal = self._adjust_length_to_target(spin_signal, target_signal)
        print(f"Adjusted Spin Signal: {spin_signal.shape}")
        
        # Apply transformation whihc is mel spectrogram
        spin_signal = self.transformation(spin_signal)
        target_signal = self.transformation(target_signal)
        
        
        # Convert to dB if not already in dB
        spin_signal_db = 10 * torch.log10(spin_signal + 1e-10)  # Adding small value to avoid log(0)
        target_signal_db = 10 * torch.log10(target_signal + 1e-10)

        # Normalize to the range [-1, 0.5]
        spin_signal_db = spin_signal_db / 20.0  # Normalizing from [-20, 10] to [-1, 0.5]
        target_signal_db = target_signal_db / 20.0
        
        # TODO: mean values of spectograms
        # TODO: check if it is in db as input to the model
        # TODO: all spectogram values are between -20 and 10 db and we divide by 20 and we will have -1 and 0.5
        
        # Plot spectrogram
        # plot = self.plot_spin_and_target_spectrogram(spin_signal, target_signal)
        # self.plot_spin_and_target_spectrogram(spin_signal, target_signal, spin_signal_db, target_signal_db)
        
        print(f"Transformed Spin Signal: {spin_signal.shape}")
        # print(f"Transformed Target Signal: {target_signal.shape}")
        
        # Cut first 2 seconds and last 1 second

        # self.plot_spin_and_target_spectrogram(spin_signal_db, target_signal_db)
        
        # Pad the spectrograms to max length
        spin_signal = self._pad_spectrogram(spin_signal, self.max_length)
        target_signal = self._pad_spectrogram(target_signal, self.max_length)
        
        self.plot_spin_and_target_spectrogram(spin_signal, target_signal, spin_signal_db, target_signal_db)
        
        spin_signal = torch.cat((spin_signal, target_signal), dim=0)

        
        print(f"Transformed and Padded Spin Signal: {spin_signal.shape}")
        print(f"Transformed and Padded Target Signal: {target_signal.shape}")

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
    
    def find_max_spectrogram_length(dataset): # 263
        max_length = 0
        for i in range(len(dataset)):
            sample = dataset[i]  # Retrieve the sample
            spin_signal = sample["spin"]
            time_dim = spin_signal.shape[-1]  # Get the time dimension
            if time_dim > max_length:
                max_length = time_dim
            print(f"Processed sample {i+1}/{len(dataset)}, Current Max Length: {max_length}")
        return max_length
    
    
    def _cut_timings(self, spin_signal, target_signal, sample_rate):
        # Debugging: Print the shape of the signals before cutting
        print(f"Before cut: Spin signal shape: {spin_signal.shape}, Target signal shape: {target_signal.shape}")
        
        # Number of samples to remove based on the sample rate (2 seconds for start, 1 second for end)
        cut_start = int(2 * sample_rate)  # 2 seconds from the start
        cut_end = int(1 * sample_rate)    # 1 second from the end
        
        # Cutting the signals
        spin_signal = spin_signal[:, cut_start:-cut_end] if spin_signal.shape[1] > cut_start + cut_end else spin_signal
        target_signal = target_signal[:, cut_start:-cut_end] if target_signal.shape[1] > cut_start + cut_end else target_signal
        
        # Debugging: Print the shape of the signals after cutting
        print(f"After cut: Spin signal shape: {spin_signal.shape}, Target signal shape: {target_signal.shape}")
        
        return spin_signal, target_signal

    def _pad_spectrogram(self, spectrogram, max_length):
        time_dim = spectrogram.shape[-1]
        if time_dim < max_length:
            padding = (0, max_length - time_dim)
            spectrogram = torch.nn.functional.pad(spectrogram, padding)
        return spectrogram
    
    def plot_spin_and_target_spectrogram(self, spin_signal, target_signal, spin_signal_db, target_signal_db):
        fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        
        # Plot raw spin spectrogram (without log)
        im1 = axs[0, 0].imshow(spin_signal[0, :, :].cpu().numpy(), aspect="auto", origin="lower", extent=[0, spin_signal.shape[2], 0, spin_signal.shape[1]])
        axs[0, 0].set_title("Raw Spin Signal Spectrogram (No Log)")
        axs[0, 0].set_xlabel("Time Frames")
        axs[0, 0].set_ylabel("Mel Frequency Bands")
        fig.colorbar(im1, ax=axs[0, 0])

        # Plot log-scaled spin spectrogram (with log)
        im2 = axs[0, 1].imshow(spin_signal_db[0, :, :].cpu().numpy(), aspect="auto", origin="lower", extent=[0, spin_signal_db.shape[2], 0, spin_signal_db.shape[1]])
        axs[0, 1].set_title("Log-Scaled Spin Signal Spectrogram")
        axs[0, 1].set_xlabel("Time Frames")
        axs[0, 1].set_ylabel("Mel Frequency Bands")
        fig.colorbar(im2, ax=axs[0, 1])

        # Plot raw target spectrogram (without log)
        im3 = axs[1, 0].imshow(target_signal[0, :, :].cpu().numpy(), aspect="auto", origin="lower", extent=[0, target_signal.shape[2], 0, target_signal.shape[1]])
        axs[1, 0].set_title("Raw Target Signal Spectrogram (No Log)")
        axs[1, 0].set_xlabel("Time Frames")
        axs[1, 0].set_ylabel("Mel Frequency Bands")
        fig.colorbar(im3, ax=axs[1, 0])

        # Plot log-scaled target spectrogram (with log)
        im4 = axs[1, 1].imshow(target_signal_db[0, :, :].cpu().numpy(), aspect="auto", origin="lower", extent=[0, target_signal_db.shape[2], 0, target_signal_db.shape[1]])
        axs[1, 1].set_title("Log-Scaled Target Signal Spectrogram")
        axs[1, 1].set_xlabel("Time Frames")
        axs[1, 1].set_ylabel("Mel Frequency Bands")
        fig.colorbar(im4, ax=axs[1, 1])

        plt.show()
        
    def plot_signals(self, spin_signal, target_signal, sample_rate):
        """
        Plots the spin and target audio signals over time.

        Args:
            spin_signal (Tensor): The spin audio signal.
            target_signal (Tensor): The target audio signal.
            sample_rate (int): The sample rate of the audio signals.
        """
        time_spin = torch.arange(spin_signal.shape[1]) / sample_rate
        time_target = torch.arange(target_signal.shape[1]) / sample_rate

        plt.figure(figsize=(12, 6))

        # Plot spin signal
        plt.subplot(2, 1, 1)
        plt.plot(time_spin.cpu(), spin_signal[0].cpu(), label="Spin Signal", color="blue")
        plt.title("Spin Signal Over Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend()

        # Plot target signal
        plt.subplot(2, 1, 2)
        plt.plot(time_target.cpu(), target_signal[0].cpu(), label="Target Signal", color="orange")
        plt.title("Target Signal Over Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend()

        plt.tight_layout()
        plt.show()






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
    for i in range(1):
        sample = dataset[i+10]
        print(f"Sample {i+1}: Spin Spectrogram Shape: {sample['spin'].shape}, Correctness: {sample['correctness']}")
        
        
        

    
    
    
    # print(f"Scene: {sample['spin']}, Correctness: {sample['correctness']}")
    # print(f"Spin Signal Shape: {sample['spin'].shape}, Target Signal Shape: {sample['target'].shape}")
    
    
    #TODO: Add masking for zeros in the spectrogram so we dont want to include in the training
    
    #TODO: i can reject fist 2 s and last 1 s for noisey part