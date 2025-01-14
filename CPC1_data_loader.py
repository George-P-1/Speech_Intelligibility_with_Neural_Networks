import json
from torch.utils.data import Dataset
import torchaudio
from pathlib import Path
import torch
import torch.nn.functional as F

class CPC1(Dataset):
    def __init__(self,
                 annotations_file,
                 spin_folder,
                 scenes_folder,
                 transformation,
                 target_sample_rate,
                 num_samples,
                 device,
                 max_length):
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        self.spin_folder = Path(spin_folder)
        self.scenes_folder = Path(scenes_folder)
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.max_length = max_length

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        entry = self.annotations[index]
        spin_path, target_path = self._get_audio_sample_paths(entry)
        correctness = entry['correctness']

        spin_signal, spin_sr = torchaudio.load(spin_path)
        target_signal, target_sr = torchaudio.load(target_path)

        spin_signal = spin_signal.to(self.device)
        target_signal = target_signal.to(self.device)

        spin_signal = self._resample_if_necessary(spin_signal, spin_sr)
        target_signal = self._resample_if_necessary(target_signal, target_sr)

        spin_signal = self._mix_down_if_necessary(spin_signal)
        target_signal = self._mix_down_if_necessary(target_signal)

        # spin_signal = self._adjust_length_to_target(spin_signal, target_signal)
        
        # Remove the first 2 seconds and last 1 second based on CPC1 guidelines
        spin_signal, target_signal = self._cut_timings(spin_signal, target_signal, self.target_sample_rate)

        spin_signal = self.transformation(spin_signal)
        target_signal = self.transformation(target_signal)

        spin_signal, target_signal = self._pad_to_same_length(spin_signal, target_signal, self.max_length)

        mask = self._create_mask(spin_signal, self.max_length)

        spin_signal = torch.cat((spin_signal, target_signal), dim=0)

        return {
            "spin": spin_signal,
            "mask": mask,
            "correctness": correctness / 100.0
        }

    def _resample_if_necessary(self, signal, sample_rate):
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate).to(self.device)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _cut_timings(self, spin_signal, target_signal, sample_rate):
        cut_start = int(2 * sample_rate)  # Remove first 2 seconds
        cut_end = int(1 * sample_rate)   # Remove last 1 second

        if spin_signal.shape[1] > cut_start + cut_end:
            spin_signal = spin_signal[:, cut_start:-cut_end]
        if target_signal.shape[1] > cut_start + cut_end:
            target_signal = target_signal[:, cut_start:-cut_end]

        return spin_signal, target_signal

    def _pad_to_same_length(self, spin_signal, target_signal, max_length):
        spin_time_dim = spin_signal.shape[-1]
        target_time_dim = target_signal.shape[-1]

        max_time_dim = max(spin_time_dim, target_time_dim, max_length)

        spin_padding = (0, max_time_dim - spin_time_dim)
        target_padding = (0, max_time_dim - target_time_dim)

        spin_signal = F.pad(spin_signal, spin_padding)
        target_signal = F.pad(target_signal, target_padding)

        return spin_signal, target_signal
    
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

    def _create_mask(self, spectrogram, max_length):
        time_dim = spectrogram.shape[-1]
        mask = torch.ones(time_dim, dtype=torch.float32, device=spectrogram.device)
        if time_dim < max_length:
            padding = max_length - time_dim
            mask = torch.cat([mask, torch.zeros(padding, dtype=torch.float32, device=spectrogram.device)])
        return mask

    def _get_audio_sample_paths(self, entry):
        spin_path = self.spin_folder / f"{entry['signal']}.wav"
        target_path = self.scenes_folder / f"{entry['scene']}_target_anechoic.wav"
        return spin_path, target_path

    def find_max_spectrogram_length(self):
        max_length = 0
        for i in range(len(self)):
            sample = self[i]
            spin_signal = sample["spin"]

            # Debug: Print the shape of the spectrogram to trace padding
            # print(f"Sample {i + 1}: Spin spectrogram shape: {spin_signal.shape}")

            time_dim = spin_signal.shape[-1]
            if time_dim > max_length:
                max_length = time_dim
                # print(f"New max length found: {max_length}")
        return max_length


if __name__ == "__main__":
    annotations_file = "C:/Users/Codeexia/FinalSemester/CPC1 Data/clarity_CPC1_data.test.v1/clarity_CPC1_data/metadata/CPC1.test.json"
    spin_folder = "C:/Users/Codeexia/FinalSemester/CPC1 Data/clarity_CPC1_data.test.v1/clarity_CPC1_data/clarity_data/HA_outputs/test"
    scenes_folder = "C:/Users/Codeexia/FinalSemester/CPC1 Data/clarity_CPC1_data.test.v1/clarity_CPC1_data/clarity_data/scenes"
    SAMPLE_RATE = 16000
    NUM_SAMPLES = 2421
    MAX_LENGTH = 169 # 263 or 169

    device = "cuda" if torch.cuda.is_available() else "cpu"

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

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

    print(f"Dataset contains {len(dataset)} samples.")

    max_length = dataset.find_max_spectrogram_length()
    print(f"Maximum spectrogram length in dataset: {max_length}")
